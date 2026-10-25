"""
Downloads a curated set of Wikipedia articles covering exoplanet detection
methods and astronomy concepts -- the unstructured-knowledge complement
to the structured planets/host_stars database.

Why these specific pages: the structured DB can answer "how many" and
"which" questions, but has no way to answer "how does the transit method
actually work" or "what is a spectral type" -- conceptual questions a
real user would naturally ask alongside data questions. This corpus
exists specifically to cover that gap, not to duplicate what SQL already
answers well.

Uses Wikipedia's public REST API (no auth needed, no scraping of raw
HTML) to fetch clean plain-text extracts.
API docs: https://en.wikipedia.org/api/rest_v1/

Handles two real issues found during development (see FINDINGS.md):
  1. Some titles (e.g. "Transit method") are REDIRECTS to a section of a
     bigger article (here, "Methods of detecting exoplanets") rather than
     their own standalone page -- the API call succeeds with no error,
     but returns an empty extract, which looks like a silent failure if
     not handled explicitly. Fixed by passing `redirects=1` and following
     MediaWiki's own redirect resolution rather than guessing replacement
     titles by hand.
  2. Fetching many pages back-to-back can trigger Wikipedia's rate
     limiting (HTTP 429). A fixed retry delay isn't enough once you're
     actually being throttled -- fixed with real exponential backoff
     specifically on 429 responses, plus a longer baseline delay between
     requests.
"""

import json
import pathlib
import time
import urllib.error
import urllib.parse
import urllib.request

OUTPUT_DIR = pathlib.Path(__file__).parent / "wiki_corpus"
WIKI_FULL_TEXT_BASE = "https://en.wikipedia.org/w/api.php"

# Curated pages: detection methods, key concepts, and a few notable
# missions/systems that come up constantly in exoplanet questions but
# have no representation in the structured schema at all.
#
# Some of these (e.g. "Transit method") are known to be redirects rather
# than standalone articles -- that's fine, fetch_full_text() below
# resolves redirects automatically via MediaWiki's own `redirects=1`
# parameter rather than needing the "real" title hand-picked in advance.
PAGES = [
    "Exoplanet",
    "Methods of detecting exoplanets",
    "Transit method",
    "Radial velocity",
    "Direct imaging",
    "Gravitational microlensing",
    "Astrometry",
    "Circumstellar habitable zone",
    "Stellar classification",
    "Equilibrium temperature",
    "Kepler space telescope",
    "Transiting Exoplanet Survey Satellite",
    "TRAPPIST-1",
    "Proxima Centauri b",
    "Hot Jupiter",
    "Super-Earth",
    "Planetary habitability",
    "Biosignature",
    "Doppler spectroscopy",
    "NASA Exoplanet Archive",
]

MAX_RETRIES = 5
BASE_RETRY_DELAY = 5  # seconds; doubles on each retry for real backoff
REQUEST_GAP = 1.5  # polite delay between successful requests, longer than before


def _fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "exo-agent-portfolio-project/1.0"})
    last_error = None
    delay = BASE_RETRY_DELAY

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_error = e
            if e.code == 429:
                print(f"    (rate limited, waiting {delay}s before retry {attempt}/{MAX_RETRIES})")
                time.sleep(delay)
                delay *= 2  # exponential backoff specifically for rate limiting
            elif attempt < MAX_RETRIES:
                time.sleep(BASE_RETRY_DELAY)
        except urllib.error.URLError as e:
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(BASE_RETRY_DELAY)

    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {last_error}")


def fetch_full_text(title: str) -> tuple[str, str]:
    """
    Fetch the full plain-text extract of a Wikipedia article via the
    MediaWiki API's TextExtracts, following redirects automatically.

    Returns (resolved_title, extract_text). resolved_title may differ
    from the input title if it was a redirect (e.g. "Transit method"
    resolves to "Methods of detecting exoplanets") -- callers should use
    the resolved title for display/citation, since that's the actual
    page the content came from.
    """
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "explaintext": "1",
            "exsectionformat": "plain",
            "redirects": "1",  # follow redirects instead of returning an empty page
        }
    )
    url = f"{WIKI_FULL_TEXT_BASE}?{params}"
    data = _fetch_json(url)

    pages = data.get("query", {}).get("pages", {})
    for page_id, page_data in pages.items():
        if page_id == "-1":
            raise ValueError(f"Page not found: {title}")
        resolved_title = page_data.get("title", title)
        return resolved_title, page_data.get("extract", "")

    raise ValueError(f"Unexpected response shape for: {title}")


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> list[str]:
    """
    Simple paragraph-aware chunking: split on double newlines first (real
    paragraph boundaries), then merge/split to stay near max_chars with a
    small overlap so a fact split across a boundary isn't lost entirely.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) <= max_chars:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            # start new chunk, carrying a small overlap from the end of the previous one
            overlap_text = current[-overlap:] if current else ""
            current = f"{overlap_text}\n\n{para}".strip()

    if current:
        chunks.append(current)

    return chunks


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    manifest = []
    seen_resolved_titles = set()  # avoid saving the same target article twice under different filenames

    for title in PAGES:
        print(f"Fetching: {title}")
        try:
            resolved_title, text = fetch_full_text(title)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        if resolved_title != title:
            print(f"  (redirected to: {resolved_title})")

        if resolved_title in seen_resolved_titles:
            print(f"  SKIPPED: already have '{resolved_title}' from an earlier entry in PAGES")
            time.sleep(REQUEST_GAP)
            continue

        if not text.strip():
            print(f"  SKIPPED: empty extract even after redirect resolution")
            time.sleep(REQUEST_GAP)
            continue

        chunks = chunk_text(text)
        safe_filename = resolved_title.lower().replace(" ", "_").replace("/", "_") + ".json"
        out_path = OUTPUT_DIR / safe_filename

        record = {
            "title": resolved_title,
            "source_url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(resolved_title.replace(' ', '_'))}",
            "chunks": chunks,
        }
        out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        seen_resolved_titles.add(resolved_title)
        manifest.append({"title": resolved_title, "file": safe_filename, "num_chunks": len(chunks)})
        print(f"  Saved {len(chunks)} chunk(s) -> {out_path.name}")

        time.sleep(REQUEST_GAP)

    manifest_path = OUTPUT_DIR / "_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total_chunks = sum(m["num_chunks"] for m in manifest)
    print(f"\nDone. {len(manifest)} unique article(s) saved from {len(PAGES)} requested titles, "
          f"{total_chunks} total chunks.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
