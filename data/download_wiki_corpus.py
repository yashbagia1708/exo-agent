"""
Download the Wikipedia articles for retrieval -> data/wiki_corpus/*.json

    python3 data/download_wiki_corpus.py

Covers what SQL can't answer: how detection methods work, what terms
mean, a few famous systems. Each article is split into ~1200-character
chunks; load_wiki_to_postgres.py embeds them.

Two things that bit us (FINDINGS.md, 2026-06-22):
  - some titles are redirects ("Transit method" -> "Methods of detecting
    exoplanets") and came back empty, so we pass redirects=1
  - Wikipedia rate-limits (429), so back off exponentially on 429
"""

import json
import pathlib
import time
import urllib.error
import urllib.parse
import urllib.request

OUTPUT_DIR = pathlib.Path(__file__).parent / "wiki_corpus"
WIKI_FULL_TEXT_BASE = "https://en.wikipedia.org/w/api.php"

# detection methods, concepts, and missions/systems people ask about
# (some are redirects; fetch_full_text follows them)
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
BASE_RETRY_DELAY = 5  # seconds; doubles each time we get a 429
REQUEST_GAP = 1.5  # pause between requests to stay under the rate limit


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
                delay *= 2  # 5s, 10s, 20s...
            elif attempt < MAX_RETRIES:
                time.sleep(BASE_RETRY_DELAY)
        except urllib.error.URLError as e:
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(BASE_RETRY_DELAY)

    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {last_error}")


def fetch_full_text(title: str) -> tuple[str, str]:
    """
    -> (resolved title, plain text). Follows redirects, so
    "Transit method" comes back as "Methods of detecting exoplanets".
    Cite the resolved title; that's the page the text came from.
    """
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "explaintext": "1",
            "exsectionformat": "plain",
            "redirects": "1",  # without this a redirect returns an empty extract
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
    """Group paragraphs into ~max_chars chunks, each starting with the last `overlap` chars of the previous one."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) <= max_chars:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            # new chunk starts with the tail of the last one
            overlap_text = current[-overlap:] if current else ""
            current = f"{overlap_text}\n\n{para}".strip()

    if current:
        chunks.append(current)

    return chunks


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    manifest = []
    seen_resolved_titles = set()  # two titles can redirect to the same article

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
            print("  SKIPPED: empty extract even after redirect resolution")
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
