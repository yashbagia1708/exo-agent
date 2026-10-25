"""
Download confirmed exoplanet data from the NASA Exoplanet Archive.

Source: PSCompPars (Planetary Systems Composite Parameters) table —
one row per confirmed planet, statistically de-duplicated by NASA.
Docs: https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html
TAP service: https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html

This script makes a single, polite, read-only GET request to NASA's public
TAP sync endpoint and saves the raw CSV. No auth required — it's a public
archive.

If this script fails (network blocked, NASA service hiccup, etc.), you can
always get the same data manually:
  1. Visit https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=PSCompPars
  2. Click "Download Table" -> CSV
  3. Save as data/raw_pscomppars.csv
"""

import csv
import io
import pathlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

TAP_BASE = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# Columns we actually use downstream (host_stars + planets schema).
# Names verified against the archive's official column documentation:
# https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html
COLUMNS = [
    "pl_name",
    "hostname",
    "discoverymethod",
    "disc_year",
    "disc_facility",
    "pl_orbper",
    "pl_rade",
    "pl_bmasse",
    "pl_eqt",
    "pl_orbeccen",
    "pl_insol",
    "tran_flag",
    "st_spectype",
    "st_teff",
    "st_rad",
    "st_mass",
    "st_met",
    "sy_dist",
    "sy_snum",
    "sy_pnum",
]

OUTPUT_PATH = pathlib.Path(__file__).parent / "raw_pscomppars.csv"

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5


def build_query() -> str:
    select_clause = ",".join(COLUMNS)
    adql = f"select {select_clause} from pscomppars where pl_name is not null"
    return adql


def fetch_csv() -> str:
    query = build_query()
    params = urllib.parse.urlencode({"query": query, "format": "csv"})
    url = f"{TAP_BASE}?{params}"
    print(f"Requesting: {url}\n")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "exo-agent-portfolio-project/1.0 (personal, non-commercial)"},
    )

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}")
                return resp.read().decode("utf-8")
        except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError) as e:
            last_error = e
            print(f"Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)

    raise RuntimeError(f"All {MAX_RETRIES} attempts failed. Last error: {last_error}")


def main():
    try:
        csv_text = fetch_csv()
    except Exception as e:
        print(f"\nERROR: failed to download data from NASA Exoplanet Archive: {e}", file=sys.stderr)
        print(
            "\nFallback: download manually from "
            "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=PSCompPars "
            f"(click 'Download Table' -> CSV) and save as {OUTPUT_PATH}",
            file=sys.stderr,
        )
        sys.exit(1)

    reader = csv.reader(io.StringIO(csv_text))
    rows = list(reader)

    if len(rows) < 2:
        print("ERROR: response had no data rows. Check the query syntax.", file=sys.stderr)
        sys.exit(1)

    OUTPUT_PATH.write_text(csv_text, encoding="utf-8")
    print(f"Saved {len(rows) - 1} planet records to {OUTPUT_PATH}")
    print(f"Columns: {rows[0]}")


if __name__ == "__main__":
    main()
