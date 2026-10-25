"""
Download the planet table from NASA's Exoplanet Archive -> data/raw_pscomppars.csv

    python3 data/download_data.py

PSCompPars: one row per confirmed planet. Public TAP API, no key needed.
Columns: https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html

If it fails, download by hand: open the PSCompPars table at
https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=PSCompPars
then Download Table -> CSV, and save it as data/raw_pscomppars.csv
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

# only what load_to_postgres.py uses (names checked against the column docs above)
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
