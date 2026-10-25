"""
Load data/raw_pscomppars.csv into Postgres as two tables:

    host_stars  one row per star (star columns averaged over its planet rows)
    planets     one row per planet, hostname -> host_stars

    python3 data/load_to_postgres.py     (drops and recreates both tables)

The CSV repeats each star's properties on every one of its planets, and
those copies don't always agree: PSCompPars picks the best literature
value per parameter, per row, so two planets of Kepler-80 can list
different star masses. We average them and write how often it happens to
data/data_quality_notes.md.
"""

import csv
import pathlib
import statistics
import sys
from collections import Counter, defaultdict

import psycopg2

from exo_agent import config

DATA_PATH = pathlib.Path(__file__).parent / "raw_pscomppars.csv"
NOTES_PATH = pathlib.Path(__file__).parent / "data_quality_notes.md"

# star numbers -> averaged across the star's planet rows
STAR_NUMERIC_FIELDS = ["st_teff", "st_rad", "st_mass", "st_met", "sy_dist"]
# star category -> most common non-null value
STAR_CATEGORICAL_FIELDS = ["st_spectype"]
# should be identical on every row for a star; we just check
STAR_CONSISTENT_FIELDS = ["sy_snum", "sy_pnum"]

# planet columns, copied as-is
PLANET_FIELDS = [
    "pl_name", "hostname", "discoverymethod", "disc_year", "disc_facility",
    "pl_orbper", "pl_rade", "pl_bmasse", "pl_eqt", "pl_orbeccen", "pl_insol",
    "tran_flag",
]

DB_DSN = config.DB_DSN  # same database as the agent (.env)

SCHEMA_SQL = """
DROP TABLE IF EXISTS planets;
DROP TABLE IF EXISTS host_stars;

CREATE TABLE host_stars (
    hostname        TEXT PRIMARY KEY,
    st_spectype     TEXT,
    st_teff         REAL,
    st_rad          REAL,
    st_mass         REAL,
    st_met          REAL,
    sy_dist         REAL,
    sy_snum         INTEGER,
    sy_pnum         INTEGER
);

CREATE TABLE planets (
    pl_name         TEXT PRIMARY KEY,
    hostname        TEXT REFERENCES host_stars(hostname),
    discoverymethod TEXT,
    disc_year       INTEGER,
    disc_facility   TEXT,
    pl_orbper       REAL,
    pl_rade         REAL,
    pl_bmasse       REAL,
    pl_eqt          REAL,
    pl_orbeccen     REAL,
    pl_insol        REAL,
    tran_flag       INTEGER
);

CREATE INDEX idx_planets_hostname ON planets(hostname);
CREATE INDEX idx_planets_discyear ON planets(disc_year);
CREATE INDEX idx_planets_discmethod ON planets(discoverymethod);
"""


def to_float_or_none(val):
    if val is None or val == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def to_int_or_none(val):
    f = to_float_or_none(val)
    return int(f) if f is not None else None


def load_rows():
    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found. Run download_data.py first.", file=sys.stderr)
        sys.exit(1)

    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loaded {len(rows)} raw rows from {DATA_PATH}")
    return rows


def build_host_stars(rows):
    """Planet rows -> one record per hostname, plus a count of disagreeing fields."""
    by_host = defaultdict(list)
    for row in rows:
        by_host[row["hostname"]].append(row)

    host_records = []
    conflict_counts = Counter()  # field -> how many stars disagree on it

    for hostname, host_rows in by_host.items():
        record = {"hostname": hostname}

        # numbers: average the non-null values, note if they differ
        for field in STAR_NUMERIC_FIELDS:
            values = [to_float_or_none(r[field]) for r in host_rows]
            non_null = [v for v in values if v is not None]
            if not non_null:
                record[field] = None
                continue
            distinct = set(round(v, 6) for v in non_null)
            if len(distinct) > 1:
                conflict_counts[field] += 1
            record[field] = round(statistics.mean(non_null), 4)

        # categories: most common non-null value
        for field in STAR_CATEGORICAL_FIELDS:
            values = [r[field] for r in host_rows if r[field]]
            if not values:
                record[field] = None
                continue
            distinct = set(values)
            if len(distinct) > 1:
                conflict_counts[field] += 1
            record[field] = Counter(values).most_common(1)[0][0]

        # should already agree: take the first, note if not
        for field in STAR_CONSISTENT_FIELDS:
            values = [to_int_or_none(r[field]) for r in host_rows if r[field]]
            if not values:
                record[field] = None
                continue
            distinct = set(values)
            if len(distinct) > 1:
                conflict_counts[field] += 1
            record[field] = values[0]

        host_records.append(record)

    return host_records, conflict_counts


def build_planets(rows):
    planet_records = []
    for row in rows:
        record = {
            "pl_name": row["pl_name"],
            "hostname": row["hostname"],
            "discoverymethod": row["discoverymethod"] or None,
            "disc_year": to_int_or_none(row["disc_year"]),
            "disc_facility": row["disc_facility"] or None,
            "pl_orbper": to_float_or_none(row["pl_orbper"]),
            "pl_rade": to_float_or_none(row["pl_rade"]),
            "pl_bmasse": to_float_or_none(row["pl_bmasse"]),
            "pl_eqt": to_float_or_none(row["pl_eqt"]),
            "pl_orbeccen": to_float_or_none(row["pl_orbeccen"]),
            "pl_insol": to_float_or_none(row["pl_insol"]),
            "tran_flag": to_int_or_none(row["tran_flag"]),
        }
        planet_records.append(record)
    return planet_records


def write_quality_notes(conflict_counts, total_hosts, total_planets):
    lines = [
        "# Data Quality Notes",
        "",
        "Generated automatically by `load_to_postgres.py`. This documents a known",
        "limitation in how host-star attributes were derived from the source data,",
        "in the interest of transparency about the pipeline's assumptions.",
        "",
        "## Why this exists",
        "",
        "The source table (NASA's PSCompPars) resolves each individual parameter",
        "(e.g. stellar mass, stellar temperature) independently, choosing the best",
        "available literature reference *per parameter* -- not per star. As a result,",
        "two planets that orbit the same host star can show slightly different",
        "values for the same stellar property in the raw data, because that",
        "property happened to be sourced from different publications for each row.",
        "This is a documented property of NASA's own table design, not an error",
        "introduced by this project.",
        "",
        "## How this loader handles it",
        "",
        "For each host star, numeric stellar fields (temperature, radius, mass,",
        "metallicity, distance) are averaged across all of that star's planet rows.",
        "Categorical fields (spectral type) use the most common non-null value.",
        "",
        f"Total unique host stars: {total_hosts}",
        f"Total planets: {total_planets}",
        "",
        "## Fields with cross-row disagreement",
        "",
        "Number of host stars where source rows disagreed on this field before averaging:",
        "",
    ]
    if conflict_counts:
        for field, count in sorted(conflict_counts.items(), key=lambda x: -x[1]):
            pct = 100 * count / total_hosts
            lines.append(f"- `{field}`: {count} hosts ({pct:.1f}%)")
    else:
        lines.append("- None found.")

    lines += [
        "",
        "## Practical impact",
        "",
        "These are small averaging effects on stellar parameters for multi-planet",
        "systems, not errors in planet-level data (orbital period, radius, mass,",
        "discovery method/year are taken directly from each planet's own row and",
        "are not affected by this averaging).",
    ]

    NOTES_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote data quality notes to {NOTES_PATH}")


def main():
    rows = load_rows()

    host_records, conflict_counts = build_host_stars(rows)
    planet_records = build_planets(rows)

    print(f"Built {len(host_records)} unique host stars")
    print(f"Built {len(planet_records)} planet records")

    if conflict_counts:
        print("\nHost-star field disagreements found (resolved by averaging):")
        for field, count in sorted(conflict_counts.items(), key=lambda x: -x[1]):
            print(f"  {field}: {count} hosts")
    else:
        print("\nNo host-star field disagreements found.")

    write_quality_notes(conflict_counts, len(host_records), len(planet_records))

    try:
        conn = psycopg2.connect(DB_DSN)
    except psycopg2.OperationalError as e:
        print(f"\nERROR: could not connect to Postgres: {e}", file=sys.stderr)
        print(
            "Make sure Postgres is running and a database/user matching DB_DSN exists.\n"
            "Easiest fix: `make db-up` (Docker). Or set it up by hand:\n"
            "  createdb exoagent\n"
            "  createuser exoagent --pwprompt   # set password 'exoagent'\n"
            "  psql exoagent -c \"GRANT ALL ON SCHEMA public TO exoagent;\"",
            file=sys.stderr,
        )
        sys.exit(1)

    cur = conn.cursor()
    print("\nCreating schema...")
    cur.execute(SCHEMA_SQL)

    print("Inserting host_stars...")
    for r in host_records:
        cur.execute(
            """
            INSERT INTO host_stars (hostname, st_spectype, st_teff, st_rad, st_mass, st_met, sy_dist, sy_snum, sy_pnum)
            VALUES (%(hostname)s, %(st_spectype)s, %(st_teff)s, %(st_rad)s, %(st_mass)s, %(st_met)s, %(sy_dist)s, %(sy_snum)s, %(sy_pnum)s)
            """,
            r,
        )

    print("Inserting planets...")
    for r in planet_records:
        cur.execute(
            """
            INSERT INTO planets (pl_name, hostname, discoverymethod, disc_year, disc_facility,
                                  pl_orbper, pl_rade, pl_bmasse, pl_eqt, pl_orbeccen, pl_insol, tran_flag)
            VALUES (%(pl_name)s, %(hostname)s, %(discoverymethod)s, %(disc_year)s, %(disc_facility)s,
                    %(pl_orbper)s, %(pl_rade)s, %(pl_bmasse)s, %(pl_eqt)s, %(pl_orbeccen)s, %(pl_insol)s, %(tran_flag)s)
            """,
            r,
        )

    conn.commit()

    cur.execute("SELECT count(*) FROM host_stars;")
    host_count = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM planets;")
    planet_count = cur.fetchone()[0]

    print(f"\nDone. host_stars: {host_count} rows, planets: {planet_count} rows.")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
