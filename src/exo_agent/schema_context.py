"""
Human-readable description of the real database schema, handed to the LLM
as context before it writes any SQL. This is what stops the agent from
inventing plausible-sounding columns that don't exist (e.g. a fictional
"surface_temperature" column) -- it can only reference what's actually here.

Source of truth: the CREATE TABLE statements in data/load_to_postgres.py.
If that schema changes, update this description to match.
"""

SCHEMA_DESCRIPTION = """
You are querying a PostgreSQL database about confirmed exoplanets, sourced
from NASA's Exoplanet Archive (PSCompPars table). There are two tables:

TABLE: host_stars
  One row per unique host star.
  Columns:
    hostname    TEXT PRIMARY KEY  -- star name, e.g. 'Kepler-80'
    st_spectype TEXT              -- spectral type, e.g. 'K5'. Often NULL.
    st_teff     REAL               -- effective temperature, Kelvin
    st_rad      REAL               -- stellar radius, in solar radii
    st_mass     REAL               -- stellar mass, in solar masses
    st_met      REAL               -- metallicity, dex
    sy_dist     REAL               -- distance from Earth, in parsecs
    sy_snum     INTEGER            -- number of stars in this system
    sy_pnum     INTEGER            -- number of known planets in this system

TABLE: planets
  One row per confirmed planet. Foreign key hostname -> host_stars.hostname
  Columns:
    pl_name         TEXT PRIMARY KEY   -- planet name, e.g. 'Kepler-80 d'
    hostname        TEXT               -- FK to host_stars
    discoverymethod TEXT               -- e.g. 'Transit', 'Radial Velocity', 'Imaging'
    disc_year       INTEGER            -- year of discovery. Can be NULL (1 row in ~6300).
    disc_facility   TEXT               -- observatory/mission name, e.g. 'Kepler', 'TESS'
    pl_orbper       REAL               -- orbital period, in days
    pl_rade         REAL               -- planet radius, in Earth radii
    pl_bmasse       REAL               -- planet mass, in Earth masses
    pl_eqt          REAL               -- equilibrium temperature, Kelvin
    pl_orbeccen     REAL               -- orbital eccentricity (0 = circular)
    pl_insol        REAL               -- insolation flux, relative to Earth (1.0 = Earth's)
    tran_flag       INTEGER            -- 1 if detected via transit, 0 otherwise

IMPORTANT LIMITS:
- There is NO column for surface temperature, habitability confirmation, or
  presence of life. pl_eqt (equilibrium temperature) and pl_insol are the
  closest proxies for "Earth-like conditions" -- do not claim more certainty
  than these proxies support.
- There is NO column for a measured radial velocity VALUE (e.g. in m/s).
  `discoverymethod` can equal the STRING 'Radial Velocity', but that only
  records WHICH detection technique found the planet -- it is not a
  numeric measurement of velocity and cannot be ordered, compared, or used
  to answer "highest/lowest radial velocity" questions. If asked something
  like "which planet has the highest radial velocity" or "how does this
  planet's radial velocity compare to Earth's", do not substitute a
  different column (orbital period, mass, etc.) as if it answered the
  question -- write a query that returns the relevant rows you CAN
  describe (e.g. which planets were found via the Radial Velocity method)
  and clearly note in your reasoning that no actual velocity measurement
  exists in this dataset, so the final answer can caveat this honestly
  instead of silently answering a different question.
- Many columns have real missing measurements (NULL), not zero. Only add an
  IS NOT NULL filter on a column when that specific column is actually used
  in the SELECT, WHERE, GROUP BY, or ORDER BY clause -- never add filters on
  columns the question doesn't touch. For example, a question asking for a
  total count of all planets should count every row in `planets`; do not
  filter on disc_year unless the question is specifically about discovery
  year (e.g. trends "by year", "since <year>", "per year").
- CRITICAL for "highest/lowest/most/least/largest/smallest" questions:
  PostgreSQL sorts NULLs as larger than any value by default, so
  `ORDER BY some_column DESC LIMIT 1` can return a row where some_column is
  NULL -- silently giving a meaningless "highest" answer. Whenever you write
  ORDER BY ... LIMIT on a column to find an extreme value (max/min/top/
  bottom), you MUST add `WHERE that_column IS NOT NULL` so the result is an
  actual measured value, not a missing one.
- SQL clause order is fixed: SELECT ... FROM ... WHERE ... GROUP BY ...
  ORDER BY ... LIMIT. WHERE always comes before ORDER BY and LIMIT, never
  after. Double-check clause order before returning a query.
- CRITICAL for matching a named planet or star (e.g. "Proxima Centauri b",
  "TRAPPIST-1", "Kepler-22"): real catalogue names in this database are
  often abbreviated or formatted differently than common usage -- for
  example, the star commonly called "Proxima Centauri" is stored as
  exactly "Proxima Cen" in this database, not the full name. NEVER use an
  exact match (`= 'Some Name'` or `LIKE 'Some Name'` with no wildcards) on
  a guessed full name, since this can silently match zero rows if the
  stored value is abbreviated or formatted differently -- and a silent
  zero-row result looks identical to "this object doesn't exist", which
  is misleading. Instead, use a case-insensitive partial match:
  `WHERE hostname ILIKE '%proxima%'` or `WHERE pl_name ILIKE '%proxima%'`
  -- matching on a distinctive substring of the name, not the full guessed
  string. If a search like this returns zero rows, that's a much stronger
  signal the object genuinely isn't in the dataset, since partial
  matching already accounts for naming/formatting differences.
- CRITICAL for "closer to/farther from the host star" questions:
  pl_insol (insolation flux relative to Earth) and pl_orbper (orbital
  period in days) move in OPPOSITE directions relative to distance from
  the star. A planet CLOSER to its star receives MORE radiation (HIGHER
  pl_insol) and completes an orbit FASTER (LOWER pl_orbper). Worked
  example: "planets closer to their host star than Earth" means planets
  receiving MORE radiation than Earth does, which is `pl_insol > 1.0`
  (NOT `< 1.0` -- Earth's own insolation is defined as 1.0, so a smaller
  value means LESS radiation, meaning FARTHER away, not closer). If using
  pl_orbper instead, "closer than Earth" means a SHORTER period than
  Earth's ~365 days, i.e. `pl_orbper < 365`. Pick ONE of these two
  columns for a "distance from star" question and use it consistently --
  do not filter on one column while ordering by the other, since they can
  disagree for individual planets (e.g. a low-mass star can give a
  close-in planet a deceptively low insolation despite a short period).
- Only write SELECT statements. Never modify data.
"""


def get_schema_context() -> str:
    return SCHEMA_DESCRIPTION
