# exo-agent — data step

## Run this on your own machine (needs internet access)

```bash
cd exo-agent
python3 data/download_data.py
```

Expected output:
```
Requesting: https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=...

Saved 5800+ planet records to data/raw_pscomppars.csv
Columns: ['pl_name', 'hostname', 'discoverymethod', 'disc_year', ...]
```

This calls NASA's public Exoplanet Archive TAP service directly — no API key,
no account needed. It's a single GET request to a public government archive.

## If it fails

The script retries 3 times automatically. If it still fails (rare — usually a
network/firewall issue on your end, not NASA's), download manually instead:

1. Go to https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=PSCompPars
2. Click **Download Table** → **CSV**
3. Save the file as `data/raw_pscomppars.csv` in this project folder

Either path gets you the same data — one row per confirmed exoplanet, ~20
columns covering discovery method/year, orbital and physical parameters, and
host star properties.

## Once you have the file

You already have `data/raw_pscomppars.csv` — great. Next step:

### Set up Postgres (one-time, local)

If you don't already have Postgres running locally:

**macOS (Homebrew):**
```bash
brew install postgresql@16
brew services start postgresql@16
```

**Linux (Debian/Ubuntu):**
```bash
sudo apt install postgresql
sudo systemctl start postgresql
```

Then create the database and user:
```bash
createdb exoagent
createuser exoagent --pwprompt   # enter password: exoagent (or your own — update DB_DSN in load_to_postgres.py if you change it)
psql exoagent -c "GRANT ALL ON SCHEMA public TO exoagent;"
```

### Install Python dependencies

```bash
pip install -r requirements.txt
```

### Run the loader

```bash
python3 data/load_to_postgres.py
```

This will:
1. Read `data/raw_pscomppars.csv`
2. Split it into two tables: `host_stars` (4,716 unique stars, aggregated) and `planets` (6,298 planets)
3. Print a summary of any data-quality conflicts found (some host stars have slightly different stellar parameter values across their planet rows — a known property of NASA's source table, not a bug)
4. Write `data/data_quality_notes.md` documenting exactly what was found
5. Create the schema in Postgres and load both tables

Expected output ends with something like:
```
Done. host_stars: 4716 rows, planets: 6298 rows.
```

### Verify it worked

```bash
psql exoagent -c "SELECT count(*) FROM planets;"
psql exoagent -c "SELECT * FROM host_stars LIMIT 5;"
```

