# Data Quality Notes

Generated automatically by `load_to_postgres.py`. This documents a known
limitation in how host-star attributes were derived from the source data,
in the interest of transparency about the pipeline's assumptions.

## Why this exists

The source table (NASA's PSCompPars) resolves each individual parameter
(e.g. stellar mass, stellar temperature) independently, choosing the best
available literature reference *per parameter* -- not per star. As a result,
two planets that orbit the same host star can show slightly different
values for the same stellar property in the raw data, because that
property happened to be sourced from different publications for each row.
This is a documented property of NASA's own table design, not an error
introduced by this project.

## How this loader handles it

For each host star, numeric stellar fields (temperature, radius, mass,
metallicity, distance) are averaged across all of that star's planet rows.
Categorical fields (spectral type) use the most common non-null value.

Total unique host stars: 4716
Total planets: 6298

## Fields with cross-row disagreement

Number of host stars where source rows disagreed on this field before averaging:

- `st_mass`: 197 hosts (4.2%)
- `st_rad`: 191 hosts (4.1%)
- `st_teff`: 191 hosts (4.1%)
- `st_met`: 153 hosts (3.2%)
- `st_spectype`: 6 hosts (0.1%)

## Practical impact

These are small averaging effects on stellar parameters for multi-planet
systems, not errors in planet-level data (orbital period, radius, mass,
discovery method/year are taken directly from each planet's own row and
are not affected by this averaging).