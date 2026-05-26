# Velocity Structure Analysis Tool (VSAT)

[![DOI](https://zenodo.org/badge/122964852.svg)](https://zenodo.org/badge/latestdoi/122964852)

VSAT analyzes velocity structure in star clusters and associations following
the method described in [Arnold & Goodwin (2018)](https://doi.org/10.1093/mnras/sty3409).
For every pair of stars, VSAT calculates their separation (`dr`) and velocity
difference (`dv`), sorts those pairs into `dr` bins, and measures the mean
`dv` in each bin.

Please cite Arnold & Goodwin (2018) if this software is used in research
output.

## What changed in this Pythonized version

- Runs on modern Python without the old f2py/Fortran extension at runtime.
- Uses NumPy vectorized calculations for pair distances, velocity differences,
  bin means, and bin errors.
- Reads CSV files or pandas DataFrames directly.
- Infers common Cartesian and Gaia-style column names, with explicit overrides.
- Provides a reusable Python API (`main.analyze`) and a CLI.
- Includes automated tests.

The historical `fort_subroutines.f90` source remains for provenance, but the
modern code path does not import Fortran or require a compiled extension.

## Installation

```bash
python -m pip install -e .
```

For test tooling:

```bash
python -m pip install -e ".[test]"
```

## Command-line use

Run against a CSV with inferred columns:

```bash
python main.py catalog.csv
```

For Gaia-style projected quantities, VSAT will infer `ra`, `dec`, `pmra`,
`pmdec`, and matching error columns such as `pmra_error` and `pmdec_error`.
For Cartesian catalogs, it will infer `x/y/z` and `vx/vy/vz` where present.

Use explicit columns when a catalog has different names:

```bash
python main.py catalog.csv \
  --position-cols x,y,z \
  --velocity-cols vx,vy,vz \
  --velocity-error-cols vx_error,vy_error,vz_error
```

Calculate without displaying plots:

```bash
python main.py catalog.csv --no-show
```

Run the built-in linear-structure demo:

```bash
python main.py --demo
```

## Python API

```python
import pandas as pd
from main import analyze

df = pd.read_csv("catalog.csv")

result = analyze(
    df,
    position_cols=["ra", "dec"],
    velocity_cols=["pmra", "pmdec"],
    velocity_error_cols=["pmra_error", "pmdec_error"],
    bin_width=0.1,
)

print(result.edges)
print(result.mean_dv)
```

`result` is a `VSATResult` dataclass with:

- `n_bins`
- `edges`
- `mean_dv`
- `error`
- `n_in_bins`
- `count_stars_bins`

## Notes for Gaia-era catalogs

VSAT is still an all-pairs analysis, so runtime and memory grow as O(N^2).
Use scientifically motivated cuts before running very large catalogs.

`ra/dec` and `pmra/pmdec` can be used as components, but VSAT does not convert
astrometry into physical Cartesian coordinates. If your analysis requires pc and
km/s, transform Gaia quantities into consistent physical units before running
VSAT.

## Tests

The suite uses the standard library `unittest` runner:

```bash
python -m unittest discover
```

If `pytest` is installed, this also works:

```bash
pytest
```

## License

VSAT has an MIT license. It is free to use and modify, but comes without
warranty.
