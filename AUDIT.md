# VSAT modernization audit

## Findings

- The runtime depended on `fort_subroutines.so`, a Python-2-era f2py extension. If
  the binary was missing or incompatible, importing the numerical modules exited
  the process.
- `main.py` executed the full workflow at import time, which made testing and
  reuse from notebooks difficult.
- `read_data.py` required users to edit source code for every catalog format.
- Pair filtering removed zero-distance or zero-velocity pairs, but the original
  Fortran bin-count reconstruction could misattribute stars after removals.
- Plotting globally enabled LaTeX rendering, which can fail on machines without
  a TeX installation.
- There were no automated tests around pair calculations, binning, catalog
  ingestion, or correction helpers.
- A modernization pass briefly filtered out zero or near-zero velocity
  differences. That is scientifically wrong for VSAT because co-moving pairs
  with `dv = 0` are physically meaningful and must contribute to bin means.

## Changes made

- Replaced Fortran-backed pair and bin calculations with vectorized NumPy code.
- Preserved the legacy public function names used by existing scripts.
- Added CSV and pandas DataFrame ingestion with common Cartesian and Gaia-style
  column inference, plus explicit column overrides.
- Added a reusable `main.analyze(...)` API for notebooks and scripts.
- Added an argparse CLI with demo mode and non-plotting mode.
- Modernized plotting helpers to create plots only when called and to avoid a
  hard LaTeX dependency.
- Added unittest coverage for numerical calculations, error weighting,
  DataFrame/CSV loading, and correction helper behavior.
- Patched pair filtering so only zero/invalid separations and non-finite `dv`
  values are removed in the unweighted path. Zero `dv_M` and `dv_D` values are
  retained.
- Guarded observational-error inflation correction so it is only applied to the
  magnitude statistic `dv_M` and only for approximately uniform uncertainties.
- Added bin-center reporting while retaining legacy edge labels.

## Remaining scientific limits

- VSAT is still an all-pairs method, so the computational cost is O(N^2). The
  implementation is vectorized and no longer needs Fortran, but very large Gaia
  selections should still be spatially or scientifically pre-filtered before
  running the pair analysis.
- Gaia `ra/dec` and `pmra/pmdec` are accepted as tabular components, but no
  astrometric transformation is applied. If physical 3D positions and velocities
  are required, convert the catalog to consistent Cartesian units before VSAT.
- The observational-error inflation correction does not implement a full
  heteroscedastic Monte Carlo treatment for star-by-star velocity uncertainties.
