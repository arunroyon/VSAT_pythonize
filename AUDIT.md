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

## Remaining scientific limits

- VSAT is still an all-pairs method, so the computational cost is O(N^2). The
  implementation is vectorized and no longer needs Fortran, but very large Gaia
  selections should still be spatially or scientifically pre-filtered before
  running the pair analysis.
- Gaia `ra/dec` and `pmra/pmdec` are accepted as tabular components, but no
  astrometric transformation is applied. If physical 3D positions and velocities
  are required, convert the catalog to consistent Cartesian units before VSAT.
