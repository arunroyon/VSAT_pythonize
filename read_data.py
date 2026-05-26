"""Catalog loading helpers for VSAT.

VSAT's numerical core works with component arrays shaped as
``(n_components, n_stars)``.  This module bridges modern tabular data sources
such as pandas DataFrames and CSV files into that representation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np


POSITION_COLUMN_GROUPS = (
    ("x", "y", "z"),
    ("x", "y"),
    ("ra", "dec"),
    ("l", "b"),
)

VELOCITY_COLUMN_GROUPS = (
    ("vx", "vy", "vz"),
    ("v_x", "v_y", "v_z"),
    ("u", "v", "w"),
    ("pmra", "pmdec", "radial_velocity"),
    ("pmra", "pmdec"),
    ("radial_velocity",),
    ("rv",),
)


def test_1(n_stars: int = 1000, seed: Optional[int] = None):
    """Generate data with no imposed velocity structure."""

    rng = np.random.default_rng(seed)
    r = rng.normal(size=(3, n_stars))
    v = rng.normal(size=(3, n_stars))
    verr = rng.uniform(0.5, 1.0, size=(3, n_stars))
    return n_stars, r, v, verr


def test_2(n_stars: int = 1000, seed: Optional[int] = None):
    """Generate data with a linear velocity structure."""

    rng = np.random.default_rng(seed)
    r = rng.normal(size=(3, n_stars))
    v = r.copy()
    verr = rng.uniform(0.5, 1.0, size=(3, n_stars))
    return n_stars, r, v, verr


def _import_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for reading CSV files or DataFrames. "
            "Install the project dependencies with `pip install -e .`."
        ) from exc
    return pd


def _normalise_requested_columns(columns: Optional[Iterable[str]]) -> Optional[list]:
    if columns is None:
        return None
    if isinstance(columns, str):
        columns = [part.strip() for part in columns.split(",")]
    return [str(column) for column in columns if str(column)]


def _case_map(df):
    return {str(column).lower(): column for column in df.columns}


def _resolve_group(df, group: Sequence[str]) -> Optional[list]:
    lookup = _case_map(df)
    resolved = []
    for column in group:
        match = lookup.get(column.lower())
        if match is None:
            return None
        resolved.append(match)
    return resolved


def _resolve_columns(df, requested, candidates, kind: str) -> list:
    requested = _normalise_requested_columns(requested)
    if requested is not None:
        lookup = _case_map(df)
        missing = [column for column in requested if column.lower() not in lookup]
        if missing:
            raise ValueError(f"missing {kind} column(s): {', '.join(missing)}")
        return [lookup[column.lower()] for column in requested]

    for group in candidates:
        resolved = _resolve_group(df, group)
        if resolved is not None:
            return resolved

    raise ValueError(
        f"could not infer {kind} columns. Pass explicit column names for this catalog."
    )


def _infer_error_columns(df, velocity_cols: Sequence[str]) -> Optional[list]:
    lookup = _case_map(df)
    resolved = []
    for column in velocity_cols:
        lower = str(column).lower()
        candidates = (
            f"{lower}_error",
            f"{lower}_err",
            f"e_{lower}",
            f"err_{lower}",
            f"sigma_{lower}",
            f"{lower}_uncertainty",
        )
        match = next((lookup[name] for name in candidates if name in lookup), None)
        if match is None:
            return None
        resolved.append(match)
    return resolved


def _table_from_input(data, **read_csv_kwargs):
    pd = _import_pandas()
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, (str, Path)):
        if str(data).strip() == "":
            raise ValueError("path_to_data is empty")
        path = Path(data)
        return pd.read_csv(path, **read_csv_kwargs)
    raise TypeError("data must be a pandas DataFrame or a path to a CSV file")


def read_data(
    data,
    position_cols: Optional[Iterable[str]] = None,
    velocity_cols: Optional[Iterable[str]] = None,
    velocity_error_cols: Optional[Iterable[str]] = None,
    dropna: bool = True,
    **read_csv_kwargs,
):
    """Read a CSV path or pandas DataFrame into VSAT arrays.

    Parameters
    ----------
    data:
        A pandas DataFrame or a path to a CSV file.
    position_cols, velocity_cols, velocity_error_cols:
        Optional explicit column names.  If omitted, common Cartesian and
        Gaia-style names are inferred, for example ``x/y/z`` and
        ``vx/vy/vz`` or ``ra/dec`` and ``pmra/pmdec``.
    dropna:
        Drop rows with missing values in the selected columns.  If False, a
        ValueError is raised instead.
    read_csv_kwargs:
        Extra keyword arguments forwarded to ``pandas.read_csv``.
    """

    df = _table_from_input(data, **read_csv_kwargs)
    position_cols = _resolve_columns(df, position_cols, POSITION_COLUMN_GROUPS, "position")
    velocity_cols = _resolve_columns(df, velocity_cols, VELOCITY_COLUMN_GROUPS, "velocity")

    velocity_error_cols = _normalise_requested_columns(velocity_error_cols)
    if velocity_error_cols is not None:
        velocity_error_cols = _resolve_columns(
            df, velocity_error_cols, (), "velocity error"
        )
    else:
        velocity_error_cols = _infer_error_columns(df, velocity_cols)

    selected_cols = list(position_cols) + list(velocity_cols)
    if velocity_error_cols is not None:
        selected_cols += list(velocity_error_cols)

    if dropna:
        df = df.dropna(subset=selected_cols)
    elif df[selected_cols].isna().any().any():
        raise ValueError("selected data contains missing values")

    if len(df) < 2:
        raise ValueError("at least two stars are required after filtering")

    r = df[position_cols].to_numpy(dtype=float).T
    v = df[velocity_cols].to_numpy(dtype=float).T
    verr = 0
    if velocity_error_cols is not None:
        verr = df[velocity_error_cols].to_numpy(dtype=float).T

    return len(df), r, v, verr
