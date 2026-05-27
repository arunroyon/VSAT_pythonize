"""Vectorized velocity-structure calculations for VSAT.

The original project delegated the pairwise distance/velocity calculations and
binning to f2py-built Fortran routines.  This module keeps the public function
names used by the old scripts, but the implementation is pure Python/NumPy and
works on modern Python without a compiled extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


ZERO_TOLERANCE = 1.0e-5


@dataclass(frozen=True)
class PairMetrics:
    """Pairwise dr/dv values and the star indices that produced them."""

    n_pairs: int
    dr_dv: np.ndarray
    max_dr: float
    pair_indices: Tuple[np.ndarray, np.ndarray]


def bin_centers(edges, bin_width: float) -> np.ndarray:
    """Return bin centers from the legacy edge labels and bin width.

    The legacy VSAT return value named ``edges`` is retained for compatibility
    with the original scripts.  For plotting and interpretation, the center of
    each dr bin is usually clearer: ``edge + 0.5 * bin_width``.
    """

    if bin_width <= 0:
        raise ValueError("bin_width must be positive")
    return np.asarray(edges, dtype=float) + (0.5 * float(bin_width))


def as_component_matrix(values, name: str, expected_n: Optional[int] = None) -> np.ndarray:
    """Return values as a ``(n_components, n_stars)`` float array.

    Existing VSAT code stores component arrays as ``[[x...], [y...], [z...]]``.
    Data-frame oriented code often produces ``(n_stars, n_components)`` arrays.
    This helper accepts both forms and transposes the latter when it can do so
    unambiguously.
    """

    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        raise ValueError(f"{name} must contain one or more components")
    if array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim == 2:
        if expected_n is not None:
            if array.shape[1] == expected_n:
                pass
            elif array.shape[0] == expected_n:
                array = array.T
            else:
                raise ValueError(
                    f"{name} has shape {array.shape}, but expected {expected_n} stars"
                )
        elif array.shape[0] > array.shape[1] and array.shape[1] <= 8:
            array = array.T
    else:
        raise ValueError(f"{name} must be a 1D or 2D array")

    if array.shape[1] < 2:
        raise ValueError(f"{name} must contain at least two stars")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return np.ascontiguousarray(array, dtype=float)


def _velocity_error_matrix(verr, v: np.ndarray, error_flag: bool) -> Optional[np.ndarray]:
    if not error_flag:
        return None
    if verr is None:
        raise ValueError("verr is required when error_flag=True")
    if np.isscalar(verr):
        uncertainty = float(verr)
        if uncertainty <= 0:
            raise ValueError("verr must be positive when error_flag=True")
        return np.full_like(v, uncertainty, dtype=float)

    verr_array = as_component_matrix(verr, "verr", expected_n=v.shape[1])
    if verr_array.shape != v.shape:
        raise ValueError(
            f"verr has shape {verr_array.shape}; expected the velocity shape {v.shape}"
        )
    if np.any(verr_array <= 0):
        raise ValueError("velocity errors must be positive")
    return verr_array


def _validate_inputs(n_stars, r, v, verr, error_flag):
    n_stars = int(n_stars)
    if n_stars < 2:
        raise ValueError("at least two stars are required")

    r_array = as_component_matrix(r, "r", expected_n=n_stars)
    v_array = as_component_matrix(v, "v", expected_n=n_stars)
    if r_array.shape[1] != n_stars or v_array.shape[1] != n_stars:
        raise ValueError("n_stars does not match r/v data")

    verr_array = _velocity_error_matrix(verr, v_array, bool(error_flag))
    return n_stars, r_array, v_array, verr_array


def _pairwise_metrics(
    n_stars,
    r,
    v,
    verr=0,
    error_flag=False,
    dv_default=True,
    bin_width=0.1,
    zero_tolerance=ZERO_TOLERANCE,
) -> PairMetrics:
    n_stars, r_array, v_array, verr_array = _validate_inputs(
        n_stars, r, v, verr, error_flag
    )
    if bin_width <= 0:
        raise ValueError("bin_width must be positive")
    if not dv_default and r_array.shape[0] != v_array.shape[0]:
        raise ValueError(
            "dv_default=False requires positions and velocities to have the same dimensions"
        )

    star_a, star_b = np.triu_indices(n_stars, k=1)
    r_delta = r_array[:, star_a] - r_array[:, star_b]
    v_delta = v_array[:, star_a] - v_array[:, star_b]

    dr = np.linalg.norm(r_delta, axis=0)
    valid = (dr > zero_tolerance) & np.isfinite(dr)
    if dv_default:
        dv = np.linalg.norm(v_delta, axis=0)
    else:
        numerator = np.einsum("ij,ij->j", r_delta, v_delta)
        dv = np.full_like(numerator, np.nan, dtype=float)
        np.divide(numerator, dr, out=dv, where=valid)

    valid &= np.isfinite(dv)

    if error_flag:
        verr_squares = verr_array[:, star_a] ** 2 + verr_array[:, star_b] ** 2
        if dv_default:
            numerator = np.sqrt(np.sum((v_delta**2) * verr_squares, axis=0))
            denominator = np.abs(dv)
        else:
            numerator = np.sqrt(np.sum((r_delta**2) * verr_squares, axis=0))
            denominator = dr

        dv_error = np.full_like(numerator, np.nan, dtype=float)
        np.divide(
            numerator,
            denominator,
            out=dv_error,
            where=np.abs(denominator) >= zero_tolerance,
        )
        valid &= np.isfinite(dv_error) & (dv_error > 0.0)
    else:
        dv_error = np.zeros_like(dv)

    dr = dr[valid]
    dv = dv[valid]
    dv_error = dv_error[valid]
    star_a = star_a[valid]
    star_b = star_b[valid]

    if len(dr) == 0:
        raise ValueError(
            "no valid star pairs remain after filtering zero/invalid dr values "
            "and non-finite dv or dv_error values"
        )

    dr_dv = np.column_stack((dr, dv, dv_error))
    max_dr = float(np.max(dr) + (2.0 * bin_width))
    return PairMetrics(len(dr_dv), dr_dv, max_dr, (star_a, star_b))


def calc_dr_dv(
    n_stars,
    r,
    v,
    verr=0,
    error_flag=False,
    dv_default=True,
    bin_width=0.1,
):
    """Calculate ``dr`` and ``dv`` for every valid pair of stars.

    Returns the legacy tuple ``(n_pairs, dr_dv, max_dr)``.  ``dr_dv`` is an
    ``(n_pairs, 3)`` array with columns ``dr``, ``dv``, and ``dv_error``.
    """

    metrics = _pairwise_metrics(
        n_stars, r, v, verr, error_flag, dv_default, bin_width
    )
    return metrics.n_pairs, metrics.dr_dv, metrics.max_dr


def _legacy_pair_indices(n_stars: int, n_pairs: int) -> Tuple[np.ndarray, np.ndarray]:
    star_a, star_b = np.triu_indices(int(n_stars), k=1)
    if n_pairs > len(star_a):
        raise ValueError("n_pairs is larger than the number of possible star pairs")
    return star_a[:n_pairs], star_b[:n_pairs]


def sort_into_bins(
    n_stars,
    n_pairs,
    dr_dv,
    max_dr,
    bin_width,
    error_flag,
    pair_indices: Optional[Tuple[np.ndarray, np.ndarray]] = None,
):
    """Sort pairwise dr/dv values into dr bins and calculate bin statistics.

    The returned ``edges`` array is the legacy lower-edge label for each
    retained bin.  Use :func:`bin_centers` when plotting against bin centers.
    """

    if bin_width <= 0:
        raise ValueError("bin_width must be positive")
    dr_dv = np.asarray(dr_dv, dtype=float)
    if dr_dv.ndim != 2 or dr_dv.shape[1] < 3:
        raise ValueError("dr_dv must be an array with columns dr, dv, dv_error")

    n_pairs = int(n_pairs)
    n_stars = int(n_stars)
    edges = np.arange(0.0, float(max_dr), float(bin_width))
    n_bins = len(edges)
    if n_bins == 0:
        raise ValueError("no bins were produced; check max_dr and bin_width")

    if pair_indices is None:
        star_a, star_b = _legacy_pair_indices(n_stars, n_pairs)
    else:
        star_a, star_b = pair_indices
        star_a = np.asarray(star_a, dtype=int)
        star_b = np.asarray(star_b, dtype=int)
    if len(star_a) != n_pairs or len(star_b) != n_pairs:
        raise ValueError("pair_indices length must match n_pairs")

    pair_bins = np.ceil(dr_dv[:, 0] / bin_width).astype(int) - 1
    pair_bins = np.clip(pair_bins, 0, n_bins - 1)

    n_in_bins = np.bincount(pair_bins, minlength=n_bins).astype(float)
    count_stars_bins = np.zeros((n_bins, n_stars), dtype=int)
    np.add.at(count_stars_bins, (pair_bins, star_a), 1)
    np.add.at(count_stars_bins, (pair_bins, star_b), 1)

    if error_flag:
        dv_error = dr_dv[:, 2]
        weights = 1.0 / (dv_error**2)
        a = np.bincount(pair_bins, weights=(dr_dv[:, 1] / dv_error) ** 2, minlength=n_bins)
        b = np.bincount(pair_bins, weights=weights, minlength=n_bins)
        c = np.bincount(pair_bins, weights=dr_dv[:, 1] * weights, minlength=n_bins)

        mean_dv = np.full(n_bins, np.nan)
        error = np.full(n_bins, np.nan)
        occupied = (n_in_bins > 0) & (b > 0)
        mean_dv[occupied] = c[occupied] / b[occupied]
        err_dv = np.zeros(n_bins)
        std_er = np.zeros(n_bins)
        err_dv[occupied] = np.sqrt(1.0 / b[occupied])
        scatter = (a[occupied] * b[occupied]) - (c[occupied] ** 2)
        scatter = np.maximum(scatter, 0.0)
        std_er[occupied] = (1.0 / b[occupied]) * np.sqrt(
            scatter / n_in_bins[occupied]
        )
        error[occupied] = np.sqrt((err_dv[occupied] ** 2) + (std_er[occupied] ** 2))
    else:
        sum_dv = np.bincount(pair_bins, weights=dr_dv[:, 1], minlength=n_bins)
        sum_dv_sq = np.bincount(pair_bins, weights=dr_dv[:, 1] ** 2, minlength=n_bins)

        mean_dv = np.full(n_bins, np.nan)
        error = np.full(n_bins, np.nan)
        occupied = n_in_bins > 0
        mean_dv[occupied] = sum_dv[occupied] / n_in_bins[occupied]
        variance = (sum_dv_sq[occupied] / n_in_bins[occupied]) - (
            mean_dv[occupied] ** 2
        )
        variance = np.maximum(variance, 0.0)
        error[occupied] = np.sqrt(variance) / np.sqrt(n_in_bins[occupied])

    return edges, n_bins, mean_dv, error, n_in_bins, count_stars_bins


def tidy_up(
    n_bins,
    edges,
    mean_dv,
    error,
    n_in_bins,
    count_stars_bins,
    min_tail_pairs=30,
):
    """Remove empty bins and sparse high-dr tails.

    ``min_tail_pairs`` keeps the historical VSAT behavior of trimming the
    high-dr tail once the average remaining occupancy drops below 30 pairs.
    Small catalogs are protected from being trimmed down to zero bins.
    """

    n_bins = int(n_bins)
    edges = np.asarray(edges, dtype=float)
    mean_dv = np.asarray(mean_dv, dtype=float)
    error = np.asarray(error, dtype=float)
    n_in_bins = np.asarray(n_in_bins, dtype=float)
    count_stars_bins = np.asarray(count_stars_bins)

    if n_bins == 0:
        return n_bins, edges, mean_dv, error, n_in_bins, count_stars_bins

    keep_until = n_bins
    if min_tail_pairs is not None and min_tail_pairs > 0:
        tail_counts = np.cumsum(n_in_bins[::-1])[::-1]
        tail_widths = np.arange(n_bins, 0, -1, dtype=float)
        sparse_tail = np.flatnonzero((tail_counts / tail_widths) < min_tail_pairs)
        if len(sparse_tail) > 0:
            keep_until = int(sparse_tail[0])
            if keep_until == 0 and np.any(n_in_bins > 0):
                keep_until = int(np.flatnonzero(n_in_bins > 0)[-1] + 1)

    edges = edges[:keep_until]
    mean_dv = mean_dv[:keep_until]
    error = error[:keep_until]
    n_in_bins = n_in_bins[:keep_until]
    count_stars_bins = count_stars_bins[:keep_until]

    occupied = n_in_bins > 0
    edges = edges[occupied]
    mean_dv = mean_dv[occupied]
    error = error[occupied]
    n_in_bins = n_in_bins[occupied]
    count_stars_bins = count_stars_bins[occupied]
    n_bins = len(edges)

    return n_bins, edges, mean_dv, error, n_in_bins, count_stars_bins


def calc_drdv_and_sort(
    n_stars,
    r,
    v,
    verr=0,
    error_flag=False,
    dv_default=True,
    bin_width=0.1,
    min_tail_pairs=30,
):
    """Calculate pairwise dr/dv values, bin them, and return VSAT statistics.

    Returns the legacy 6-tuple ``(n_bins, edges, mean_dv, error, n_in_bins,
    count_stars_bins)``.  ``edges`` is retained for compatibility; bin centers
    can be calculated with ``bin_centers(edges, bin_width)``.
    """

    metrics = _pairwise_metrics(
        n_stars, r, v, verr, error_flag, dv_default, bin_width
    )
    edges, n_bins, mean_dv, error, n_in_bins, count_stars_bins = sort_into_bins(
        n_stars,
        metrics.n_pairs,
        metrics.dr_dv,
        metrics.max_dr,
        bin_width,
        error_flag,
        pair_indices=metrics.pair_indices,
    )
    return tidy_up(
        n_bins,
        edges,
        mean_dv,
        error,
        n_in_bins,
        count_stars_bins,
        min_tail_pairs=min_tail_pairs,
    )
