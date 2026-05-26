"""Plotting helpers for VSAT results."""

from __future__ import annotations

import numpy as np


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. Install the project dependencies "
            "with `pip install -e .`."
        ) from exc
    return plt


def v_struct_plot(edges, mean_dv, error, ax=None):
    """Plot velocity structure as mean dv versus dr-bin lower edge."""

    plt = _pyplot()
    if ax is None:
        ax = plt.gca()

    edges = np.asarray(edges, dtype=float)
    mean_dv = np.asarray(mean_dv, dtype=float)
    error = np.asarray(error, dtype=float)
    if len(edges) == 0:
        raise ValueError("no bins to plot")

    ax.errorbar(edges, mean_dv, yerr=error, capsize=3.0)
    ax.set_xlabel(r"$\Delta r$ (pc)")
    ax.set_ylabel(r"$\Delta v$ (km s$^{-1}$)")
    ax.set_title("Velocity structure")
    ax.set_xlim(left=0.0, right=float(edges[-1]) if len(edges) else 1.0)
    return ax


def col_code_plot(n_stars, r, edges, count_stars_bins, dr_start, dr_end, ax=None):
    """Plot projected positions colored by contribution to selected dr bins."""

    plt = _pyplot()
    if ax is None:
        ax = plt.gca()

    r = np.asarray(r, dtype=float)
    edges = np.asarray(edges, dtype=float)
    count_stars_bins = np.asarray(count_stars_bins)
    if r.ndim != 2 or r.shape[0] < 2:
        raise ValueError("at least 2D spatial information is required for this plot")
    if len(edges) == 0:
        raise ValueError("no bins to plot")
    if dr_end <= dr_start:
        raise ValueError("dr_end must be greater than dr_start")

    start_bin = int(np.argmin(np.abs(edges - dr_start)))
    end_bin = int(np.argmin(np.abs(edges - dr_end)))
    if end_bin <= start_bin:
        raise ValueError("selected dr range maps to an empty bin range")

    count_stars_range = count_stars_bins[start_bin:end_bin].sum(axis=0)
    scatter = ax.scatter(
        r[0],
        r[1],
        c=count_stars_range,
        alpha=0.8,
        cmap="viridis",
        s=8.0,
    )
    colorbar = plt.colorbar(scatter, ax=ax)
    colorbar.set_label("Counts", rotation=270, labelpad=20)
    ax.set_xlabel("x (pc)")
    ax.set_ylabel("y (pc)")
    ax.set_title(
        r"Counts in $\Delta r$ bins between "
        f"{dr_start:.1f} and {dr_end:.1f} pc"
    )
    return ax
