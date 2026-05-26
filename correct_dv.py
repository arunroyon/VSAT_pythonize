"""Correct velocity-structure inflation caused by measurement errors."""

from __future__ import annotations

import numpy as np

from calc_drdv_and_sort import as_component_matrix


def average_pairwise_velocity_difference(v, block_size: int = 512) -> float:
    """Average pairwise velocity difference without requiring Fortran."""

    velocities = as_component_matrix(v, "v").T
    n_stars = len(velocities)
    if n_stars < 2:
        raise ValueError("at least two stars are required")

    total = 0.0
    count = 0
    for start in range(0, n_stars, block_size):
        stop = min(start + block_size, n_stars)
        block = velocities[start:stop]

        local_a, local_b = np.triu_indices(len(block), k=1)
        if len(local_a):
            diffs = block[local_a] - block[local_b]
            total += float(np.linalg.norm(diffs, axis=1).sum())
            count += len(local_a)

        for other_start in range(stop, n_stars, block_size):
            other_stop = min(other_start + block_size, n_stars)
            other = velocities[other_start:other_stop]
            diffs = block[:, None, :] - other[None, :, :]
            total += float(np.linalg.norm(diffs, axis=2).sum())
            count += len(block) * len(other)

    return total / count


def _gaussian_kde_pdf(samples, grid):
    samples = np.asarray(samples, dtype=float)
    grid = np.asarray(grid, dtype=float)
    std = float(np.std(samples))
    bandwidth = 1.06 * std * (len(samples) ** (-1.0 / 5.0)) if std > 0 else 0.1
    bandwidth = max(bandwidth, 1.0e-6)
    scaled = (grid[:, None] - samples[None, :]) / bandwidth
    pdf = np.exp(-0.5 * scaled**2).sum(axis=1)
    pdf /= len(samples) * bandwidth * np.sqrt(2.0 * np.pi)
    return pdf


def draw_velocities(pdf_v_width, pdf_prob, n_stars, rng=None):
    """Draw velocities from a sampled probability density function."""

    rng = np.random.default_rng(rng)
    pdf_v_width = np.asarray(pdf_v_width, dtype=float)
    pdf_prob = np.asarray(pdf_prob, dtype=float)
    pdf_prob = np.clip(pdf_prob, 0.0, None)
    if pdf_v_width.ndim != 1 or pdf_prob.ndim != 1 or len(pdf_v_width) != len(pdf_prob):
        raise ValueError("pdf_v_width and pdf_prob must be same-length 1D arrays")
    if np.sum(pdf_prob) <= 0:
        raise ValueError("pdf_prob must contain positive probability")

    cdf = np.cumsum(pdf_prob)
    cdf /= cdf[-1]
    draws = rng.random(int(n_stars))
    return np.interp(draws, cdf, pdf_v_width)


def optimum_model(
    n_stars,
    v_component,
    uncertainty,
    pdf_v,
    pdf_prob,
    rng=None,
    n_widths: int = 100,
    n_tests: int = 100,
):
    """Return the velocity grid width that best reproduces observations."""

    rng = np.random.default_rng(rng)
    observed = np.sort(np.asarray(v_component, dtype=float))
    observed_width = float(np.std(observed))
    if observed_width <= 0:
        return np.asarray(pdf_v, dtype=float)

    min_width = max(0.05, observed_width - float(uncertainty) - 0.25)
    if min_width >= observed_width:
        min_width = max(0.05, observed_width * 0.5)
    widths = np.linspace(min_width, observed_width, int(n_widths), endpoint=False)

    areas = np.empty(len(widths), dtype=float)
    for index, width in enumerate(widths):
        ratio_adjust = width / observed_width
        pdf_v_width = np.asarray(pdf_v, dtype=float) * ratio_adjust
        test_areas = np.empty(int(n_tests), dtype=float)
        for test in range(int(n_tests)):
            simulated = draw_velocities(pdf_v_width, pdf_prob, n_stars, rng=rng)
            simulated = rng.normal(loc=simulated, scale=uncertainty)
            simulated.sort()
            test_areas[test] = np.mean(np.abs(observed - simulated))
        areas[index] = np.mean(test_areas)

    optimum_width = widths[int(np.argmin(areas))]
    return np.asarray(pdf_v, dtype=float) * (optimum_width / observed_width)


def correct(
    n_stars,
    v,
    uncertainty,
    mean_dv,
    rng=None,
    n_model_samples: int = 10000,
    n_widths: int = 100,
    n_tests: int = 100,
):
    """Estimate and subtract measurement-error inflation from mean dv values."""

    rng = np.random.default_rng(rng)
    velocities = as_component_matrix(v, "v", expected_n=int(n_stars))
    obs_av_dv = average_pairwise_velocity_difference(velocities)

    v_true_sim = []
    for component in velocities:
        component = np.asarray(component, dtype=float)
        pdf_v = np.linspace(np.min(component) - 0.2, np.max(component) + 0.2, 100)
        pdf_prob = _gaussian_kde_pdf(component, pdf_v)
        pdf_v_opt = optimum_model(
            n_stars,
            component,
            uncertainty,
            pdf_v,
            pdf_prob,
            rng=rng,
            n_widths=n_widths,
            n_tests=n_tests,
        )
        v_true_sim.append(
            draw_velocities(pdf_v_opt, pdf_prob, n_model_samples, rng=rng)
        )

    est_true_av_dv = average_pairwise_velocity_difference(np.asarray(v_true_sim))
    correction_factor = obs_av_dv - est_true_av_dv
    return np.asarray(mean_dv, dtype=float) - correction_factor
