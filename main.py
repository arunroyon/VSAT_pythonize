"""Command-line and Python API entry points for VSAT."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np

import calc_drdv_and_sort
import correct_dv
import parameters
import read_data


@dataclass(frozen=True)
class VSATResult:
    n_bins: int
    edges: np.ndarray
    mean_dv: np.ndarray
    error: np.ndarray
    n_in_bins: np.ndarray
    count_stars_bins: np.ndarray
    bin_width: Optional[float] = None

    @property
    def bin_centers(self) -> np.ndarray:
        """Centers of the dr bins corresponding to ``edges``."""

        if self.bin_width is None:
            raise ValueError("bin_width is required to calculate bin centers")
        return calc_drdv_and_sort.bin_centers(self.edges, self.bin_width)


def _has_velocity_errors(verr) -> bool:
    return not (np.isscalar(verr) and float(verr) == 0.0)


def _apply_inflation_correction(
    n_stars,
    v,
    verr,
    mean_dv,
    error_flag: bool,
    dv_default: bool,
    correct_inflation: bool,
):
    if not correct_inflation:
        return mean_dv
    if not dv_default:
        raise ValueError(
            "correct_inflation is only implemented for dv_default=True "
            "(the magnitude statistic dv_M). Arnold & Goodwin do not apply "
            "this correction to the directional statistic dv_D."
        )
    if not error_flag:
        raise ValueError("correct_inflation requires velocity uncertainties")

    uncertainty = correct_dv.uniform_uncertainty(verr)
    return correct_dv.correct(n_stars, v, uncertainty, mean_dv)


def analyze(
    data,
    position_cols=None,
    velocity_cols=None,
    velocity_error_cols=None,
    error_flag: Optional[bool] = None,
    dv_default: bool = True,
    correct_inflation: bool = False,
    bin_width: float = 0.1,
    min_tail_pairs: int = 30,
    **read_csv_kwargs,
) -> VSATResult:
    """Run VSAT on a CSV path or pandas DataFrame."""

    n_stars, r, v, verr = read_data.read_data(
        data,
        position_cols=position_cols,
        velocity_cols=velocity_cols,
        velocity_error_cols=velocity_error_cols,
        **read_csv_kwargs,
    )
    if error_flag is None:
        error_flag = _has_velocity_errors(verr)

    if correct_inflation and not dv_default:
        raise ValueError(
            "correct_inflation is only implemented for dv_default=True "
            "(the magnitude statistic dv_M)."
        )

    result = calc_drdv_and_sort.calc_drdv_and_sort(
        n_stars,
        r,
        v,
        verr,
        error_flag,
        dv_default,
        bin_width,
        min_tail_pairs=min_tail_pairs,
    )
    n_bins, edges, mean_dv, error, n_in_bins, count_stars_bins = result

    mean_dv = _apply_inflation_correction(
        n_stars, v, verr, mean_dv, error_flag, dv_default, correct_inflation
    )

    return VSATResult(
        n_bins, edges, mean_dv, error, n_in_bins, count_stars_bins, bin_width
    )


def _split_columns(value):
    if value is None:
        return None
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return value


def build_parser():
    parser = argparse.ArgumentParser(description="Velocity Structure Analysis Tool")
    parser.add_argument("data", nargs="?", help="CSV catalog path")
    parser.add_argument("--position-cols", help="Comma-separated position columns")
    parser.add_argument("--velocity-cols", help="Comma-separated velocity columns")
    parser.add_argument("--velocity-error-cols", help="Comma-separated velocity error columns")
    parser.add_argument("--bin-width", type=float, default=parameters.DEFAULTS.bin_width)
    parser.add_argument("--dr-start", type=float, default=parameters.DEFAULTS.dr_start)
    parser.add_argument("--dr-end", type=float, default=parameters.DEFAULTS.dr_end)
    parser.add_argument("--min-tail-pairs", type=int, default=30)
    parser.add_argument(
        "--errors",
        dest="error_flag",
        action="store_true",
        default=None,
        help="Force velocity-error weighting",
    )
    parser.add_argument(
        "--no-errors",
        dest="error_flag",
        action="store_false",
        help="Ignore velocity-error columns",
    )
    parser.add_argument(
        "--radial-dv",
        action="store_true",
        help="Use dr/dt-style signed dv instead of velocity-difference magnitude",
    )
    parser.add_argument(
        "--correct-inflation",
        action="store_true",
        default=parameters.DEFAULTS.correct_inflation,
        help="Estimate and subtract measurement-error inflation",
    )
    parser.add_argument("--demo", action="store_true", help="Run the built-in demo catalog")
    parser.add_argument("--no-show", action="store_true", help="Calculate without displaying plots")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    dv_default = not args.radial_dv

    if args.demo:
        n_stars, r, v, verr = read_data.test_2(seed=42)
        error_flag = args.error_flag if args.error_flag is not None else True
        result = calc_drdv_and_sort.calc_drdv_and_sort(
            n_stars,
            r,
            v,
            verr,
            error_flag,
            dv_default,
            args.bin_width,
            min_tail_pairs=args.min_tail_pairs,
        )
        n_bins, edges, mean_dv, error, n_in_bins, count_stars_bins = result
        mean_dv = _apply_inflation_correction(
            n_stars,
            v,
            verr,
            mean_dv,
            error_flag,
            dv_default,
            args.correct_inflation,
        )
        vsat_result = VSATResult(
            n_bins,
            edges,
            mean_dv,
            error,
            n_in_bins,
            count_stars_bins,
            args.bin_width,
        )
    else:
        data = args.data or parameters.DEFAULTS.path_to_data
        if not data:
            raise SystemExit("Provide a CSV path or run with --demo.")
        vsat_result = analyze(
            data,
            position_cols=_split_columns(args.position_cols),
            velocity_cols=_split_columns(args.velocity_cols),
            velocity_error_cols=_split_columns(args.velocity_error_cols),
            error_flag=args.error_flag,
            dv_default=dv_default,
            correct_inflation=args.correct_inflation,
            bin_width=args.bin_width,
            min_tail_pairs=args.min_tail_pairs,
        )

    if args.no_show:
        print(f"calculated {vsat_result.n_bins} populated dr bins")
        return vsat_result

    import plotting

    plt = plotting._pyplot()
    _, axes = plt.subplots(1, 2, figsize=(12, 5))
    plotting.v_struct_plot(
        vsat_result.edges,
        vsat_result.mean_dv,
        vsat_result.error,
        ax=axes[0],
        bin_width=vsat_result.bin_width,
    )
    if args.demo:
        plot_r = r
    else:
        n_stars, plot_r, _, _ = read_data.read_data(
            args.data,
            position_cols=_split_columns(args.position_cols),
            velocity_cols=_split_columns(args.velocity_cols),
            velocity_error_cols=_split_columns(args.velocity_error_cols),
        )
    plotting.col_code_plot(
        n_stars,
        plot_r,
        vsat_result.edges,
        vsat_result.count_stars_bins,
        args.dr_start,
        args.dr_end,
        ax=axes[1],
    )
    plt.tight_layout()
    plt.show()
    return vsat_result


if __name__ == "__main__":
    main()
