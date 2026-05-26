"""Default VSAT parameters.

Command-line runs can override these values in ``main.py``.  The legacy
``set_parameters`` function is retained for scripts that already import it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VSATParameters:
    path_to_data: str = ""
    error_flag: bool = True
    dv_default: bool = True
    correct_inflation: bool = False
    bin_width: float = 0.1
    dr_start: float = 0.0
    dr_end: float = 1.0


DEFAULTS = VSATParameters()


def set_parameters():
    """Return parameters in the tuple order used by the original script."""

    return (
        DEFAULTS.path_to_data,
        DEFAULTS.error_flag,
        DEFAULTS.dv_default,
        DEFAULTS.correct_inflation,
        DEFAULTS.bin_width,
        DEFAULTS.dr_start,
        DEFAULTS.dr_end,
    )
