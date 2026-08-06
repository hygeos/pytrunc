"""Shared test case definitions for the gt_phase_approx regression
tests."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pytrunc.phase import henyey_greenstein, two_term_henyey_greenstein

THETA_DEG: NDArray[np.float64] = np.linspace(0.0, 180.0, 1801)
TRUNC_FRAC: float = 0.2

# gt_phase_approx assumes the phase function is normalized as
# ∫P(θ)sinθ dθ = 2
PHASES = {
    "hg085": lambda theta: henyey_greenstein(theta, g=0.85, normalize=2),
    "tthg": lambda theta: two_term_henyey_greenstein(
        theta, g1=0.8, g2=-0.5, f=0.9, normalize=2
    ),
}

# (method, lobatto_optimization) — None means the flag is not passed
METHODS = [
    ("lobatto", True),
    ("lobatto", False),
    ("trapezoid", None),
    ("simpson", None),
]

MODES = {
    "forced": {"th_f": 8.0},
    "searched": {"th_tol": 20.0},
}


def make_cases() -> dict[str, tuple[str, dict[str, Any]]]:
    """Return {case_key: (phase_name, gt_phase_approx kwargs)}."""
    cases = {}
    for pname in PHASES:
        for method, lob_opt in METHODS:
            mkey = method if lob_opt is None else f"{method}_opt{int(lob_opt)}"
            for mode, mode_kw in MODES.items():
                kwargs = dict(method=method, **mode_kw)
                if lob_opt is not None:
                    kwargs["lobatto_optimization"] = lob_opt
                cases[f"{pname}__{mkey}__{mode}"] = (pname, kwargs)
    return cases


CASES = make_cases()
