"""Generate golden reference values for the gt_phase_approx regression tests.

Run from the repository root:

    python tests/generate_references.py

Each case is stored as one .npz file in tests/data/. Cases that raise are
reported and skipped (e.g. method='simpson' before the scipy >= 1.14
keyword-argument fix), so the generator can be re-run to fill them in later.
"""
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cases import CASES, PHASES, THETA_DEG, TRUNC_FRAC  # noqa: E402

from pytrunc.truncation import gt_phase_approx  # noqa: E402

DATA_DIR = Path(__file__).parent / 'data'


def main():
    DATA_DIR.mkdir(exist_ok=True)
    phases = {name: fn(THETA_DEG) for name, fn in PHASES.items()}
    n_ok = n_fail = 0
    for key, (pname, kwargs) in sorted(CASES.items()):
        try:
            ds = gt_phase_approx(phases[pname], THETA_DEG, TRUNC_FRAC, **kwargs)
        except Exception:
            print(f"FAILED  {key}")
            traceback.print_exc(limit=1)
            n_fail += 1
            continue
        np.savez(
            DATA_DIR / f"{key}.npz",
            phase_approx=ds['phase_approx'].values,
            phase_tr=ds['phase_tr'].values,
            f=float(ds['f']),
            theta_f=float(ds['theta_f']),
            chi_star=ds['chi_star'].values,
            chi_star_ideal=ds['chi_star_ideal'].values,
        )
        print(f"written {key}")
        n_ok += 1
    print(f"\n{n_ok} references written, {n_fail} failed")


if __name__ == '__main__':
    main()
