"""Regression and invariant tests for gt_phase_approx."""
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import simpson, trapezoid

from cases import CASES, PHASES, THETA_DEG, TRUNC_FRAC
from pytrunc.truncation import gt_phase_approx
from pytrunc.utils import integrate_lobatto

DATA_DIR = Path(__file__).parent / 'data'


@pytest.fixture(scope='module')
def phases():
    return {name: fn(THETA_DEG) for name, fn in PHASES.items()}


@pytest.fixture(scope='module')
def results(phases):
    """Lazy per-case computation, memoized across tests of the module."""
    cache = {}

    def get(key):
        if key not in cache:
            pname, kwargs = CASES[key]
            cache[key] = gt_phase_approx(phases[pname], THETA_DEG, TRUNC_FRAC,
                                         **kwargs)
        return cache[key]

    return get


@pytest.mark.parametrize('case_key', sorted(CASES))
def test_golden(case_key, results):
    """Results must match the reference values captured from the
    pre-optimization implementation (tests/generate_references.py)."""
    ref_file = DATA_DIR / f"{case_key}.npz"
    assert ref_file.exists(), \
        f"missing reference {ref_file}; run: python tests/generate_references.py"
    ref = np.load(ref_file)
    ds = results(case_key)
    np.testing.assert_allclose(ds['phase_approx'].values, ref['phase_approx'],
                               rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(ds['phase_tr'].values, ref['phase_tr'],
                               rtol=1e-10, atol=1e-14)
    assert float(ds['f']) == float(ref['f'])
    np.testing.assert_allclose(float(ds['theta_f']), float(ref['theta_f']),
                               rtol=1e-12)
    np.testing.assert_allclose(ds['chi_star'].values, ref['chi_star'], rtol=1e-10)
    np.testing.assert_allclose(ds['chi_star_ideal'].values, ref['chi_star_ideal'],
                               rtol=1e-10)


@pytest.mark.parametrize('case_key', sorted(CASES))
def test_invariants(case_key, phases, results):
    pname, kwargs = CASES[case_key]
    method = kwargs['method']
    ds = results(case_key)
    phase = phases[pname]
    theta = np.deg2rad(THETA_DEG)
    pha_star = ds['phase_tr'].values
    pha_approx = ds['phase_approx'].values
    f = float(ds['f'])
    th_f = np.deg2rad(float(ds['theta_f']))
    id_f = int(np.argmin(np.abs(theta - th_f)))

    assert f == TRUNC_FRAC

    # flat plateau below the truncation angle
    assert id_f > 0
    np.testing.assert_allclose(pha_star[:id_f], pha_star[0], rtol=1e-12)

    # proportional to the exact phase above the truncation angle
    ratio = pha_star[id_f:] / phase[id_f:]
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-12)

    # truncated phase normalization: (1/2) ∫ P*(θ) sin(θ) dθ = 1,
    # checked with the same integrator that built it
    if method == 'lobatto':
        norm = 0.5 * integrate_lobatto(pha_star * np.sin(theta), theta,
                                       assume_sorted=True)
    else:
        mu = np.cos(theta)
        idmu = np.argsort(mu)
        integrator = simpson if method == 'simpson' else trapezoid
        norm = 0.5 * integrator(pha_star[idmu], x=mu[idmu])
    np.testing.assert_allclose(norm, 1.0, rtol=1e-3)

    # phase_approx = (1-f) * phase_star away from the forward-peak delta
    np.testing.assert_allclose(pha_approx[2:], (1 - f) * pha_star[2:],
                               rtol=1e-12)

    # everything finite (note: the plateau P_F may legitimately be negative
    # when trunc_frac is larger than the energy of the truncated peak, and
    # the simpson-normalized dirac spike may be negative on non-uniform mu)
    assert np.all(np.isfinite(pha_approx))


def test_forced_angle_is_respected(phases):
    ds = gt_phase_approx(phases['hg085'], THETA_DEG, TRUNC_FRAC,
                         method='lobatto', th_f=8.0)
    np.testing.assert_allclose(float(ds['theta_f']), 8.0, atol=0.1)


def test_searched_angle_below_tolerance(phases):
    ds = gt_phase_approx(phases['hg085'], THETA_DEG, TRUNC_FRAC,
                         method='lobatto', th_tol=20.0)
    assert 0.0 < float(ds['theta_f']) < 20.0
