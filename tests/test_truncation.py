"""Tests of the truncation module."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from cases import CASES, PHASES, THETA_DEG, TRUNC_FRAC
from numpy.typing import NDArray
from scipy.integrate import simpson, trapezoid

from pytrunc.phase import fournier_forand, henyey_greenstein
from pytrunc.truncation import gt_phase_approx
from pytrunc.utils import integrate_lobatto

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def phases() -> dict[str, NDArray[np.float64]]:
    return {name: fn(THETA_DEG) for name, fn in PHASES.items()}


@pytest.fixture(scope="module")
def results(
    phases: dict[str, NDArray[np.float64]],
) -> Callable[[str], xr.Dataset]:
    """Lazy per-case computation, memoized across tests of the
    module."""
    cache: dict[str, xr.Dataset] = {}

    def get(key: str) -> xr.Dataset:
        if key not in cache:
            pname, kwargs = CASES[key]
            ds = gt_phase_approx(
                phases[pname], THETA_DEG, TRUNC_FRAC, **kwargs
            )
            assert isinstance(ds, xr.Dataset)
            cache[key] = ds
        return cache[key]

    return get


@pytest.mark.parametrize("case_key", sorted(CASES))
def test_golden(case_key: str, results: Callable[[str], xr.Dataset]) -> None:
    """Results must match the reference values captured from the
    pre-optimization implementation (tests/generate_references.py)."""
    ref_file = DATA_DIR / f"{case_key}.npz"
    assert ref_file.exists(), (
        f"missing reference {ref_file}; "
        "run: python tests/generate_references.py"
    )
    ref = np.load(ref_file)
    ds = results(case_key)
    np.testing.assert_allclose(
        ds["phase_approx"].values, ref["phase_approx"], rtol=1e-10, atol=1e-14
    )
    np.testing.assert_allclose(
        ds["phase_tr"].values, ref["phase_tr"], rtol=1e-10, atol=1e-14
    )
    assert float(ds["f"]) == float(ref["f"])
    np.testing.assert_allclose(
        float(ds["theta_f"]), float(ref["theta_f"]), rtol=1e-12
    )
    np.testing.assert_allclose(
        ds["chi_star"].values, ref["chi_star"], rtol=1e-10
    )
    np.testing.assert_allclose(
        ds["chi_star_ideal"].values, ref["chi_star_ideal"], rtol=1e-10
    )


@pytest.mark.parametrize("case_key", sorted(CASES))
def test_invariants(
    case_key: str,
    phases: dict[str, NDArray[np.float64]],
    results: Callable[[str], xr.Dataset],
) -> None:
    pname, kwargs = CASES[case_key]
    method = kwargs["method"]
    ds = results(case_key)
    phase = phases[pname]
    theta = np.deg2rad(THETA_DEG)
    pha_star = ds["phase_tr"].values
    pha_approx = ds["phase_approx"].values
    f = float(ds["f"])
    th_f = np.deg2rad(float(ds["theta_f"]))
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
    if method == "lobatto":
        norm = 0.5 * integrate_lobatto(
            pha_star * np.sin(theta), theta, assume_sorted=True
        )
    else:
        mu = np.cos(theta)
        idmu = np.argsort(mu)
        integrator = simpson if method == "simpson" else trapezoid
        norm = 0.5 * integrator(pha_star[idmu], x=mu[idmu])
    np.testing.assert_allclose(norm, 1.0, rtol=1e-3)

    # phase_approx = (1-f) * phase_star away from the forward-peak delta
    np.testing.assert_allclose(
        pha_approx[2:], (1 - f) * pha_star[2:], rtol=1e-12
    )

    # everything finite (note: with an imposed angle, the plateau P_F
    # may legitimately be negative when trunc_frac is larger than the
    # energy of the truncated peak, and the simpson-normalized dirac
    # spike may be negative on non-uniform mu)
    assert np.all(np.isfinite(pha_approx))

    # a searched angle leaves a non-negative plateau
    if "th_f" not in kwargs:
        assert pha_star[0] >= 0.0


def test_forced_angle_is_respected(
    phases: dict[str, NDArray[np.float64]],
) -> None:
    ds = gt_phase_approx(
        phases["hg085"], THETA_DEG, TRUNC_FRAC, method="lobatto", th_f=8.0
    )
    assert isinstance(ds, xr.Dataset)
    np.testing.assert_allclose(float(ds["theta_f"]), 8.0, atol=0.1)


def test_searched_angle_below_tolerance(
    phases: dict[str, NDArray[np.float64]],
) -> None:
    ds = gt_phase_approx(
        phases["hg085"], THETA_DEG, TRUNC_FRAC, method="lobatto", th_tol=20.0
    )
    assert isinstance(ds, xr.Dataset)
    assert 0.0 < float(ds["theta_f"]) < 20.0



def _fournier_forand(n: float, mu: float) -> NDArray[np.float64]:
    """Fournier-Forand on THETA_DEG, normalized to 2.

    Its value at 0 degree, 0/0, is replaced by the one at the next
    angle; the normalization is the trapezoid rule in theta.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        phase = fournier_forand(THETA_DEG, n=n, mu=mu)
    phase[0] = phase[1]
    theta = np.deg2rad(THETA_DEG)
    return 2.0 * phase / trapezoid(phase * np.sin(theta), x=theta)


@pytest.mark.parametrize("method", ["trapezoid", "simpson", "lobatto"])
def test_search_skips_negative_plateaus(method: str) -> None:
    """The searched angle leaves a non-negative plateau.

    Within the sharp peak of Fournier-Forand, the first moment matches
    best at angles within which the peak holds less than trunc_frac of
    the scattering, where the plateau is negative: pytrunc 2.0.0
    returned a plateau of -445 at 1.1 degree with the trapezoid rule.
    """
    ds = gt_phase_approx(
        _fournier_forand(1.10, 3.5), THETA_DEG, 0.3, method=method
    )
    assert isinstance(ds, xr.Dataset)
    assert np.all(ds["phase_tr"].values >= 0.0)


def test_search_without_valid_angle_raises(
    phases: dict[str, NDArray[np.float64]],
) -> None:
    """Within 5 degrees the tthg peak holds less than 10 % of the
    scattering: no angle leaves a non-negative plateau for 0.5."""
    with pytest.raises(ValueError, match="non-negative plateau"):
        gt_phase_approx(
            phases["tthg"], THETA_DEG, 0.5, method="trapezoid", th_tol=5.0
        )


@pytest.mark.parametrize("method", ["lobatto", "trapezoid", "simpson"])
@pytest.mark.parametrize("pname", sorted(PHASES))
def test_continuous_plateau(
    pname: str, method: str, phases: dict[str, NDArray[np.float64]]
) -> None:
    """With trunc_frac=None, the plateau meets the phase at th_f.

    Its f is the fraction that imposing the angle takes to give that
    plateau: both give the same matrix.
    """
    phase = phases[pname]
    ds = gt_phase_approx(phase, THETA_DEG, None, method=method, th_f=8.0)
    assert isinstance(ds, xr.Dataset)
    pha_star = ds["phase_tr"].values
    id_f = int(np.argmin(np.abs(THETA_DEG - 8.0)))
    np.testing.assert_allclose(pha_star[:id_f], pha_star[id_f], rtol=1e-12)
    f = float(ds["f"])
    assert 0.0 < f < 1.0
    assert ds["trunc_frac"].values.item() is None
    imposed = gt_phase_approx(phase, THETA_DEG, f, method=method, th_f=8.0)
    assert isinstance(imposed, xr.Dataset)
    np.testing.assert_array_equal(imposed["phase_tr"].values, pha_star)
    assert float(imposed["trunc_frac"]) == f


def test_continuous_plateau_cuts_the_phase_flat() -> None:
    """Fournier-Forand cut flat at its value at 5 degrees.

    (1 - f) P* is the phase matrix with its forward peak replaced by
    that value, but for the normalization of P* by the integrator: the
    truncation of the ocean phase matrices of SMART-G up to 1.2.
    """
    phase = _fournier_forand(1.10, 3.5)
    ds = gt_phase_approx(phase, THETA_DEG, None, th_f=5.0)
    assert isinstance(ds, xr.Dataset)
    id_f = int(np.argmin(np.abs(THETA_DEG - 5.0)))
    flat = phase.copy()
    flat[:id_f] = phase[id_f]
    ratio = (1.0 - float(ds["f"])) * ds["phase_tr"].values / flat
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-12)
    np.testing.assert_allclose(ratio[0], 1.0, rtol=1e-3)


def test_continuous_plateau_needs_the_angle(
    phases: dict[str, NDArray[np.float64]],
) -> None:
    with pytest.raises(ValueError, match="needs th_f"):
        gt_phase_approx(phases["hg085"], THETA_DEG, None)


def test_continuous_plateau_needs_a_forward_peak() -> None:
    """A backward-peaked phase matrix has no forward peak above its
    value at th_f: f would be negative."""
    phase = henyey_greenstein(THETA_DEG, g=-0.5, normalize=2)
    with pytest.raises(ValueError, match="no forward peak"):
        gt_phase_approx(phase, THETA_DEG, None, th_f=8.0)
