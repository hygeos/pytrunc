"""Tests of the phase module."""

import numpy as np
import pytest
from scipy.integrate import simpson

from pytrunc.phase import fournier_forand

# Real-world (n, mu) pairs from Park & Ruddick (2005), as used in
# SMART-G
FF_PARAMS = [(1.117, 3.695), (1.05, 3.259)]


def backscatter_fraction(n: float, mu: float) -> float:
    # Analytic B = bb/b of the Fournier-Forand phase function, from
    # http://www.oceanopticsbook.info/view/scattering/the-fournier-forand-phase-function
    nu = (3 - mu) / 2
    d90 = 2 / (3 * (n - 1) ** 2)
    return 1 - (1 - d90 ** (nu + 1) - 0.5 * (1 - d90**nu)) / (
        (1 - d90) * d90**nu
    )


@pytest.mark.parametrize("n,mu", FF_PARAMS)
def test_backscatter_fraction(n: float, mu: float) -> None:
    # The bare formula integrates to 1 over the sphere by construction,
    # so the integral over the backward hemisphere equals B directly (no
    # need to integrate the divergent forward peak).
    theta = np.linspace(np.pi / 2, np.pi, 20001)
    phase = fournier_forand(theta, n, mu, theta_unit="rad")
    b_num = 2 * np.pi * simpson(phase * np.sin(theta), x=theta)
    np.testing.assert_allclose(b_num, backscatter_fraction(n, mu), rtol=1e-4)


def test_default_normalization() -> None:
    # ∫F dΩ ≈ 1 for a moderate forward peak; the mass below the grid
    # start angle is lost, hence the loose tolerance
    theta = np.deg2rad(np.linspace(1e-3, 180.0, 100001))
    phase = fournier_forand(theta, 1.117, 3.695, theta_unit="rad")
    total = 2 * np.pi * simpson(phase * np.sin(theta), x=theta)
    np.testing.assert_allclose(total, 1.0, rtol=1e-2)


@pytest.mark.parametrize("n,mu", FF_PARAMS)
def test_normalize_parameter(n: float, mu: float) -> None:
    theta = np.linspace(1e-3, 180.0, 10001)
    phase = fournier_forand(theta, n, mu, normalize=2)
    mu_cos = np.cos(np.deg2rad(theta))
    idmu = np.argsort(mu_cos)
    integral = simpson(phase[idmu], x=mu_cos[idmu])
    np.testing.assert_allclose(integral, 2.0, rtol=1e-12)


@pytest.mark.parametrize("n,mu", FF_PARAMS)
def test_theta_unit_consistency(n: float, mu: float) -> None:
    theta_deg = np.linspace(0.1, 180.0, 1801)
    res_deg = fournier_forand(theta_deg, n, mu)
    res_rad = fournier_forand(np.deg2rad(theta_deg), n, mu, theta_unit="rad")
    np.testing.assert_allclose(res_rad, res_deg, rtol=1e-12)


def test_invalid_theta_unit() -> None:
    with pytest.raises(ValueError):
        fournier_forand(np.linspace(0.1, 180.0, 181), 1.117, 3.695, "bad")
