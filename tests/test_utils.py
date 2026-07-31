"""Tests for the Lobatto quadrature utilities."""

import numpy as np
import pytest

from pytrunc.utils import integrate_lobatto, quadrature_lobatto

# Closed-form Gauss-Lobatto nodes and weights on [-1, 1]
KNOWN_RULES = {
    3: (np.array([-1.0, 0.0, 1.0]), np.array([1 / 3, 4 / 3, 1 / 3])),
    4: (
        np.array([-1.0, -1 / np.sqrt(5), 1 / np.sqrt(5), 1.0]),
        np.array([1 / 6, 5 / 6, 5 / 6, 1 / 6]),
    ),
    5: (
        np.array([-1.0, -np.sqrt(3 / 7), 0.0, np.sqrt(3 / 7), 1.0]),
        np.array([1 / 10, 49 / 90, 32 / 45, 49 / 90, 1 / 10]),
    ),
}


@pytest.mark.parametrize("n", sorted(KNOWN_RULES))
def test_known_rules(n: int) -> None:
    xk, wk = quadrature_lobatto(n=n)
    xk_ref, wk_ref = KNOWN_RULES[n]
    np.testing.assert_allclose(xk, xk_ref, atol=1e-8)
    np.testing.assert_allclose(wk, wk_ref, rtol=1e-8)


@pytest.mark.parametrize("n", [10, 51, 200])
def test_symmetry_and_weight_sum(n: int) -> None:
    xk, wk = quadrature_lobatto(n=n)
    assert xk[0] == -1.0 and xk[-1] == 1.0
    assert np.all(np.diff(xk) > 0)
    np.testing.assert_allclose(xk, -xk[::-1], atol=1e-8)
    np.testing.assert_allclose(wk, wk[::-1], rtol=1e-8)
    np.testing.assert_allclose(np.sum(wk), 2.0, rtol=1e-10)


@pytest.mark.parametrize("n", [4, 8, 16])
def test_polynomial_exactness(n: int) -> None:
    # An n-point Lobatto rule is exact for polynomials up to degree 2n-3
    xk, wk = quadrature_lobatto(n=n)
    for deg in range(2 * n - 2):
        exact = 2.0 / (deg + 1) if deg % 2 == 0 else 0.0
        np.testing.assert_allclose(
            np.sum(wk * xk**deg),
            exact,
            rtol=1e-8,
            atol=1e-8,
            err_msg=f"degree {deg}",
        )


def test_rescaled_interval() -> None:
    a, b = 0.0, np.pi
    n = 50
    xk, wk = quadrature_lobatto(abscissa_min=a, abscissa_max=b, n=n)
    xk_std, wk_std = quadrature_lobatto(n=n)
    alpha = (b - a) / 2
    np.testing.assert_allclose(xk, (xk_std + 1) * alpha + a, rtol=1e-12)
    np.testing.assert_allclose(wk, wk_std * alpha, rtol=1e-12)
    np.testing.assert_allclose(np.sum(wk), b - a, rtol=1e-10)


def test_returned_arrays_are_writable_and_independent() -> None:
    # Mutating a result must not corrupt later calls (guards the node cache)
    xk1, wk1 = quadrature_lobatto(n=20)
    xk1[:] = 0.0
    wk1[:] = 0.0
    xk2, wk2 = quadrature_lobatto(n=20)
    xk_ref, _ = KNOWN_RULES[5]
    assert xk2[0] == -1.0 and xk2[-1] == 1.0
    np.testing.assert_allclose(np.sum(wk2), 2.0, rtol=1e-10)


def test_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        quadrature_lobatto(n=1)
    with pytest.raises(ValueError):
        quadrature_lobatto(abscissa_min=1.0, abscissa_max=-1.0, n=10)


def test_integrate_sine() -> None:
    theta = np.linspace(0.0, np.pi, 1801)
    result = integrate_lobatto(np.sin(theta), theta, assume_sorted=True)
    np.testing.assert_allclose(result, 2.0, rtol=1e-5)


def test_integrate_with_provided_nodes() -> None:
    theta = np.linspace(0.0, np.pi, 1801)
    f = np.sin(theta)
    xk, wk = quadrature_lobatto(
        abscissa_min=0.0, abscissa_max=np.pi, n=len(theta)
    )
    res_provided = integrate_lobatto(
        f, theta, xk=xk, wk=wk, assume_sorted=True
    )
    res_default = integrate_lobatto(f, theta, assume_sorted=True)
    np.testing.assert_allclose(res_provided, res_default, rtol=1e-12)


def test_integrate_unsorted_input() -> None:
    rng = np.random.default_rng(42)
    theta = np.linspace(0.0, np.pi, 501)
    f = np.sin(theta) * (1 + 0.3 * np.cos(3 * theta))
    perm = rng.permutation(len(theta))
    res_sorted = integrate_lobatto(f, theta, assume_sorted=True)
    res_shuffled = integrate_lobatto(f[perm], theta[perm], assume_sorted=False)
    np.testing.assert_allclose(res_shuffled, res_sorted, rtol=1e-12)


def test_integrate_lp_parameter() -> None:
    theta = np.linspace(0.0, np.pi, 501)
    f = np.sin(theta)
    res = integrate_lobatto(f, theta, lp=101, assume_sorted=True)
    np.testing.assert_allclose(res, 2.0, rtol=1e-4)
