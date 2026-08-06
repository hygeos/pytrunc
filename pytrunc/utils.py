"""
Numerical utilities.

This module provides the numerical routines used by the phase and
truncation modules: the Legendre polynomials and their derivatives, the
Bessel functions of the first kind, and the Lobatto quadrature
abscissas, weights and integration.
"""

import functools
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammaln, j1, jn_zeros, jvp


def legendre_polynomials(n: int, x: ArrayLike) -> NDArray[np.float64]:
    """
    Use the recursion formulas to compute the Legendre polynomials Pn(x)

    - see Eq. 9 in Michels (1963)

    Parameters
    ----------
    n : int
        The Legendre polynomial order
    x : ndarray
        The x values of Pn(x), it must be 1-D

    Returns
    -------
    ndarray
        The Legendre series, a 1-D ndarray with the same shape as x

    Notes
    -----
    The numpy equivalent ->
    numpy.polynomial.legendre.Legendre.basis(n)(x)

    References
    ----------
    Michels, H. (1963). Abscissas and weight coefficients for Lobatto
    quadrature. Mathematics of Computation, 17(83), 237-244.

    Wiscombe, W. J. (1977). The delta-M method: Rapid yet accurate
    radiative flux calculations for strongly asymmetric phase functions.
    Journal of Atmospheric Sciences, 34(9), 1408-1422.

    Examples
    --------
    >>> import numpy as np
    >>> from pytrunc.utils import legendre_polynomials
    >>> x = np.array([-1.0, 0.0, 1.0])
    >>> legendre_polynomials(2, x)
    array([ 1. , -0.5,  1. ])
    """
    x = np.asarray(x)

    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x

    pnm1 = np.ones_like(x)
    pn = x
    for k in range(1, n):
        pnp1 = (1.0 / (k + 1)) * ((2 * k + 1) * x * pn - k * pnm1)
        pnm1, pn = pn, pnp1

    return pnp1


def legendre_polynomials_derivative(
    n: int, x: ArrayLike
) -> NDArray[np.float64]:
    """
    Use the recursion formulas to compute the derivative Legendre
    polynomials d(Pn(x))

    - see Eq. 10 in Michels (1963)

    Parameters
    ----------
    n : int
        The Legendre polynomial order
    x : ndarray
        The x values of d(Pn(x)), it must be 1-D

    Returns
    -------
    ndarray
        The derivative Legendre series, a 1-D ndarray with the same
        shape as x

    Notes
    -----
    The numpy equivalent ->
    numpy.polynomial.legendre.Legendre.basis(n).deriv(1)(x)

    References
    ----------
    Michels, H. (1963). Abscissas and weight coefficients for Lobatto
    quadrature. Mathematics of Computation, 17(83), 237-244.

    Wiscombe, W. J. (1977). The delta-M method: Rapid yet accurate
    radiative flux calculations for strongly asymmetric phase functions.
    Journal of Atmospheric Sciences, 34(9), 1408-1422.

    Examples
    --------
    >>> import numpy as np
    >>> from pytrunc.utils import legendre_polynomials_derivative
    >>> x = np.array([-1.0, 0.0, 1.0])
    >>> legendre_polynomials_derivative(2, x)
    array([-3.,  0.,  3.])
    """
    x = np.asarray(x)

    if n == 0:
        return np.zeros_like(x)
    if n == 1:
        return np.ones_like(x)

    pnm1_p = np.zeros_like(x)
    pn_p = np.ones_like(x)
    for k in range(1, n):
        pnp1_p = (1.0 / k) * ((2 * k + 1) * x * pn_p - (k + 1) * pnm1_p)
        pnm1_p, pn_p = pn_p, pnp1_p

    return pnp1_p


def legendre_polynomials_second_derivative(
    n: int, x: ArrayLike
) -> NDArray[np.float64]:
    """
    Use the recursion formulas to compute the second derivative Legendre
    polynomials d²(Pn(x))

    - see Eq. 11 in Michels (1963)

    Parameters
    ----------
    n : int
        The Legendre polynomial order
    x : ndarray
        The x values of d²(Pn(x)), it must be 1-D

    Returns
    -------
    ndarray
        The second derivative Legendre series, a 1-D ndarray with the
        same shape as x

    Notes
    -----
    The numpy equivalent ->
    numpy.polynomial.legendre.Legendre.basis(n).deriv(2)(x)

    References
    ----------
    Michels, H. (1963). Abscissas and weight coefficients for Lobatto
    quadrature. Mathematics of Computation, 17(83), 237-244.

    Wiscombe, W. J. (1977). The delta-M method: Rapid yet accurate
    radiative flux calculations for strongly asymmetric phase functions.
    Journal of Atmospheric Sciences, 34(9), 1408-1422.

    Examples
    --------
    >>> import numpy as np
    >>> from pytrunc.utils import (
    ...     legendre_polynomials_second_derivative,
    ... )
    >>> x = np.array([-1.0, 0.0, 1.0])
    >>> legendre_polynomials_second_derivative(3, x)
    array([-15.,   0.,  15.])
    """
    x = np.asarray(x)

    if n == 0:
        return np.zeros_like(x)
    if n == 1:
        return np.zeros_like(x)
    if n == 2:
        return np.full_like(x, 3)

    pnm1_pp = np.zeros_like(x)
    pn_pp = np.full_like(x, 3)
    for k in range(2, n):
        pnp1_pp = (1.0 / (k - 1)) * (
            (2 * k + 1) * x * pn_pp - (k + 2) * pnm1_pp
        )
        pnm1_pp, pn_pp = pn_pp, pnp1_pp

    return pnp1_pp


def bessel_j1(
    x: ArrayLike, acc: float = 1e-8, max_iter: int = 50
) -> NDArray[np.float64]:
    """
    The Bessel first kind function J1(x) of order 1

    Parameters
    ----------
    x : ndarray
        The variable x of J1(x), it must be 1-D
    acc : float, optional
        The tolerance for numerical errors. Default is 1e-8
    max_iter : int, optional
        The maximum number of iterations trying to improve the error
        accuracy. Default is 50

    Returns
    -------
    ndarray
        The values of the Bessel function J1(x), a 1-D ndarray with the
        same shape as x

    Notes
    -----
    The scipy equivalent -> scipy.special.j1(x)

    Examples
    --------
    >>> import numpy as np
    >>> from pytrunc.utils import bessel_j1
    >>> x = np.array([0.0, 1.0, 2.0])
    >>> bessel_j1(x)
    array([0.        , 0.44005059, 0.57672481])
    """

    x = np.asarray(x, dtype=np.float64)
    j1 = np.zeros_like(x)

    x_small = x <= 35
    xs = x[x_small]
    if xs.size:
        for m in range(max_iter):
            with np.errstate(divide="ignore"):
                j1m = (
                    (-1) ** m / (math.factorial(m) * math.factorial(m + 1))
                ) * (xs / 2.0) ** (2 * m + 1.0)

            j1m = np.where(xs == 0, 0.0, j1m)
            j1[x_small] += j1m

            if np.max(np.abs(j1m)) < acc:
                break

    # For x > 35 the ascending series is unusable in double precision
    # (the alternating terms peak around 1e15 and cancellation destroys
    # every significant digit), use Hankel's asymptotic expansion with
    # the polynomial coefficients of Abramowitz & Stegun (eq. 9.4.6)
    x_large = x > 35
    xl = x[x_large]
    if xl.size:
        z = 8.0 / xl
        y = z * z
        p1 = 1.0 + y * (
            0.183105e-2
            + y * (
                -0.3516396496e-4
                + y * (0.2457520174e-5 + y * (-0.240337019e-6))
            )
        )
        q1 = 0.04687499995 + y * (
            -0.2002690873e-3
            + y * (
                0.8449199096e-5
                + y * (-0.88228987e-6 + y * 0.105787412e-6)
            )
        )
        chi = xl - 2.356194491
        j1[x_large] = np.sqrt(0.636619772 / xl) * (
            np.cos(chi) * p1 - z * np.sin(chi) * q1
        )

    return j1


def bessel_jn(
    x: ArrayLike, n: int, acc: float = 1e-8, max_iter: int = 50
) -> NDArray[np.float64]:
    """
    The Bessel first kind function Jn(x) of order n

    Parameters
    ----------
    x : ndarray
        The variable x of Jn(x), it must be 1-D
    n : int
        The Bessel first kind function order
    acc : float, optional
        The tolerance for numerical errors. Default is 1e-8
    max_iter : int, optional
        The maximum number of iterations trying to improve the error
        accuracy. Default is 50

    Returns
    -------
    ndarray
        The values of the Bessel function Jn(x), a 1-D ndarray with the
        same shape as x

    Notes
    -----
    The scipy equivalent -> scipy.special.jn(n,x)

    Examples
    --------
    >>> import numpy as np
    >>> from pytrunc.utils import bessel_jn
    >>> x = np.array([0.0, 1.0, 2.0])
    >>> bessel_jn(x, n=2)
    array([0.        , 0.11490348, 0.35283403])
    """

    x = np.asarray(x, dtype=np.float64)

    if np.any(x < 0):
        raise ValueError("The values of x must be >= 0")

    if n < 0:
        raise ValueError("The order n must be >= 0")

    jn = np.zeros_like(x)
    zero_mask = x == 0

    for m in range(max_iter):
        # divide: log(0) = -inf at x = 0; invalid: 0 * -inf = nan for
        # m = n = 0 — both samples are overwritten by the mask below
        with np.errstate(divide="ignore", invalid="ignore"):
            exp_term = -(gammaln(m + 1) + gammaln(m + n + 1)) + (
                2 * m + n
            ) * np.log(x / 2)

        jnm = (-1) ** m * np.exp(exp_term)

        # the x = 0 samples: Jn(0) = 1 for n = 0 and 0 otherwise
        if n == 0 and m == 0:
            jnm = np.where(zero_mask, 1.0, jnm)
        else:
            jnm = np.where(zero_mask, 0.0, jnm)

        jn += jnm

        if np.max(np.abs(jnm)) < acc:
            break

    return jn


def bessel_j1_derivative(
    x: ArrayLike, acc: float = 1e-8, max_iter: int = 50
) -> NDArray[np.float64]:
    """
    Compute the Bessel first kind derivative of order 1 d(J1(x))

    Parameters
    ----------
    x : ndarray
        The variable x of d(J1(x)), it must be 1-D
    acc : float, optional
        The tolerance for numerical errors. Default is 1e-8
    max_iter : int, optional
        The maximum number of iterations trying to improve the error
        accuracy. Default is 50

    Returns
    -------
    ndarray
        The values of the Bessel function derivative d(J1(x)), a 1-D
        ndarray with the same shape as x

    Notes
    -----
    The scipy equivalent -> scipy.special.jvp(1,x)

    Examples
    --------
    >>> import numpy as np
    >>> from pytrunc.utils import bessel_j1_derivative
    >>> x = np.array([0.0, 1.0, 2.0])
    >>> bessel_j1_derivative(x)
    array([ 0.5       ,  0.3251471 , -0.06447162])
    """

    j0 = bessel_jn(x, 0, acc=acc, max_iter=max_iter)
    j2 = bessel_jn(x, 2, acc=acc, max_iter=max_iter)

    return 0.5 * (j0 - j2)


def bessel_j1_roots(
    nb_roots: int, acc: float = 1e-8, max_iter: int = 50
) -> NDArray[np.float64]:
    """
    Find roots of Bessel first kind function j1(x) using Newton-Raphson
    iteration

    - First k approximations equation j1_roots ~ pi * (k + 1/4). See
      Baricz et al. (2025)

    Parameters
    ----------
    nb_roots : int
        The number of j1(x)=0 to find
    acc : float, optional
        The tolerance for numerical errors. Default is 1e-8
    max_iter : int, optional
        The maximum number of iterations trying to improve the error
        accuracy. Default is 50

    Returns
    -------
    ndarray
        The roots of j1(x), a 1-D ndarray of size nb_roots

    Notes
    -----
    The scipy equivalent -> scipy.special.jn_zeros(1,x)

    References
    ----------
    Baricz, Á., Kumar, P., & Ponnusamy, S. (2025). Asymptotic behavior
    of zeros of Bessel function derivatives. arXiv preprint
    arXiv:2510.12353.

    Examples
    --------
    >>> from pytrunc.utils import bessel_j1_roots
    >>> bessel_j1_roots(3)
    array([ 3.83170597,  7.01558667, 10.17346814])
    """

    if nb_roots < 1:
        raise ValueError("nb_roots must be >= 1")

    roots = np.zeros(nb_roots, dtype=np.float64)

    for k in range(1, nb_roots + 1):
        x0 = (k + 0.25) * math.pi
        x = x0
        for _ in range(max_iter):
            f = j1(x)
            df = jvp(1, x)
            dx = f / df
            x -= dx
            if abs(dx) < acc:
                break
        roots[k - 1] = x

    return roots


def _legendre_dp_d2p(
    n: int, x: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute d(Pn(x)) and d²(Pn(x)) with a single fused recurrence loop

    :meta private:

    Advances the recurrences of legendre_polynomials_derivative and
    legendre_polynomials_second_derivative together, saving one O(n)
    loop per Newton iteration of the Lobatto node solve. The float
    sequences are identical to the ones of the two public functions.
    """
    if n == 0:
        return np.zeros_like(x), np.zeros_like(x)
    if n == 1:
        return np.ones_like(x), np.zeros_like(x)

    # seeds: d(P0) = 0, d(P1) = 1, d²(P1) = 0, d²(P2) = 3
    pnm1_p = np.zeros_like(x)
    pn_p = np.ones_like(x)
    pnm1_pp = np.zeros_like(x)
    pn_pp = np.full_like(x, 3)
    for k in range(1, n):
        pnp1_p = (1.0 / k) * ((2 * k + 1) * x * pn_p - (k + 1) * pnm1_p)
        pnm1_p, pn_p = pn_p, pnp1_p
        if k >= 2:
            pnp1_pp = (1.0 / (k - 1)) * (
                (2 * k + 1) * x * pn_pp - (k + 2) * pnm1_pp
            )
            pnm1_pp, pn_pp = pn_pp, pnp1_pp

    return pn_p, pn_pp


def legendre_polynomials_derivative_roots(
    n: int, acc: float = 1e-8, max_iter: int = 50
) -> NDArray[np.float64]:
    """
    Find roots of legendre polynomial derivative d(Pn(x)), for x > -1
    and x < 1

    - Use of the Newton-Raphson iteration as in Michels (1963), see Eq.
      7 and 8

    Parameters
    ----------
    n : int
        The Legendre polynomial order
    acc : float, optional
        The tolerance for numerical errors. Default is 1e-8
    max_iter : int, optional
        The maximum number of iterations trying to improve the error
        accuracy. Default is 50

    Returns
    -------
    ndarray
        The roots of the legendre polynomial derivative d(Pn(x)), a 1-D
        ndarray of size n-1

    Notes
    -----
    - The numpy equivalent -> Legendre.basis(n).deriv().roots()
    - Faster than the numpy equivalent!

    References
    ----------
    Michels, H. (1963). Abscissas and weight coefficients for Lobatto
    quadrature. Mathematics of Computation, 17(83), 237-244.

    Examples
    --------
    >>> from pytrunc.utils import (
    ...     legendre_polynomials_derivative_roots,
    ... )
    >>> legendre_polynomials_derivative_roots(4)
    array([-0.65465367,  0.        ,  0.65465367])
    """

    x = np.sort(
        np.cos(
            jn_zeros(1, n - 1)
            / ((n + 1 - 0.5) ** 2 + ((math.pi**2 - 4) / (4 * math.pi**2)))
            ** 0.5
        )
    )
    for _ in range(max_iter):
        dp, d2p = _legendre_dp_d2p(n, x)
        dx = dp / d2p
        x -= dx
        if np.max(np.abs(dx)) < acc:
            break

    return x


def quadrature_lobatto(
    abscissa_min: float = -1, abscissa_max: float = 1, n: int = 100
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute the abscissas (sample points) and weights for Lobatto
    quadrature

    Parameters
    ----------
    abscissa_min : float, optional
        The min fixed abscissa. Default is -1
    abscissa_max : float, optional
        The max fixed abscissa. Default is 1
    n : int, optional
        The number of abscissas / weights. Default is 100

    Returns
    -------
    abscissas : ndarray
        The Lobatto quadrature abscissas, a 1-D ndarray of size n
    weights : ndarray
        The Lobatto quadrature weights, a 1-D ndarray of size n

    References
    ----------
    Michels, H. (1963). Abscissas and weight coefficients for Lobatto
    quadrature. Mathematics of Computation, 17(83), 237-244.

    Wiscombe, W. J. (1977). The delta-M method: Rapid yet accurate
    radiative flux calculations for strongly asymmetric phase functions.
    Journal of Atmospheric Sciences, 34(9), 1408-1422.

    Examples
    --------
    >>> from pytrunc.utils import quadrature_lobatto
    >>> xk, wk = quadrature_lobatto(n=4)
    >>> xk
    array([-1.       , -0.4472136,  0.4472136,  1.       ])
    >>> wk
    array([0.16666667, 0.83333333, 0.83333333, 0.16666667])
    """

    if n < 2:
        raise ValueError(
            "the legendre polynomial order must be >= 2 for Lobatto quadrature"
        )

    if abscissa_max <= abscissa_min:
        raise ValueError("abscissa_max must be > to abscissa_min")

    abscissas, weights = _quadrature_lobatto_standard(n)

    # rescale if min and max values different to -1 and 1
    if abscissa_min != -1 or abscissa_max != 1:
        alpha = (abscissa_max - abscissa_min) / 2.0
        abscissas = (abscissas + 1) * alpha + abscissa_min
        weights = weights * alpha
    else:
        abscissas = abscissas.copy()
        weights = weights.copy()

    return abscissas, weights


@functools.lru_cache(maxsize=8)
def _quadrature_lobatto_standard(
    n: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute the Lobatto abscissas and weights on the standard interval
    [-1, 1]

    The Newton-Raphson node solve is expensive (~50 ms for n ~ 2000), so
    the result is cached per n. The returned arrays are read-only;
    callers must copy before modifying (quadrature_lobatto does).

    :meta private:
    """

    # Get lobatto abcissa
    abscissas_int = legendre_polynomials_derivative_roots(n - 1)
    abscissas = np.concatenate(([-1.0], abscissas_int, [1.0]))

    weights = np.empty_like(abscissas)
    weights[0] = weights[-1] = 2.0 / (n * (n - 1.0))
    pnm1 = legendre_polynomials(n - 1, abscissas[1:-1])
    weights[1:-1] = weights[0] / (pnm1**2)

    abscissas.flags.writeable = False
    weights.flags.writeable = False
    return abscissas, weights


def integrate_lobatto(
    f: NDArray[np.float64],
    x: NDArray[np.float64],
    lp: int | None = None,
    xk: NDArray[np.float64] | None = None,
    wk: NDArray[np.float64] | None = None,
    assume_sorted: bool = False,
) -> float:
    """
    Integrate using lobatto quadrature

    Parameters
    ----------
    f : ndarray
        The ordinates of the function (array to be integrated), it must
        be 1-D
    x : ndarray
        The abscissas, it must be 1-D
    lp : int or None, optional
        The number of lobatto points for the integration. Default is
        None, meaning lp is equal to len(x)
    xk : ndarray or None, optional
        Force the Lobatto quadrature abscissas, it must be 1-D and of
        size lp. Considered only if wk is also provided
    wk : ndarray or None, optional
        Force the Lobatto weights, it must be 1-D and of size lp.
        Considered only if xk is also provided
    assume_sorted : bool, optional
        If True, the x array is assumed to be sorted in ascending order.
        Default is False

    Returns
    -------
    float
        The estimated integral calculated using the Lobatto quadrature

    Examples
    --------
    >>> import numpy as np
    >>> from pytrunc.utils import integrate_lobatto
    >>> theta = np.linspace(0.0, np.pi, 1801)
    >>> integrate_lobatto(np.sin(theta), theta, assume_sorted=True)
    1.999999497717348
    """

    if lp is None:
        lp = len(x)

    # sort x and modify f consequently
    if assume_sorted:
        x_sorted = x
        f_sorted = f
    else:
        id_sorted = np.argsort(x)
        x_sorted = x[id_sorted]
        f_sorted = f[id_sorted]
    # lobatto distribution and interpolation
    if (xk is None) or (wk is None):
        xk, wk = quadrature_lobatto(
            abscissa_min=x_sorted[0], abscissa_max=x_sorted[-1], n=lp
        )

    f_ = np.interp(xk, x_sorted, f_sorted)

    # return integral
    return float(np.sum(wk * f_))
