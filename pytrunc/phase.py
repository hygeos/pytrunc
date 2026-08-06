"""
Scattering phase functions and Legendre moments.

This module provides analytic scattering phase functions
(Henyey-Greenstein, two-term Henyey-Greenstein and Fournier-Forand) and
the calculation of the phase function Legendre moments, either
numerically for an arbitrary phase function or analytically for the
Henyey-Greenstein family.
"""

import math
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import simpson, trapezoid

from pytrunc.utils import (
    integrate_lobatto,
    legendre_polynomials,
    quadrature_lobatto,
)


def henyey_greenstein(
    theta: ArrayLike,
    g: float,
    theta_unit: str = "deg",
    normalize: float | None = None,
) -> NDArray[np.float64]:
    """
    Compute the Henyey-Greenstein phase matrix

    Parameters
    ----------
    theta : ndarray
        The phase matrix angles, it must be 1-D. See the theta_unit
        parameter for the unit
    g : float
        The Henyey-Greenstein parameter g (measures the asymmetry of the
        phase matrix)
    theta_unit : str, optional
        The unit of the theta angles. Default is 'deg', other choice is
        'rad'
    normalize : float or None, optional
        The normalization value of the integral of F_HG(θ)dcosθ, where
        F_HG(θ) is the phase matrix. The scipy simpson function is used
        for the normalization. Default is None, meaning no normalization

    Returns
    -------
    ndarray
        The phase matrix, a 1-D ndarray with the same shape as theta

    Notes
    -----
    The Henyey-Greenstein equation:

    - :math:`F_HG(θ) =
      (1/(4*pi))*[(1-g**2) / (1+g**2-(2*g*cos(θ)))**(3/2)]`
    - By default the integral of F_HG(θ)dcosθ is equal to 1/(2*pi). The
      integral value can be different due to a very low discretization
      of θ and/or a high g value. The use of the normalize parameter can
      be useful to renormalize the phase function

    References
    ----------
    Henyey, L. G., & Greenstein, J. L. (1941). Diffuse radiation in the
    galaxy. Astrophysical Journal, 93, 70-83.

    Ocean Optics Web Book: https://www.oceanopticsbook.info

    Examples
    --------
    >>> import numpy as np
    >>> from pytrunc.phase import henyey_greenstein
    >>> theta = np.linspace(0.0, 180.0, 3)
    >>> henyey_greenstein(theta, g=0.85)
    array([6.54303655e+00, 9.76819403e-03, 3.48769050e-03])
    """

    if theta_unit == "rad":
        mu = np.cos(theta)
    elif theta_unit == "deg":
        mu = np.cos(np.deg2rad(theta))
    else:
        raise ValueError(
            "The accepted values for parameter theta_unit are: 'deg' or 'rad'"
        )

    phase = (1.0 / (4 * math.pi)) * (
        (1 - g * g) / (1 + g * g - (2 * g * mu)) ** (1.5)
    )
    if normalize is not None:
        idmu = np.argsort(mu)
        phase = (normalize * phase) / simpson(phase[idmu], x=mu[idmu])

    return phase


def two_term_henyey_greenstein(
    theta: ArrayLike,
    g1: float,
    g2: float,
    f: float,
    theta_unit: str = "deg",
    normalize: float | None = None,
) -> NDArray[np.float64]:
    """
    Compute the two-term Henyey-Greenstein phase matrix

    Parameters
    ----------
    theta : ndarray
        The phase matrix angles, it must be 1-D. See the theta_unit
        parameter for the unit
    g1 : float
        The first H-G term parameter g (forward part)
    g2 : float
        The second H-G term parameter g (backward part)
    f : float
        The fraction parameter between the two H-G terms (see notes)
    theta_unit : str, optional
        The unit of the theta angles. Default is 'deg', other choice is
        'rad'
    normalize : float or None, optional
        The normalization value of the integral of F_TTHG(θ)dcosθ, where
        F_TTHG(θ) is the phase matrix. The scipy simpson function is
        used for the normalization. Default is None, meaning no
        normalization

    Returns
    -------
    ndarray
        The phase matrix, a 1-D ndarray with the same shape as theta

    Notes
    -----
    The two term Henyey-Greenstein equation:

    - :math:`F_TTHG(θ) = f*F_HG1(θ) + (1-f)*F_HG2(θ)`
    - By default the integral of F_TTHG(θ)dcosθ is equal to 1/(2*pi).
      The integral value can be different due to a very low
      discretization of θ and/or a high g value. The use of the
      normalize parameter can be useful to renormalize the phase
      function

    References
    ----------
    Irvine, W. M. (1965). Multiple scattering by large particles (No.
    NASA-CR-64638).

    Examples
    --------
    >>> import numpy as np
    >>> from pytrunc.phase import two_term_henyey_greenstein
    >>> theta = np.linspace(0.0, 180.0, 3)
    >>> two_term_henyey_greenstein(theta, g1=0.85, g2=-0.6, f=0.9)
    array([5.88997629, 0.01200253, 0.08271639])
    """

    f_hg1 = henyey_greenstein(
        theta=theta, g=g1, theta_unit=theta_unit, normalize=normalize
    )
    f_hg2 = henyey_greenstein(
        theta=theta, g=g2, theta_unit=theta_unit, normalize=normalize
    )
    phase = f * f_hg1 + (1 - f) * f_hg2

    return phase


def fournier_forand(
    theta: ArrayLike,
    n: float,
    mu: float,
    theta_unit: str = "deg",
    normalize: float | None = None,
) -> NDArray[np.float64]:
    """
    Compute the Fournier-Forand phase matrix

    Parameters
    ----------
    theta : ndarray
        The phase matrix angles, it must be 1-D. See the theta_unit
        parameter for the unit
    n : float
        The real index of refraction of the particles
    mu : float
        The slope parameter of the hyperbolic (Junge) particle size
        distribution (typically between 3 and 5 for oceanic particles)
    theta_unit : str, optional
        The unit of the theta angles. Default is 'deg', other choice is
        'rad'
    normalize : float or None, optional
        The normalization value of the integral of F_FF(θ)dcosθ, where
        F_FF(θ) is the phase matrix. The scipy simpson function is used
        for the normalization. Default is None, meaning no normalization

    Returns
    -------
    ndarray
        The phase matrix, a 1-D ndarray with the same shape as theta

    Notes
    -----
    The Fournier-Forand equation:

    - :math:`F_FF(θ) = [1 / (4*pi*(1-δ)**2*δ**v)] *
      [v*(1-δ) - (1-δ**v) + (δ*(1-δ**v) - v*(1-δ))*sin(θ/2)**(-2)]
      + [(1-δ_180**v) / (16*pi*(δ_180-1)*δ_180**v)] *
      (3*cos(θ)**2 - 1)`
    - with :math:`v = (3-μ)/2` and
      :math:`δ = (4 / (3*(n-1)**2))*sin(θ/2)**2`, where δ_180 is δ
      evaluated at θ = 180°
    - By default the integral of F_FF(θ)dcosθ is equal to 1/(2*pi). The
      integral value can be different due to a very low discretization
      of θ and/or a strong forward peak. The use of the normalize
      parameter can be useful to renormalize the phase function
    - F_FF(θ) diverges as θ → 0: at θ = 0 the result is NaN/inf (with
      numpy invalid-value warnings). Use a grid starting above 0, or
      overwrite the forward peak values before integrating

    References
    ----------
    Fournier, G. R., & Forand, J. L. (1994). Analytic phase function for
    ocean water. In Ocean Optics XII (Vol. 2258, pp. 194-201). SPIE.

    Ocean Optics Web Book: https://www.oceanopticsbook.info

    Examples
    --------
    >>> import numpy as np
    >>> from pytrunc.phase import fournier_forand
    >>> theta = np.linspace(10.0, 170.0, 3)
    >>> fournier_forand(theta, n=1.117, mu=3.695)
    array([1.24049454, 0.0065938 , 0.00479607])
    """

    if theta_unit == "rad":
        theta_rad = np.asarray(theta, dtype=np.float64)
    elif theta_unit == "deg":
        theta_rad = np.deg2rad(theta)
    else:
        raise ValueError(
            "The accepted values for parameter theta_unit are: 'deg' or 'rad'"
        )

    v = (3 - mu) / 2
    sin_half_sq = np.sin(theta_rad / 2) ** 2
    delta = (4 / (3 * (n - 1) ** 2)) * sin_half_sq
    delta_180 = 4 / (3 * (n - 1) ** 2)

    phase = (1 / (4 * math.pi * (1 - delta) ** 2 * delta**v)) * (
        v * (1 - delta)
        - (1 - delta**v)
        + (delta * (1 - delta**v) - v * (1 - delta)) / sin_half_sq
    ) + (
        (1 - delta_180**v)
        / (16 * math.pi * (delta_180 - 1) * delta_180**v)
    ) * (3 * np.cos(theta_rad) ** 2 - 1)
    if normalize is not None:
        mu_cos = np.cos(theta_rad)
        idmu = np.argsort(mu_cos)
        phase = (normalize * phase) / simpson(phase[idmu], x=mu_cos[idmu])

    return phase


def calc_moments(
    phase: NDArray[np.float64],
    theta: NDArray[np.float64],
    m_max: int,
    method: str = "lobatto",
    theta_unit: str = "deg",
    normalize: bool = False,
    xk: NDArray[np.float64] | None = None,
    wk: NDArray[np.float64] | None = None,
    pl_costh: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """
    Calculate the phase matrix moments until m_max moment

    Parameters
    ----------
    phase : ndarray
        The phase matrix, it must be 1-D
    theta : ndarray
        The phase matrix angles, it must be 1-D. See the theta_unit
        parameter for the unit
    m_max : int
        The maximum moment number to compute, i.e., compute m[0], ...,
        m[m_max]
    method : str, optional
        The method used to calculate the moments. Default is 'lobatto'
        (very efficient with a "gauss kind" theta distribution), other
        choices are 'simpson' and 'trapezoid' (efficient with a regular
        theta distribution, use the scipy simpson and trapezoid
        functions)
    theta_unit : str, optional
        The unit of the theta angles. Default is 'deg', other choice is
        'rad'
    normalize : bool, optional
        If True, normalize such that the first moment is exactly equal
        to 1. Default is False
    xk : ndarray or None, optional
        Force the Lobatto quadrature abscissas, it must be 1-D.
        Considered only if wk is also provided
    wk : ndarray or None, optional
        Force the Lobatto weights, it must be 1-D. Considered only if xk
        is also provided
    pl_costh : ndarray or None, optional
        Force the Legendre polynomials values of cos(theta). The 2-D
        ndarray shape must be (m_max+1, len(theta))

    Returns
    -------
    ndarray
        The computed phase moments, a 1-D ndarray of size m_max + 1

    Notes
    -----
    - See Eq. A2 in Wiscombe (1977) for the moment computation using the
      Lobatto quadrature in [0,pi]
    - For the Lobatto quadrature abscissas and weights calculation see
      Michels (1963)

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
    >>> from pytrunc.phase import calc_moments, henyey_greenstein
    >>> theta = np.linspace(0.0, 180.0, 1801)
    >>> phase = henyey_greenstein(theta, g=0.85, normalize=2)
    >>> calc_moments(phase, theta, m_max=3)
    array([0.99998936, 0.84998937, 0.72248935, 0.61411436])
    """

    methods_ok = ["lobatto", "simpson", "trapezoid"]

    if method not in methods_ok:
        raise ValueError(f"Only available methods are: {methods_ok}")

    if theta_unit == "deg":
        theta = np.deg2rad(theta)
    elif theta_unit != "rad":
        raise ValueError(
            "The accepted values for parameter theta_unit are: 'deg' or 'rad'"
        )

    if theta[0] < 0 or theta[-1] > np.pi:
        warnings.warn(
            "The range of theta must be [0, π] (rad unit) or [0,180] "
            "(deg unit)",
            UserWarning,
            stacklevel=2,
        )

    # initialize moments
    chi = np.zeros(m_max + 1)

    if method == "lobatto":
        nth = len(phase)
        if nth < 2:
            raise ValueError(
                "The phase size must be >= 2 for Lobatto quadrature"
            )

        if (xk is None) or (wk is None):
            xk, wk = quadrature_lobatto(
                abscissa_min=0.0, abscissa_max=math.pi, n=nth
            )

        if pl_costh is None:
            mu = np.cos(theta)
        sin_th = np.sin(theta)

        for deg in range(m_max + 1):
            if pl_costh is None:
                pl_mu = legendre_polynomials(deg, mu)
            else:
                pl_mu = pl_costh[deg]
            chi[deg] = 0.5 * integrate_lobatto(
                phase * sin_th * pl_mu, theta, xk=xk, wk=wk
            )

    else:
        mu = np.cos(theta)
        idmu = np.argsort(mu)
        for deg in range(m_max + 1):
            if pl_costh is None:
                pl_mu = legendre_polynomials(deg, mu[idmu])
            else:
                pl_mu = pl_costh[deg][idmu]
            if method == "simpson":
                chi[deg] = 0.5 * simpson(phase[idmu] * pl_mu, x=mu[idmu])
            elif method == "trapezoid":
                chi[deg] = 0.5 * trapezoid(phase[idmu] * pl_mu, x=mu[idmu])

    # normalization
    if normalize:
        chi /= chi[0]

    return chi


def calc_hg_moments(g: float, m_max: int) -> NDArray[np.float64]:
    """
    Compute exact Henyey-Greenstein phase moments

    - see Eq. 8 in Kattawar (1975)

    Parameters
    ----------
    g : float
        The Henyey-Greenstein parameter g (measures the asymmetry of the
        phase matrix)
    m_max : int
        The maximum moment number to compute, i.e., compute m[0], ...,
        m[m_max]

    Returns
    -------
    ndarray
        The phase moments, a 1-D ndarray of size m_max + 1

    References
    ----------
    Kattawar, G. W. (1975). A three-parameter analytic phase function
    for multiple scattering calculations. Journal of Quantitative
    Spectroscopy and Radiative Transfer, 15(9), 839-849.

    Examples
    --------
    >>> from pytrunc.phase import calc_hg_moments
    >>> calc_hg_moments(g=0.85, m_max=3)
    array([1.      , 0.85    , 0.7225  , 0.614125])
    """

    return np.array([g**n for n in range(m_max + 1)])


def calc_tthg_moments(
    g1: float, g2: float, f: float, m_max: int
) -> NDArray[np.float64]:
    """
    Compute exact Two-term Henyey-Greenstein phase moments

    - see Eq. 11 in Kattawar (1975)

    Parameters
    ----------
    g1 : float
        The first H-G term parameter g (forward part)
    g2 : float
        The second H-G term parameter g (backward part)
    f : float
        The fraction parameter between the two H-G terms
    m_max : int
        The maximum moment number to compute, i.e., compute m[0], ...,
        m[m_max]

    Returns
    -------
    ndarray
        The phase moments, a 1-D ndarray of size m_max + 1

    References
    ----------
    Kattawar, G. W. (1975). A three-parameter analytic phase function
    for multiple scattering calculations. Journal of Quantitative
    Spectroscopy and Radiative Transfer, 15(9), 839-849.

    Examples
    --------
    >>> from pytrunc.phase import calc_tthg_moments
    >>> calc_tthg_moments(g1=0.85, g2=-0.6, f=0.9, m_max=3)
    array([1.       , 0.705    , 0.68625  , 0.5311125])
    """

    return np.array([(f * g1**n + (1 - f) * g2**n) for n in range(m_max + 1)])
