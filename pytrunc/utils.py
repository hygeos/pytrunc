import numpy as np
import math


def legendre_polynomials(n, x):
    """
    Use the recursion formulas to compute the Legendre polynomials Pn(x)

    - see Eq. 9 in [1]

    Parameters
    ----------
    n : int
        The Legendre polynomial order
    x : 1-D ndarray
        The x values of Pn(x)
    
    Returns
    -------
    P : 1-D ndarray
        The Legendre series

    Notes
    -----
    The numpy equivalent -> numpy.polynomial.legendre.Legendre.basis(n)(x)
    
    References
    ----------

    - [1] Michels, H. (1963). Abscissas and weight coefficients for Lobatto quadrature. 
          Mathematics of Computation, 17(83), 237-244.
    
    - [2] Wiscombe, W. J. (1977). The delta-M method: Rapid yet accurate radiative flux calculations 
          for strongly asymmetric phase functions. Journal of Atmospheric Sciences, 34(9), 1408-1422.


    """
    x = np.asarray(x)

    if n == 0: return np.ones_like(x)
    if n == 1: return x

    P0 = np.ones_like(x)
    P1 = x

    Pnm1 = P0.copy()
    Pn = P1.copy()
    for k in range(1, n):
        Pnp1 = (1. / (k + 1)) * ( ((2*k + 1)*x*Pn - k*Pnm1) )
        Pnm1, Pn = Pn, Pnp1

    return Pnp1


def legendre_polynomials_derivative(n, x):
    """
    Use the recursion formulas to compute the derivative Legendre polynomials d(Pn(x))

    - see Eq. 10 in [1]

    Parameters
    ----------
    n : int
        The Legendre polynomial order
    x : 1-D ndarray
        The x values of d(Pn(x))
    
    Returns
    -------
    P : 1-D ndarray
        The derivative Legendre series
    
    Notes
    -----
    The numpy equivalent -> numpy.polynomial.legendre.Legendre.basis(n).deriv(1)(x)
    
    References
    ----------

    - [1] Michels, H. (1963). Abscissas and weight coefficients for Lobatto quadrature. 
          Mathematics of Computation, 17(83), 237-244.
    
    - [2] Wiscombe, W. J. (1977). The delta-M method: Rapid yet accurate radiative flux calculations 
          for strongly asymmetric phase functions. Journal of Atmospheric Sciences, 34(9), 1408-1422.

    """
    x = np.asarray(x)

    if n == 0: return np.ones_like(x)
    if n == 1: return x

    P0_p = np.zeros_like(x)
    P1_p = np.ones_like(x)

    Pnm1_p = P0_p.copy()
    Pn_p = P1_p.copy()
    for k in range(1, n):
        Pnp1_p = (1. / k) * ( ((2*k + 1)*x*Pn_p - (k + 1)*Pnm1_p) )
        Pnm1_p, Pn_p = Pn_p, Pnp1_p

    return Pnp1_p


def legendre_polynomials_second_derivative(n, x):
    """
    Use the recursion formulas to compute the second derivative Legendre polynomials d²(Pn(x))

    - see Eq. 11 in [1]

    Parameters
    ----------
    n : int
        The Legendre polynomial order
    x : 1-D ndarray
        The x values of d²(Pn(x))
    
    Returns
    -------
    P : 1-D ndarray
        The second derivative Legendre series
    
    Notes
    -----
    The numpy equivalent -> numpy.polynomial.legendre.Legendre.basis(n).deriv(2)(x)
    
    References
    ----------

    - [1] Michels, H. (1963). Abscissas and weight coefficients for Lobatto quadrature. 
          Mathematics of Computation, 17(83), 237-244.
    
    - [2] Wiscombe, W. J. (1977). The delta-M method: Rapid yet accurate radiative flux calculations 
          for strongly asymmetric phase functions. Journal of Atmospheric Sciences, 34(9), 1408-1422.

    """
    x = np.asarray(x)

    if n == 0: return np.ones_like(x)
    if n == 1: return x

    P1_pp = np.zeros_like(x)
    P2_pp = np.full_like(x,3)

    Pnm1_pp = P1_pp.copy()
    Pn_pp = P2_pp.copy()
    for k in range(2, n):
        Pnp1_pp = (1. / (k - 1)) * ( ((2*k + 1)*x*Pn_pp - (n + 2)*Pnm1_pp) )
        Pnm1_pp, Pn_pp = Pn_pp, Pnp1_pp

    return Pnp1_pp


def bessel_j1(x, acc=1e-8, max_iter=50):
    """
    The Bessel first kind function J1(x) of order 1
    
    Parameters
    ----------
    x : 1-D ndarray
        The variable x of J1(x)
    acc : float, optional
        The tolerance for numerical errors. Default is 1e-8.
    
    Returns
    -------
    J1 : 1-D ndarray
        The values of the Bessel function J1(x)
    """

    j1 = np.zeros_like(x, dtype=np.float64)
    for m in range(max_iter):
        j1m = ( (-1)**m / ( (math.factorial(m) * math.gamma(m + 2)) ) ) * (x / 2.)**(2.*m + 1.) 
        j1 += j1m
        if np.max(np.abs(j1m)) < acc:
            break

    return j1