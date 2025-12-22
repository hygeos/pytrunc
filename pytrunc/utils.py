import numpy as np


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