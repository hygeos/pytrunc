import numpy as np
import math
from scipy.special import gammaln
from scipy.special import j1, jvp, jn_zeros


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

    if n == 0: return np.zeros_like(x)
    if n == 1: return np.ones_like(x)

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

    if n == 0: return np.zeros_like(x)
    if n == 1: return np.zeros_like(x)
    if n == 2: return np.full_like(x,3)

    P1_pp = np.zeros_like(x)
    P2_pp = np.full_like(x,3)

    Pnm1_pp = P1_pp.copy()
    Pn_pp = P2_pp.copy()
    for k in range(2, n):
        Pnp1_pp = (1. / (k - 1)) * ( ((2*k + 1)*x*Pn_pp - (k + 2)*Pnm1_pp) )
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
    max_iter : float, optional
        The maximun number of iteration trying to improve the error accuracy
    
    Returns
    -------
    J1 : 1-D ndarray
        The values of the Bessel function J1(x)

    Notes
    -----
    The scipy equivalent -> scipy.special.j1(x)
    """
    
    x = np.asarray(x, dtype=np.float64)
    j1 = np.zeros_like(x)

    x_small = x <= 35
    xs = x[x_small]
    if np.any(x_small):
        for m in range(max_iter):
            with np.errstate(divide='ignore'):
                j1m = ( (-1)**m / ( (math.factorial(m) * math.factorial(m + 1)) ) ) * (xs / 2.)**(2*m + 1.)

            j1m = np.where(x == 0, 0.0, j1m)
            j1[x_small] += j1m

            if np.max(np.abs(j1m)) < acc:
                break

    x_large = x > 35
    xl = x[x_large]
    if np.any(x_large):
        for m in range(max_iter):
            with np.errstate(divide='ignore'):
                exp_term = - (gammaln(m + 1) + gammaln(m + 2)) + (2*m + 1) * np.log(xl / 2)
            j1m = (-1)**m * np.exp(exp_term)

            j1m = np.where(x == 0, 0.0, j1m)
            j1[x_small] += j1m

            if np.max(np.abs(j1m)) < acc:
                break

    return j1


def bessel_jn(x, n, acc=1e-8, max_iter=50):
    """
    The Bessel first kind function Jn(x) of order n
    
    Parameters
    ----------
    x : 1-D ndarray
        The variable x of Jn(x)
    n : float
        The Bessel first kind function order
    acc : float, optional
        The tolerance for numerical errors. Default is 1e-8.
    max_iter : float, optional
        The maximun number of iteration trying to improve the error accuracy
    
    Returns
    -------
    Jn : 1-D ndarray
        The values of the Bessel function Jn(x)

    Notes
    -----
    The scipy equivalent -> scipy.special.jn(n,x)
    """


    x = np.asarray(x, dtype=np.float64)

    if np.any(x < 0):
        raise ValueError("The values of x must be >= 1")
    
    if n < 0:
        raise ValueError('The order n must be >= 0')

    jn = np.zeros_like(x)

    for m in range(max_iter):

        with np.errstate(divide='ignore'):
            exp_term = - (gammaln(m + 1) + gammaln(m + n + 1)) + (2*m + n) * np.log(np.abs(x / 2))

        jnm = (-1)**m * np.exp(exp_term)
        #jnm = ( (-1)**m / ( (math.factorial(m) * math.gamma(m + n + 1)) ) ) * (x / 2.)**(2*m + n) 

        if n == 0:
            jnm = np.where((x == 0) & (m == 0), 1.0, jnm)
            jnm = np.where((x == 0) & (m > 0), 0.0, jnm)
        else:
            jnm = np.where(x == 0, 0.0, jnm)

        jn += jnm

        if np.max(np.abs(jnm)) < acc:
            break

    return jn


def bessel_j1_derivative(x, acc=1e-8, max_iter=50):
    """
    Compute the Bessel first kind derivative of order 1 d(J1(x))

    Parameters
    ----------
    x : 1-D ndarray
        The variable x of d(J1(x))
    acc : float, optional
        The tolerance for numerical errors. Default is 1e-8.
    max_iter : float, optional
        The maximun number of iteration trying to improve the error accuracy
    
    Returns
    -------
    dj1 : 1-D ndarray
        The values of the Bessel function derivative d(J1(x))

    Notes
    -----
    The scipy equivalent -> scipy.special.jvp(1,x)
    """

    j0 = bessel_jn(x,0, acc=acc, max_iter=max_iter)
    j2 = bessel_jn(x,2, acc=acc, max_iter=max_iter)

    return 0.5 * ( j0 - j2 )


def bessel_j1_roots(nb_roots, acc=1e-8, max_iter=50):
    """
    Find roots of Bessel first kind function j1(x) using Newton-Raphson iteration
    
    - First k approximations equation ji_roots ~ π * ( k + π/2 + π/4 ). See ref[1]

    Parameters
    ----------
    nb_roots : int
        The number of j1(x)=0 to find
    acc : float, optional
        The tolerance for numerical errors. Default is 1e-8.
    max_iter : float, optional
        The maximun number of iteration trying to improve the error accuracy

    Returns
    -------
    roots : 1-D ndarray
        The roots of the function f(x)

    Notes
    -----
    The scipy equivalent -> scipy.special.jn_zeros(1,x)
        
    References
    ----------

    - [1] Baricz, Á., Kumar, P., & Ponnusamy, S. (2025). Asymptotic behavior of zeros 
          of Bessel function derivatives. arXiv preprint arXiv:2510.12353.
    """

    if nb_roots < 1:
        raise ValueError("nb_roots must be >= 1")

    
    roots = np.zeros(nb_roots, dtype=np.float64)

    for k in range (1, nb_roots+1):
        x0 = (k + 0.25) * math.pi
        x = x0
        for i in range (max_iter):
            f = j1(x)
            df = jvp(1,x)
            dx = f / df
            x -= dx
            if abs(dx) < acc:
                break
        roots[k-1] = x

    return roots


def legendre_polynomials_derivative_roots(n, acc=1e-8, max_iter=50):
    """
    Find roots of legendre polynomial derivative d(Pn(x)), for x > -1 and x < 1

    - Use of Newton-raphson iteration as [1], see Eq.7 and 8.

    Parameters
    ----------
    n : int
        The Legendre polynomial order
    acc : float, optional
        The tolerance for numerical errors. Default is 1e-8.
    max_iter : float, optional
        The maximun number of iteration trying to improve the error accuracy
    
    Returns
    -------
    roots : 1-D ndarray
        The roots of legendre polynomial derivative d(Pn(x))

    Notes
    -----
    
    - The numpy equivalent -> Legendre.basis(n).deriv().roots()
    - Faster than the numpy equivalent!

    References
    ----------

    - [1] Michels, H. (1963). Abscissas and weight coefficients for Lobatto quadrature. 
          Mathematics of Computation, 17(83), 237-244.
    """

    x0 = np.sort(np.cos(jn_zeros(1, n-1) / ( (n+1-0.5)**2 + ((math.pi**2-4)/(4*math.pi**2)) )**0.5))
    x = np.asarray(x0)
    for i in range (max_iter):
        dp = legendre_polynomials_derivative(n,x)
        d2p = legendre_polynomials_second_derivative(n,x)
        dx = dp / d2p
        x -= dx
        if np.max(np.abs(dx)) < acc:
            break

    return x


def quadrature_lobatto(abscissa_min=-1, abscissa_max=1, n=100):
    """
    Compute the abscissas (sample points) and weigths for Lobatto quadrature.

    Parameters
    ----------
    abscissa_min : float
        The min fixed abscissa
    abscissa_max : float
        The max fixed abscissa
    n : int, optional
        The number of abscissas / weights

    Returns
    -------
    abscissas : 1-D ndarray
        The Lobatto quadrature abscissas
    weights : 1-D ndarray
        The Lobatto quadrature weights

    References
    ----------

    - [1] Michels, H. (1963). Abscissas and weight coefficients for Lobatto quadrature. 
          Mathematics of Computation, 17(83), 237-244.
    
    - [2] Wiscombe, W. J. (1977). The delta-M method: Rapid yet accurate radiative flux calculations 
          for strongly asymmetric phase functions. Journal of Atmospheric Sciences, 34(9), 1408-1422.
    """

    if n < 2:
        raise ValueError("the legendre polynomial order must be >= 2 for Lobatto quadrature")

    if abscissa_max <= abscissa_min:
        raise ValueError("abscissa_max must be > to abscissa_min")

    # Get lobatto abcissa
    abscissas_int = legendre_polynomials_derivative_roots(n-1)
    abscissas = np.concatenate(([-1.0], abscissas_int, [1.0]))

    weights = np.zeros_like(abscissas)
    weights[0] = weights[-1] = 2. / (n * (n - 1.))
    Pnm1 = legendre_polynomials(n-1, abscissas[1:-1])
    weights[1:-1] = weights[0] / (Pnm1 ** 2)

    # rescale if min and max values different to -1 and 1
    if abscissa_min != -1 or abscissa_max != 1:
        alpha = (abscissa_max - abscissa_min) / 2.
        abscissas = (abscissas + 1) * alpha + abscissa_min
        alpha = (abscissa_max - abscissa_min) / 2.
        weights *= alpha

    return abscissas, weights


def integrate_lobatto(f, x, lp=None, xk=None, wk=None):
    """
    Integrate using lobatto quadrature

    Parameters
    ----------
    f : 1-D ndarray
        The ordinates of the function (array to be integrated).
    x : 1-D ndarray
        The abscissas
    lp : None | int, optional
        The number of lobatto points for the integration. If None lp = len(x).
    xk : None | 1-D ndarray
        Force the Lobatto quadrature abscissas. Considered if wk is also provided.
    wk : None | 1-D ndarray
        Force the Lobatto weights. Considered if xk is also provided.
    Return
    ------
    int : float
        The estimated integral calculated using the Lobatto quadrature 
    """

    if lp is None: lp = len(x)

    # sort x and modify f consequently
    id_sorted = np.argsort(x)
    x_sorted = x[id_sorted]
    f_sorted = f[id_sorted]

    # lobatto distribution and interpolation
    if ((xk is None) or (wk is None)):
        xk, wk = quadrature_lobatto(abscissa_min=x_sorted[0], abscissa_max=x_sorted[-1], n=lp)
    f_ = np.interp(xk, x_sorted, f_sorted)

    # return integral
    return np.sum(wk * f_)