import math
import numpy as np
from scipy.integrate import simpson
from pytrunc.utils import legendre_polynomials, quadrature_lobatto


def henyey_greenstein(theta, g, theta_unit='deg', normalize=None):
    """
    Compute the Henyey-Greenstein phase matrix

    Parameters
    ----------
    theta : 1-D ndarray
        The phase matrix angles. See parameter `theta_unit`
    g : float
        The Henyey-Greenstein parameter g (measures the asymmetry of the phase matrix)
    theta_unit : str, optional
        The unit for theta angles: 
        - 'deg' (default value)
        - 'rad'
    normalize : None | float, optional
        The normalization value of the integral ∫F_HG(θ)dcosθ, where F_HG(θ) is the phase matrix.  
        The scipy simpson function is used for the normalization.
    Return
    ------
    F_HG : 1-D ndarray
        The phase matrix.

    Notes
    -----
    The Henyey-Greenstein equation:
    
    - :math:`F_HG(θ) = (1/π) * [ (1 - g**2) / (1 + g**2 - (2*g*cos(θ)))**(3/2) ]`
    
    - By default the integral ∫F(θ)dcosθ = 1/2π.  
    The integral value can be different due to a very low dicretization of θ and/or a high g value.  
    The use of the parameter `normalize` can be useful to renormalize the phase function.

    References
    ----------
    Henyey, L. G., & Greenstein, J. L. (1941). Diffuse radiation in the galaxy.  
    Astrophysical Journal, vol. 93, p. 70-83 (1941)., 93, 70-83.

    `other reference <http://www.oceanopticsbook.info/view/scattering/level-2/the-henyey-greenstein-phase-function>`_
    """
        
    if theta_unit == 'rad':
        mu = np.cos(theta)
    elif theta_unit == 'deg':
        mu = np.cos(np.deg2rad(theta))
    else:
        raise ValueError("The accepted values for parameter theta_unit are: 'deg' or 'rad'")
    
 
    phase = (1./(4*math.pi)) * (  (1-g*g) / (1 + g*g - (2*g*mu))**(1.5)  )
    if normalize is not None: phase = (normalize * phase) / simpson(phase, np.sort(mu))
    
    return phase


def two_term_henyey_greenstein(theta, g1, g2, f, theta_unit='deg', normalize=None):
    """
    Compute the two-term Henyey-Greenstein phase matrix

    Parameters
    ----------
    theta : 1-D ndarray
        The phase matrix angles. See parameter `theta_unit`
    g1 : float
        The first H-G term parameter g (forward part)
    g2 : float
        The second H-G term parameter g (backward part)
    f : float
        The fraction parameter between the two H-G terms (see notes)
    theta_unit : str, optional
        The unit for theta angles: 
        - 'deg' (default value)
        - 'rad'
    normalize : None | float, optional
        The normalization value of the integral ∫F(θ)dcosθ, where F(θ) is the phase matrix.  
        The scipy simpson function is used for the normalization.
    Return
    ------
    F_TTHG : 1-D ndarray
        The phase matrix.

    Notes
    -----
    The two term Henyey-Greenstein equation:
    
    - :math:`F_TTHG(θ) = f*F_HG1(θ) + (1-f)*F_HG2(θ)`
    
    - By default the integral ∫F_TTHG(θ)dcosθ = 1/2π.  
    The integral value can be different due to a very low dicretization of θ and/or a high g value.  
    The use of the parameter `normalize` can be useful to renormalize the phase function.

    References
    ----------
    Irvine, W. M. (1965). Multiple scattering by large particles (No. NASA-CR-64638).
    """

    F_HG1 = henyey_greenstein(theta=theta, g=g1, theta_unit=theta_unit, normalize=normalize)
    F_HG2 = henyey_greenstein(theta=theta, g=g2, theta_unit=theta_unit, normalize=normalize)
    phase = f*F_HG1 + (1-f)*F_HG2
    
    return phase


def calc_moments(phase, theta, m_max, method='lobatto', theta_unit='deg', normalize=False):
    """ 
    Calculate the phase matrix moments until m_max moment

    Parameters
    ----------
    phase : 1-D ndarray
        The phase matrix
    theta : 1-D ndarray
        The phase matrix angles. See parameter `theta_unit`
    m_max : int
        The maximum moment number to compute, i.e., compute m[0], ..., m[m_max]
    method : str, optional
        The method used to calculate the moments, choices:
        - 'lobatto' -> Default value, very effcient if "gauss kind" theta distribution
        - 'simpson' -> efficient if regular theta distribution (use scipy.integrate.simpson)
    theta_unit : str, optional
        The unit for theta angles: 
        - 'deg' (default value)
        - 'rad'
    normalize : bool, optional
        If normalize = True -> normalize such that first moment exactly = 1
    
    Returns
    -------
    m : 1-D ndarray
        The computed phase moment of size m_max + 1

    Notes
    -----
    
    - See Eq.A2 in ref[2] for moment computation using Lobatto quadrature in [0,π] 
    - For Lobatto quadrature abcissas and weights calculation see ref[2]

    References
    ----------

    - [1] Michels, H. (1963). Abscissas and weight coefficients for Lobatto quadrature. 
          Mathematics of Computation, 17(83), 237-244.
    
    - [2] Wiscombe, W. J. (1977). The delta-M method: Rapid yet accurate radiative flux calculations 
          for strongly asymmetric phase functions. Journal of Atmospheric Sciences, 34(9), 1408-1422.
    """

    methods_ok = ['lobatto', 'simpson']

    if method not in methods_ok:
        raise ValueError(f"Only available methods are: {methods_ok}")
    
    if theta_unit == 'deg':
            theta = (np.deg2rad(theta))
    elif ( theta_unit != 'rad' ):
        raise ValueError("The accepted values for parameter theta_unit are: 'deg' or 'rad'")

    if theta[0] < 0 or theta[-1] > np.pi:
        print("The range of theta must be [0, π] (rad unit) or [0,180] (deg unit)")

    # initialize moments
    chi = np.zeros(m_max + 1)

    if method == 'lobatto':
        nth = len(phase)
        if nth < 2:
            raise ValueError("The phase size must be >= 2 for Lobatto quadrature")
        
        xk, wk = quadrature_lobatto(abscissa_min=0., abscissa_max=math.pi, n=nth)

        cos_xk = np.cos(xk)
        sin_xk = np.sin(xk)
        pha = np.interp(xk, theta, phase)

        for l in range (m_max + 1):
            pl_cosxk = legendre_polynomials(l, cos_xk)
            chi[l] = 0.5 * np.sum(wk * pha * pl_cosxk * sin_xk)

    if method == 'simpson':
        mu = np.cos(theta)
        idmu = np.argsort(mu)
        for l in range(m_max+1):
            pl_mu = legendre_polynomials(l, mu[idmu])
            chi[l]= 0.5 * simpson(phase[idmu] * pl_mu, mu[idmu]) 

    # normalization
    if normalize: chi /= chi[0]

    return chi


def calc_hg_moments(g, m_max):
    """ 
    Compute exact Henyey-Greenstein phase moments

    - see Eq.8 in [1]

    Parameters
    ----------
    g : float
        The Henyey-Greenstein parameter g (measures the asymmetry of the phase matrix)
    m_max : int
        The maximum moment number to compute, i.e., compute m[0], ..., m[m_max]

    Returns
    -------
    m : 1-D ndarray
        The phase moment of size m_max + 1

    References
    ----------

    - [1] Kattawar, G. W. (1975). A three-parameter analytic phase function for multiple 
          scattering calculations. Journal of Quantitative Spectroscopy and Radiative Transfer, 
          15(9), 839-849.
    """

    return np.array([g**n for n in range(m_max+1)])


def calc_tthg_moments(g1, g2, f, m_max):
    """ 
    Compute exact Two-term Henyey-Greenstein phase moments

    - see Eq.11 in [1]

    Parameters
    ----------
    g1 : float
        The first H-G term parameter g (forward part)
    g2 : float
        The second H-G term parameter g (backward part)
    f : float
        The fraction parameter between the two H-G terms
    m_max : int
        The maximum moment number to compute, i.e., compute m[0], ..., m[m_max]
    
    Returns
    -------
    m : 1-D ndarray
        The phase moment of size m_max + 1

    References
    ----------

    - [1] Kattawar, G. W. (1975). A three-parameter analytic phase function for multiple 
          scattering calculations. Journal of Quantitative Spectroscopy and Radiative Transfer, 
          15(9), 839-849.
    """

    return np.array([( f*g1**n + (1-f)*g2**n ) for n in range(m_max+1)])