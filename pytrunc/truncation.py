import numpy as np
from pytrunc.phase import calc_moments
from pytrunc.utils import legendre_polynomials
from scipy.signal import unit_impulse


def delta_m_phase_approx(phase, theta, m_max, theta_unit='deg', phase_moments=None):
    """
    Caculate the aproximation of the exact phase matrix using the delta-m method

    Parameters
    ----------
    phase : 1-D ndarray
        The exact phase matrix
    theta : 1-D ndarray
        The phase matrix angles
    m_max : int
        The maximum term number
    theta_unit : str, optional
        The unit for theta angles: 
        - 'deg' (default value)
        - 'rad'
    phase_moments : 1-D ndarray
        The moments of the phase matrix. the size of phase_moments must be >= m_max+1.  
        If this parameter is not None circumvent the calculation of phase matrix moments.  
        This parameter can be useful in case we have the exact moment values like for H-G 
        phase function

    Returns
    -------
    phase_approx : 1-D ndarray
        The approximation of the exact phase matrix
    f : float
        The truncation factor
    phase_star : 1-D ndarray
        The truncated scattering phase matrix
    chi_star : 1-D ndarray
        The delta-m chi_star coefficients for moment conservation between 0 and m_max-1

    References
    ----------
    
    - [1] Wiscombe, W. J. (1977). The delta-M method: Rapid yet accurate radiative flux calculations 
          for strongly asymmetric phase functions. Journal of Atmospheric Sciences, 34(9), 1408-1422.
    """

    if theta_unit == 'deg':
            theta = (np.deg2rad(theta))
    elif ( theta_unit != 'rad' ):
        raise ValueError("The accepted values for parameter theta_unit are: 'deg' or 'rad'")

    if phase_moments is not None:
        if len(phase_moments) <= m_max:
            raise ValueError(f"The number of moments must be >= {m_max+1}" + \
                             f", but only {len(phase_moments)} given")
        chi = phase_moments
    else:
        chi = calc_moments(phase, theta, m_max=m_max, theta_unit='rad', normalize=True)

    f = chi[m_max]

    # here m_max = 2M (wiscombe 77)
    # chi_star[n] = (chi[n] - f)/(1 - f) for n = 0 to n=m_max-1
    chi_star = np.zeros(m_max, dtype=np.float64)
    for n in range(m_max):
        chi_star[n] = (chi[n] - f) / (1 - f)

    cos_th = np.cos(theta)
    phase_star = np.zeros_like(theta)

    # phase_star = (1-f) * Σ [ (2n+1) * chi[n]* * pn(cosθ) ]
    for n in range(m_max):
        pn_costh = legendre_polynomials(n, cos_th)
        phase_star += (1 - f) * (2*n + 1) * chi_star[n] * pn_costh

    phase_approx = phase_star.copy()
    if f > 0:
        delta_part = unit_impulse(len(theta), 0)
        delta_part = (2*f) * delta_part
        phase_approx += delta_part
    
    return phase_approx, f, phase_star, chi_star