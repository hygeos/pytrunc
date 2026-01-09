import numpy as np
from pytrunc.phase import calc_moments
from pytrunc.utils import legendre_polynomials
from scipy.integrate import trapezoid


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
        phase_star += (2*n + 1) * chi_star[n] * pn_costh

    phase_approx = phase_star.copy() * (1 - f)
    if f > 0:
        idmu = np.argsort(cos_th)
        delta_part = np.zeros_like(theta)
        delta_part[0] = 1. / np.abs(cos_th[1] - cos_th[0])
        delta_part[0] = delta_part[0] / trapezoid(delta_part[idmu], cos_th[idmu]) # normalize dirac to 1
        delta_part = (2*f) * delta_part
        phase_approx += delta_part
    
    return phase_approx, f, phase_star, chi_star


def gt_phase_approx(phase, theta, trunc_frac, theta_unit='deg', 
                    method='trapezoid', phase_moments_1=None, th_f=None):
    """
    Compute the aproximation of the exact phase matrix using the Iwabuchi GT method

    Parameters
    ----------
    phase : 1-D ndarray
        The exact phase matrix
    theta : 1-D ndarray
        The phase matrix angles
    trunc_fract : float
        The truncature fraction
    theta_unit : str, optional
        The unit for theta angles: 
        - 'deg' (default value)
        - 'rad'
    method : str, optional
        The method parameter of function calc_moments. Default use 'trapezoid'.
    phase_moments : None | 1-D ndarray, optional
        The moments of the phase matrix. the size of phase_moments must be >= m_max+1.  
        If this parameter is not None circumvent the calculation of phase matrix moments.  
        This parameter can be useful in case we have the exact moment values like for H-G 
        phase function
    th_f : None | float, optional
        Still in dev...

    Returns
    -------
    phase_approx : 1-D ndarray
        The approximation of the exact phase matrix
    f : float
        The truncation factor
    phase_star : 1-D ndarray
        The truncated scattering phase matrix
    """
    if theta_unit == 'deg':
            theta = (np.deg2rad(theta))
    elif ( theta_unit != 'rad' ):
        raise ValueError("The accepted values for parameter theta_unit are: 'deg' or 'rad'")

    mu = np.cos(theta)
    idmu = np.argsort(mu)

    if phase_moments_1 is not None:
        chi_1 = phase_moments_1[1]
    else:
        chi_1 = calc_moments(phase, theta, m_max=1, theta_unit='rad', method=method, normalize=True)[1]
    
    f = trunc_frac
    chi_star_1 = (chi_1 - f) / (1 - f)

    # Find th_f and PF
    pha_star = np.zeros_like(phase, dtype=np.float64)
    mu1 = mu[0:2]
    idmu1 = np.argsort(mu1)
    mu2 = mu[1:]
    idmu2 = np.argsort(mu2)
    Pf = (2 - (1./(1-f))*trapezoid(phase[1:][idmu2], mu2[idmu2]) ) / \
            ((1./(1-f)) * trapezoid(np.ones_like(mu1), mu1[idmu1]))
    pha_star[1:] = phase[1:]
    pha_star[0:1] = Pf
    pha_star *= 1./(1-f)
    pha_star = (2 * pha_star) / trapezoid(pha_star[idmu], mu[idmu])
    chi_star_1_approx = calc_moments(pha_star, theta, m_max=1, theta_unit='rad', method=method, normalize=True)[1]
    err1 = abs(chi_star_1 - chi_star_1_approx)
    id_approx = 1
    for id in range (0, len(phase)-2):     
        th_f = theta[id]

        # Find Pf:
        # normalization condition between 0 and π ->  ∫ P*(θ) sin(θ) dθ = 2
        mu1 = mu[0:id+1]
        idmu1 = np.argsort(mu1)
        mu2 = mu[id:]
        idmu2 = np.argsort(mu2)
        Pf_tmp = (2 - (1./(1-f))*trapezoid(phase[id:][idmu2], mu2[idmu2]) ) / \
            ((1./(1-f)) * trapezoid(np.ones_like(mu1), mu1[idmu1]))
        
        pha_star_tmp = np.zeros_like(phase, dtype=np.float64)
        pha_star_tmp[id:] = phase[id:]
        pha_star_tmp[0:id] = Pf_tmp
        pha_star_tmp *= 1./(1-f)
        pha_star_tmp = (2 * pha_star_tmp) / trapezoid(pha_star_tmp[idmu], mu[idmu])

        chi_star_1_approx_tmp = calc_moments(pha_star_tmp, theta, m_max=1, theta_unit='rad', method=method, normalize=True)[1]
        err2 = abs(chi_star_1 - chi_star_1_approx_tmp)

        if (err2 < err1 and theta[id] < 0.5*np.pi):
            id_approx = id
            pha_star = pha_star_tmp
            chi_star_1_approx = chi_star_1_approx_tmp
        err1 = err2

        #print(theta[id], chi_star_1_approx_tmp)
        if (theta[id] >= 0.5*np.pi): break

        pha_approx = pha_star.copy() * (1-f)
        delta_part = np.zeros_like(mu)
        delta_part[0] = 1. / np.abs(mu[1] - mu[0])
        delta_part[0] = delta_part[0] / trapezoid(delta_part[idmu], mu[idmu]) # normalize dirac to 1
        delta_part = (2*f) * delta_part
        pha_approx += delta_part

    return pha_approx, f, pha_star

