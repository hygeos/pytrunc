import numpy as np
from pytrunc.phase import calc_moments
from pytrunc.utils import legendre_polynomials, integrate_lobatto, quadrature_lobatto
from scipy.integrate import trapezoid, simpson
import math


def delta_m_phase_approx(phase, theta, m_max, theta_unit='deg', phase_moments=None,
                         method='trapezoid'):
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
    method : str, optional
        The method parameter of function calc_moments. Default use 'trapezoid'.
        Also the integral method for the dirac normalization.

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

    INTEGRATORS = {
    "simpson": simpson,
    "trapezoid": trapezoid,
    "lobatto": integrate_lobatto
    }
    integrate_m = INTEGRATORS[method]

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
        delta_part[0] = 1. 
        if method == "lobatto":
            delta_part[1] = 1. # because sin(pi) = 0
            delta_part = delta_part / integrate_m(delta_part*np.sin(theta), theta) # normalize dirac to 1
        else:
            delta_part[0] = delta_part[0] / integrate_m(delta_part[idmu], cos_th[idmu]) # normalize dirac to 1
        delta_part = (2*f) * delta_part
        phase_approx += delta_part
    
    return phase_approx, f, phase_star, chi_star


def gt_phase_approx(phase, theta, trunc_frac, theta_unit='deg', 
                    method='trapezoid', phase_moments_1=None, 
                    th_tol = None, th_f=None, lobatto_optimization=False):
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
        Also the integral method for the dirac normalization.
    phase_moments : None | 1-D ndarray, optional
        The moments of the phase matrix. the size of phase_moments must be >= m_max+1.  
        If this parameter is not None circumvent the calculation of phase matrix moments.  
        This parameter can be useful in case we have the exact moment values like for H-G 
        phase function
    th_tol : None | float, optional
        While finding matching moments for Pf we look between 0 and th_tol. 
        The unit is depending on the parameter theta_unit. Default th_tol = π/2
    th_f : None | float, optional
        Impose the truncation angle. The unit is depending on the parameter theta_unit.
    lobatto_optimization : bool, optional
        Whether to use lobatto optimization for integration. Default is False.

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
            theta = np.deg2rad(theta)
            if th_tol is not None: th_tol = np.deg2rad(th_tol)
            if th_f is not None: th_f = np.deg2rad(th_f)
    elif ( theta_unit != 'rad' ):
        raise ValueError("The accepted values for parameter theta_unit are: 'deg' or 'rad'")
    
    if th_tol is None: th_tol = 0.5*math.pi
    mu = np.cos(theta)
    idmu = np.argsort(mu)

    if phase_moments_1 is not None:
        chi_1 = phase_moments_1[1]
    else:
        if method == 'lobatto':
            sin_th = np.sin(theta)
            xk, wk = quadrature_lobatto(abscissa_min=theta[0], abscissa_max=theta[-1], n=len(theta))
            lp_costh = np.zeros((2, len(theta)))
            for l in range(2):
                lp_costh[l] = legendre_polynomials(l, np.cos(theta))
            chi_1 = calc_moments(phase, theta, m_max=1, theta_unit='rad', 
                                 method=method, normalize=True, xk=xk, wk=wk, pl_costh=lp_costh)[1]
        else:
            chi_1 = calc_moments(phase, theta, m_max=1, theta_unit='rad', method=method, normalize=True)[1]
    
    INTEGRATORS = {
    "simpson": simpson,
    "trapezoid": trapezoid,
    "lobatto": integrate_lobatto
    }
    integrate_m = INTEGRATORS[method]

    f = trunc_frac
    chi_star_1 = (chi_1 - f) / (1 - f)

    delta_part = np.zeros_like(mu)
    delta_part[0] = 1.
    if method == "lobatto":
        delta_part[1] = 1. # because sin(pi) = 0
        delta_part = delta_part / integrate_m(delta_part*sin_th, theta, xk=xk, wk=wk) # normalize dirac to 1
        
    else:
        delta_part[0] = delta_part[0] / integrate_m(delta_part[idmu], mu[idmu]) # normalize dirac to 1
    delta_part = (2*f) * delta_part

    if th_f is not None:
        pha_star = np.zeros_like(phase, dtype=np.float64)
        id_f = np.argmin(np.abs(theta - th_f))

        mu1 = mu[0:id_f+1]
        if method == "lobatto":
            sin_th = np.sin(theta)
            th2 = theta[id_f:]
 
            Pf_tmp = (2 - (1./(1-f))*integrate_m(phase[id_f:]*sin_th[id_f:], th2, lp=len(th2), assume_sorted=True) ) / \
                ((1./(1-f)) * (np.max(mu1) - np.min(mu1)))
        else:

            mu2 = mu[id_f:]
            idmu2 = np.argsort(mu2)
            Pf_tmp = (2 - (1./(1-f))*integrate_m(phase[id_f:][idmu2], mu2[idmu2]) ) / \
                    ((1./(1-f)) * (np.max(mu1) - np.min(mu1)))#integrate_m(np.ones_like(mu1), mu1[idmu1]))
        
        pha_star = np.zeros_like(phase, dtype=np.float64)
        pha_star[id_f:] = phase[id_f:]
        pha_star[0:id_f] = Pf_tmp
        pha_star *= 1./(1-f)

        if method == "lobatto":
            pha_star = (2 * pha_star) / integrate_m(pha_star*sin_th, theta, xk=xk, wk=wk, 
                                                            assume_sorted=True)
            chi_star_1_approx = calc_moments(pha_star, theta, m_max=1, theta_unit='rad', 
                                                 method=method, normalize=True, xk=xk, wk=wk, pl_costh=lp_costh)[1]
        else:
            pha_star = (2 * pha_star) / integrate_m(pha_star[idmu], mu[idmu])
            chi_star_1_approx = calc_moments(pha_star, theta, m_max=1, theta_unit='rad', 
                                                 method=method, normalize=True)[1]
            

        pha_approx = pha_star.copy() * (1-f)
        pha_approx += delta_part

    else:
        # Find th_f and PF
        pha_star = np.zeros_like(phase, dtype=np.float64)
        mu1 = mu[0:2]
        if method == "lobatto":
            #th1 = theta[0:2]
            th2 = theta[1:]
            Pf = (2 - (1./(1-f))*integrate_m(phase[1:]*sin_th[1:], th2) ) / \
                ((1./(1-f)) * (np.max(mu1) - np.min(mu1))) #integrate_m(sin_th[0:2], th1))
        else:
            #idmu1 = np.argsort(mu1)
            mu2 = mu[1:]
            idmu2 = np.argsort(mu2)
            Pf = (2 - (1./(1-f))*integrate_m(phase[1:][idmu2], mu2[idmu2]) ) / \
                    ((1./(1-f)) *(np.max(mu1) - np.min(mu1))) #integrate_m(np.ones_like(mu1), mu1[idmu1]))
        pha_star[1:] = phase[1:]
        pha_star[0:1] = Pf
        pha_star *= 1./(1-f)

        if method == "lobatto":
            pha_star = (2 * pha_star) / integrate_m(pha_star*sin_th, theta, xk=xk, wk=wk)
        else:
            pha_star = (2 * pha_star) / integrate_m(pha_star[idmu], mu[idmu])

        chi_star_1_approx = calc_moments(pha_star, theta, m_max=1, theta_unit='rad', method=method, normalize=True)[1]
        err1 = abs(chi_star_1 - chi_star_1_approx)
        #id_approx = 1

        for id in range (1, len(phase)-2):     
            if (theta[id] >= th_tol):
                break

            # Find Pf:
            # normalization condition between 0 and π ->  ∫ P*(θ) sin(θ) dθ = 2
            mu1 = mu[0:id+1]
            if method == "lobatto":
                sin_th = np.sin(theta)
                #th1 = theta[0:id+1]
                th2 = theta[id:]

                # rescale of xk and wk in the tmp interval
                if lobatto_optimization:
                    abscissa_min = np.min(th2)
                    abscissa_max = np.max(th2)
                    alpha = (abscissa_max - abscissa_min) / (np.max(xk) - np.min(xk))
                    xk_ = abscissa_min + (xk - np.min(xk)) * alpha 
                    wk_ = wk * alpha

                    Pf_tmp = (2 - (1./(1-f))*integrate_m(phase[id:]*sin_th[id:], th2, xk=xk_, wk=wk_, assume_sorted=True) ) / \
                        ((1./(1-f)) * (np.max(mu1) - np.min(mu1)))#integrate_m(sin_th[0:id+1], th1))
                else:
                    Pf_tmp = (2 - (1./(1-f))*integrate_m(phase[id:]*sin_th[id:], th2, lp=len(th2), assume_sorted=True) ) / \
                        ((1./(1-f)) * (np.max(mu1) - np.min(mu1)))#integrate_m(sin_th[0:id+1], th1))
            else:
                #idmu1 = np.argsort(mu1)
                mu2 = mu[id:]
                idmu2 = np.argsort(mu2)
                Pf_tmp = (2 - (1./(1-f))*integrate_m(phase[id:][idmu2], mu2[idmu2]) ) / \
                        ((1./(1-f)) * (np.max(mu1) - np.min(mu1)))#integrate_m(np.ones_like(mu1), mu1[idmu1]))
            
            if np.isnan(Pf_tmp) or np.isinf(Pf_tmp):
                continue
            
            pha_star_tmp = np.zeros_like(phase, dtype=np.float64)
            pha_star_tmp[id:] = phase[id:]
            pha_star_tmp[0:id] = Pf_tmp
            pha_star_tmp *= 1./(1-f)

            if method == "lobatto":
                pha_star_tmp = (2 * pha_star_tmp) / integrate_m(pha_star_tmp*sin_th, theta, xk=xk, wk=wk, 
                                                                assume_sorted=True)
                chi_star_1_approx_tmp = calc_moments(pha_star_tmp, theta, m_max=1, theta_unit='rad', 
                                                    method=method, normalize=True, xk=xk, wk=wk, pl_costh=lp_costh)[1]
            else:
                pha_star_tmp = (2 * pha_star_tmp) / integrate_m(pha_star_tmp[idmu], mu[idmu])
                chi_star_1_approx_tmp = calc_moments(pha_star_tmp, theta, m_max=1, theta_unit='rad', 
                                                    method=method, normalize=True)[1]
                
            err2 = abs(chi_star_1 - chi_star_1_approx_tmp)

            if (err2 < err1 and theta[id] < th_tol):
                id_approx = id
                pha_star = pha_star_tmp.copy()
                chi_star_1_approx = chi_star_1_approx_tmp
                err1 = err2

            pha_approx = pha_star.copy() * (1-f)
            pha_approx += delta_part

    return pha_approx, f, pha_star

