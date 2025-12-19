import math
import numpy as np
from scipy.integrate import simpson


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