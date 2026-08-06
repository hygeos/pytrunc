"""
Scattering phase function truncation methods.

This module provides the approximation of a scattering phase function by
truncation of its forward peak, using either the delta-m method of
Wiscombe (1977) or the geometrical truncation (GT) method of Iwabuchi
and Suzuki (2009), and returns the truncated phase function together
with its truncation factor.
"""

import math
from collections.abc import Callable
from datetime import datetime

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from scipy.integrate import simpson, trapezoid

from pytrunc.constants import VERSION
from pytrunc.phase import calc_moments
from pytrunc.utils import integrate_lobatto, quadrature_lobatto


# scipy >= 1.14 made the x argument of simpson keyword-only, so the
# dispatch below cannot call the scipy integrators positionally
def _simpson(y: NDArray[np.float64], x: NDArray[np.float64]) -> float:
    """
    Integrate y over x with the scipy simpson function

    :meta private:
    """

    return float(simpson(y, x=x))


def _trapezoid(y: NDArray[np.float64], x: NDArray[np.float64]) -> float:
    """
    Integrate y over x with the scipy trapezoid function

    :meta private:
    """

    return float(trapezoid(y, x=x))


INTEGRATORS: dict[str, Callable[..., float]] = {
    "simpson": _simpson,
    "trapezoid": _trapezoid,
    "lobatto": integrate_lobatto,
}


def delta_m_phase_approx(
    phase: NDArray[np.float64],
    theta: NDArray[np.float64],
    m_max: int,
    theta_unit: str = "deg",
    phase_moments: NDArray[np.float64] | None = None,
    method: str = "trapezoid",
    ds_output: bool = True,
) -> xr.Dataset | tuple[NDArray[np.float64], float, NDArray[np.float64]]:
    """
    Calculate the approximation of the exact phase matrix using the
    delta-m method

    Parameters
    ----------
    phase : ndarray
        The exact phase matrix, it must be 1-D
    theta : ndarray
        The phase matrix angles, it must be 1-D. See the theta_unit
        parameter for the unit
    m_max : int
        The maximum term number
    theta_unit : str, optional
        The unit of the theta angles. Default is 'deg', other choice is
        'rad'
    phase_moments : ndarray or None, optional
        The moments of the phase matrix, it must be 1-D. The size of
        phase_moments must be >= m_max+1. If this parameter is not None,
        circumvent the calculation of the phase matrix moments. This
        parameter can be useful in case we have the exact moment values
        like for the H-G phase function. Default is None
    method : str, optional
        The method parameter of the calc_moments function, and also the
        integral method for the dirac normalization. Default is
        'trapezoid'
    ds_output : bool, optional
        If True the output is a dataset, else return a tuple. Default is
        True

    Returns
    -------
    Dataset or tuple
        Xarray dataset containing the truncation information if
        ds_output is True, else a tuple.

        Key variables included:

        - **phase_approx**: The approximation of the exact phase matrix
        - **f**: The truncation factor
        - **phase_tr**: The truncated phase matrix
        - **chi**: The moments of the exact phase matrix
        - **chi_star**: The moments of the truncated phase matrix

        Form of the tuple:

        * phase_approx : ndarray
            -> The approximation of the exact phase matrix, it is 1-D
        * f : float
            -> The truncation factor
        * phase_star : ndarray
            -> The truncated scattering phase matrix, it is 1-D

    References
    ----------
    Wiscombe, W. J. (1977). The delta-M method: Rapid yet accurate
    radiative flux calculations for strongly asymmetric phase functions.
    Journal of Atmospheric Sciences, 34(9), 1408-1422.

    Examples
    --------
    >>> import numpy as np
    >>> from pytrunc.phase import henyey_greenstein
    >>> from pytrunc.truncation import delta_m_phase_approx
    >>> theta = np.linspace(0.0, 180.0, 1801)
    >>> phase = henyey_greenstein(theta, g=0.85, normalize=2)
    >>> ds = delta_m_phase_approx(phase, theta, m_max=8)
    >>> ds['f'].values
    array(0.27250572)
    """

    if theta_unit == "deg":
        theta = np.deg2rad(theta)
    elif theta_unit != "rad":
        raise ValueError(
            "The accepted values for parameter theta_unit are: 'deg' or 'rad'"
        )

    if phase_moments is not None:
        if len(phase_moments) <= m_max:
            raise ValueError(
                f"The number of moments must be >= {m_max + 1}"
                + f", but only {len(phase_moments)} given"
            )
        chi = phase_moments
    else:
        chi = calc_moments(
            phase,
            theta,
            m_max=m_max,
            theta_unit="rad",
            method=method,
            normalize=True,
        )

    integrate_m = INTEGRATORS[method]

    f = chi[m_max]

    # here m_max = 2M (wiscombe 77)
    # chi_star[n] = (chi[n] - f)/(1 - f) for n = 0 to n=m_max-1
    chi_star = (chi[:m_max] - f) / (1 - f)

    cos_th = np.cos(theta)
    phase_star = np.zeros_like(theta)

    # phase_star = (1-f) * Σ [ (2n+1) * chi[n]* * pn(cosθ) ], with the
    # three-term Legendre recurrence advanced incrementally across the
    # loop instead of being recomputed from degree 0 at each order
    pnm1 = np.ones_like(cos_th)
    pn = cos_th
    for n in range(m_max):
        if n == 0:
            pn_costh = pnm1
        elif n == 1:
            pn_costh = pn
        else:
            k = n - 1
            pn_costh = (1.0 / (k + 1)) * (
                (2 * k + 1) * cos_th * pn - k * pnm1
            )
            pnm1, pn = pn, pn_costh
        phase_star += (2 * n + 1) * chi_star[n] * pn_costh

    phase_approx = phase_star * (1 - f)
    if f > 0:
        idmu = np.argsort(cos_th)
        delta_part = np.zeros_like(theta)
        delta_part[0] = 1.0
        if method == "lobatto":
            delta_part[1] = 1.0  # because sin(pi) = 0
            delta_part = delta_part / integrate_m(
                delta_part * np.sin(theta), theta
            )  # normalize dirac to 1
        else:
            delta_part[0] = delta_part[0] / integrate_m(
                delta_part[idmu], cos_th[idmu]
            )  # normalize dirac to 1
        delta_part = (2 * f) * delta_part
        phase_approx += delta_part

    if ds_output:
        ds = xr.Dataset(
            coords={
                "theta": np.rad2deg(theta),
                "exp_order": np.arange(m_max + 1),
            }
        )
        ds.coords["theta"].attrs.update(
            {"units": "degrees", "description": "scattering angle"}
        )
        ds["phase_approx"] = xr.DataArray(phase_approx, dims=["theta"])
        ds["phase_approx"].attrs.update(
            {
                "units": "none",
                "description": "the approximation of the exact phase matrix",
            }
        )
        ds["f"] = xr.DataArray(f)
        ds["f"].attrs.update(
            {"units": "none", "description": "the truncation factor"}
        )
        ds["phase_tr"] = xr.DataArray(phase_star, dims=["theta"])
        ds["phase_tr"].attrs.update(
            {"units": "none", "description": "the truncated phase matrix"}
        )
        ds["chi"] = xr.DataArray(chi[0 : m_max + 1], dims=["exp_order"])
        ds["chi"].attrs.update(
            {
                "units": "none",
                "description": "the moments of the exact phase matrix",
            }
        )
        ds["chi_star"] = xr.DataArray(
            np.concatenate((chi_star, np.array([0.0]))), dims=["exp_order"]
        )
        ds["chi_star"].attrs.update(
            {
                "units": "none",
                "description": "the moments of the truncated phase matrix",
            }
        )
        ds.attrs = {"truncation method": "DM"}
        date = datetime.now().strftime("%Y-%m-%d")
        ds.attrs.update({"date": date})
        ds.attrs.update({"m_max": m_max})
        ds.attrs.update({"integration method": method})
        ds.attrs.update({"pytrunc_version": VERSION})
        return ds
    else:
        return phase_approx, f, phase_star


def gt_phase_approx(
    phase: NDArray[np.float64],
    theta: NDArray[np.float64],
    trunc_frac: float,
    theta_unit: str = "deg",
    method: str = "trapezoid",
    phase_moments_1: float | None = None,
    th_tol: float | None = None,
    th_f: float | None = None,
    lobatto_optimization: bool = True,
    ds_output: bool = True,
) -> xr.Dataset | tuple[NDArray[np.float64], float, NDArray[np.float64]]:
    """
    Compute the approximation of the exact phase matrix using the
    Iwabuchi GT method

    Parameters
    ----------
    phase : ndarray
        The exact phase matrix, it must be 1-D
    theta : ndarray
        The phase matrix angles, it must be 1-D. See the theta_unit
        parameter for the unit
    trunc_frac : float
        The truncation fraction
    theta_unit : str, optional
        The unit of the theta angles. Default is 'deg', other choice is
        'rad'
    method : str, optional
        The method parameter of the calc_moments function, and also the
        integral method for the dirac normalization. Default is
        'trapezoid'
    phase_moments_1 : float or None, optional
        The value of the first moment of the phase matrix. Default is
        None, meaning it is computed with the calc_moments function
    th_tol : float or None, optional
        While finding matching moments for Pf we look between 0 and
        th_tol. The unit depends on the theta_unit parameter. Default is
        None, meaning th_tol is equal to pi/2
    th_f : float or None, optional
        Impose the truncation angle. The unit depends on the theta_unit
        parameter. Default is None, meaning the truncation angle is
        searched
    lobatto_optimization : bool, optional
        Whether to use lobatto optimization for integration (reuse the
        full-grid Lobatto quadrature, affinely rescaled to the
        truncation sub-interval, instead of solving for a new quadrature
        at every candidate angle during the truncation angle search).
        Default is True
    ds_output : bool, optional
        If True the output is a dataset, else return a tuple. Default is
        True

    Returns
    -------
    Dataset or tuple
        Xarray dataset containing the truncation information if
        ds_output is True, else a tuple.

        Key variables included:

        - **phase_approx**: The approximation of the exact phase matrix
        - **f**: The truncation factor
        - **phase_tr**: The truncated phase matrix
        - **chi_star_ideal**: The truncated phase matrix moments if
          moment conservation (ideal case)
        - **chi_star**: The actual truncated phase matrix moments
        - **theta_f**: The truncation angle
        - **th_f**: The th_f parameter value (to force the truncation
          angle)
        - **th_tol**: The th_tol parameter value

        Form of the tuple:

        * phase_approx : ndarray
            -> The approximation of the exact phase matrix, it is 1-D
        * f : float
            -> The truncation factor
        * phase_star : ndarray
            -> The truncated scattering phase matrix, it is 1-D

    References
    ----------
    Iwabuchi, H., & Suzuki, T. (2009). Fast and accurate radiance
    calculations using truncation approximation for anisotropic
    scattering phase functions. Journal of Quantitative Spectroscopy and
    Radiative Transfer, 110(17), 1926-1939.

    Examples
    --------
    >>> import numpy as np
    >>> from pytrunc.phase import henyey_greenstein
    >>> from pytrunc.truncation import gt_phase_approx
    >>> theta = np.linspace(0.0, 180.0, 1801)
    >>> phase = henyey_greenstein(theta, g=0.85, normalize=2)
    >>> ds = gt_phase_approx(phase, theta, trunc_frac=0.2)
    >>> ds['theta_f'].values
    array(16.8)
    """
    if theta_unit == "deg":
        theta = np.deg2rad(theta)
        if th_tol is not None:
            th_tol = np.deg2rad(th_tol)
        if th_f is not None:
            th_f = np.deg2rad(th_f)
    elif theta_unit != "rad":
        raise ValueError(
            "The accepted values for parameter theta_unit are: 'deg' or 'rad'"
        )

    th_tol_bis = th_tol
    th_f_bis = th_f
    if th_tol is None:
        th_tol = 0.5 * math.pi
    mu = np.cos(theta)

    if method == "lobatto":
        sin_th = np.sin(theta)
        xk, wk = quadrature_lobatto(
            abscissa_min=theta[0], abscissa_max=theta[-1], n=len(theta)
        )
        # Legendre polynomials of degree 0 and 1: ones and mu itself
        lp_costh = np.zeros((2, len(theta)))
        lp_costh[0] = 1.0
        lp_costh[1] = mu
    else:
        idmu = np.argsort(mu)

    if phase_moments_1 is not None:
        chi_1 = phase_moments_1
    else:
        if method == "lobatto":
            chi_1 = calc_moments(
                phase,
                theta,
                m_max=1,
                theta_unit="rad",
                method=method,
                normalize=True,
                xk=xk,
                wk=wk,
                pl_costh=lp_costh,
            )[1]
        else:
            chi_1 = calc_moments(
                phase,
                theta,
                m_max=1,
                theta_unit="rad",
                method=method,
                normalize=True,
            )[1]

    integrate_m = INTEGRATORS[method]

    f = trunc_frac
    chi_star_1 = (chi_1 - f) / (1 - f)

    delta_part = np.zeros_like(mu)
    delta_part[0] = 1.0
    if method == "lobatto":
        delta_part[1] = 1.0  # because sin(pi) = 0
        delta_part = delta_part / integrate_m(
            delta_part * sin_th, theta, xk=xk, wk=wk
        )  # normalize dirac to 1

    else:
        delta_part[0] = delta_part[0] / integrate_m(
            delta_part[idmu], mu[idmu]
        )  # normalize dirac to 1
    delta_part = (2 * f) * delta_part

    if th_f is not None:
        id_f = np.argmin(np.abs(theta - th_f))

        mu1 = mu[0 : id_f + 1]
        if method == "lobatto":
            th2 = theta[id_f:]

            pf_tmp = (
                2
                - (1.0 / (1 - f))
                * integrate_m(
                    phase[id_f:] * sin_th[id_f:],
                    th2,
                    lp=len(th2),
                    assume_sorted=True,
                )
            ) / ((1.0 / (1 - f)) * (np.max(mu1) - np.min(mu1)))
        else:
            mu2 = mu[id_f:]
            idmu2 = np.argsort(mu2)
            pf_tmp = (
                2
                - (1.0 / (1 - f))
                * integrate_m(phase[id_f:][idmu2], mu2[idmu2])
            ) / (
                (1.0 / (1 - f)) * (np.max(mu1) - np.min(mu1))
            )  # integrate_m(np.ones_like(mu1), mu1[idmu1]))

        pha_star = np.zeros_like(phase, dtype=np.float64)
        pha_star[id_f:] = phase[id_f:]
        pha_star[0:id_f] = pf_tmp
        pha_star *= 1.0 / (1 - f)

        if method == "lobatto":
            pha_star = (2 * pha_star) / integrate_m(
                pha_star * sin_th, theta, xk=xk, wk=wk, assume_sorted=True
            )
            chi_star_1_approx = calc_moments(
                pha_star,
                theta,
                m_max=1,
                theta_unit="rad",
                method=method,
                normalize=True,
                xk=xk,
                wk=wk,
                pl_costh=lp_costh,
            )[1]
        else:
            pha_star = (2 * pha_star) / integrate_m(pha_star[idmu], mu[idmu])
            chi_star_1_approx = calc_moments(
                pha_star,
                theta,
                m_max=1,
                theta_unit="rad",
                method=method,
                normalize=True,
            )[1]

        pha_approx = pha_star * (1 - f)
        pha_approx += delta_part

    else:
        # Find th_f and PF
        pha_star = np.zeros_like(phase, dtype=np.float64)
        mu1 = mu[0:2]
        if method == "lobatto":
            # th1 = theta[0:2]
            th2 = theta[1:]
            pf = (
                2 - (1.0 / (1 - f)) * integrate_m(phase[1:] * sin_th[1:], th2)
            ) / (
                (1.0 / (1 - f)) * (np.max(mu1) - np.min(mu1))
            )  # integrate_m(sin_th[0:2], th1))
        else:
            # idmu1 = np.argsort(mu1)
            mu2 = mu[1:]
            idmu2 = np.argsort(mu2)
            pf = (
                2 - (1.0 / (1 - f)) * integrate_m(phase[1:][idmu2], mu2[idmu2])
            ) / (
                (1.0 / (1 - f)) * (np.max(mu1) - np.min(mu1))
            )  # integrate_m(np.ones_like(mu1), mu1[idmu1]))
        pha_star[1:] = phase[1:]
        pha_star[0:1] = pf
        pha_star *= 1.0 / (1 - f)

        if method == "lobatto":
            pha_star = (2 * pha_star) / integrate_m(
                pha_star * sin_th, theta, xk=xk, wk=wk
            )
        else:
            pha_star = (2 * pha_star) / integrate_m(pha_star[idmu], mu[idmu])

        chi_star_1_approx = calc_moments(
            pha_star,
            theta,
            m_max=1,
            theta_unit="rad",
            method=method,
            normalize=True,
        )[1]
        err1 = abs(chi_star_1 - chi_star_1_approx)
        id_approx = 1

        xk_min = xk_span = 0.0
        if method == "lobatto":
            xk_min = float(np.min(xk))
            xk_span = float(np.max(xk)) - xk_min

        # loop invariants, hoisted out of the search loop
        inv_1mf = 1.0 / (1 - f)
        # for a sorted theta in [0, π], mu = cos(theta) is strictly
        # decreasing: the per-iteration min/max reductions reduce to
        # direct indexing and the argsorts of the mu slices to simple
        # reversals (views, no copies)
        sorted_th = bool(np.all(np.diff(theta) >= 0))
        descending = bool(np.all(np.diff(mu) < 0))
        if method == "lobatto":
            phase_sin = phase * sin_th
        else:
            mu_sorted = mu[idmu]
        # scratch buffer of the candidate phase: the [id:] tail always
        # equals phase/(1 - f), only the [0:id] plateau changes across
        # the iterations (and each write covers the previous one)
        pha_star_scratch = phase * inv_1mf

        for id in range(1, len(phase) - 2):
            if theta[id] >= th_tol:
                break

            # Find pf:
            # normalization condition between 0 and π ->
            # ∫ P*(θ) sin(θ) dθ = 2
            if descending:
                dmu1 = mu[0] - mu[id]
            else:
                mu1 = mu[0 : id + 1]
                dmu1 = np.max(mu1) - np.min(mu1)
            if method == "lobatto":
                th2 = theta[id:]

                # rescale of xk and wk in the tmp interval
                if lobatto_optimization:
                    if sorted_th:
                        abscissa_min = theta[id]
                        abscissa_max = theta[-1]
                    else:
                        abscissa_min = np.min(th2)
                        abscissa_max = np.max(th2)
                    alpha = (abscissa_max - abscissa_min) / xk_span
                    xk_ = abscissa_min + (xk - xk_min) * alpha
                    wk_ = wk * alpha

                    pf_tmp = (
                        2
                        - inv_1mf
                        * integrate_m(
                            phase_sin[id:],
                            th2,
                            xk=xk_,
                            wk=wk_,
                            assume_sorted=True,
                        )
                    ) / (
                        inv_1mf * dmu1
                    )  # integrate_m(sin_th[0:id+1], th1))
                else:
                    pf_tmp = (
                        2
                        - inv_1mf
                        * integrate_m(
                            phase_sin[id:],
                            th2,
                            lp=len(th2),
                            assume_sorted=True,
                        )
                    ) / (
                        inv_1mf * dmu1
                    )  # integrate_m(sin_th[0:id+1], th1))
            else:
                if descending:
                    phase2_s = phase[id:][::-1]
                    mu2_s = mu[id:][::-1]
                else:
                    mu2 = mu[id:]
                    idmu2 = np.argsort(mu2)
                    phase2_s = phase[id:][idmu2]
                    mu2_s = mu2[idmu2]
                pf_tmp = (2 - inv_1mf * integrate_m(phase2_s, mu2_s)) / (
                    inv_1mf * dmu1
                )  # integrate_m(np.ones_like(mu1), mu1[idmu1]))

            if np.isnan(pf_tmp) or np.isinf(pf_tmp):
                continue

            pha_star_tmp = pha_star_scratch
            pha_star_tmp[0:id] = pf_tmp * inv_1mf

            if method == "lobatto":
                pha_star_tmp = (2 * pha_star_tmp) / integrate_m(
                    pha_star_tmp * sin_th,
                    theta,
                    xk=xk,
                    wk=wk,
                    assume_sorted=True,
                )
                chi_star_1_approx_tmp = calc_moments(
                    pha_star_tmp,
                    theta,
                    m_max=1,
                    theta_unit="rad",
                    method=method,
                    normalize=True,
                    xk=xk,
                    wk=wk,
                    pl_costh=lp_costh,
                )[1]
            else:
                pha_star_sorted = (
                    pha_star_tmp[::-1]
                    if descending
                    else pha_star_tmp[idmu]
                )
                pha_star_tmp = (2 * pha_star_tmp) / integrate_m(
                    pha_star_sorted, mu_sorted
                )
                chi_star_1_approx_tmp = calc_moments(
                    pha_star_tmp,
                    theta,
                    m_max=1,
                    theta_unit="rad",
                    method=method,
                    normalize=True,
                )[1]

            err2 = abs(chi_star_1 - chi_star_1_approx_tmp)

            # theta[id] < th_tol always holds here (the loop breaks
            # first), so the winner test is on the error alone
            if err2 < err1:
                id_approx = id
                pha_star = pha_star_tmp
                chi_star_1_approx = chi_star_1_approx_tmp
                err1 = err2

        pha_approx = pha_star * (1 - f)
        pha_approx += delta_part

    if ds_output:
        ds = xr.Dataset(
            coords={"theta": np.rad2deg(theta), "exp_order": np.arange(2)}
        )
        ds.coords["theta"].attrs.update(
            {"units": "degrees", "description": "scattering angle"}
        )
        ds["phase_approx"] = xr.DataArray(pha_approx, dims=["theta"])
        ds["phase_approx"].attrs.update(
            {
                "units": "none",
                "description": "the approximation of the exact phase matrix",
            }
        )
        ds["f"] = xr.DataArray(f)
        ds["f"].attrs.update(
            {"units": "none", "description": "the truncation factor"}
        )
        ds["phase_tr"] = xr.DataArray(pha_star, dims=["theta"])
        ds["phase_tr"].attrs.update(
            {"units": "none", "description": "the truncated phase matrix"}
        )
        ds["chi_star_ideal"] = xr.DataArray(np.array([1.0, chi_star_1]))
        ds["chi_star_ideal"].attrs.update(
            {
                "units": "none",
                "description": "the truncated phase matrix moments "
                "if moment conservation (ideal case)",
            }
        )
        ds["chi_star"] = xr.DataArray(np.array([1.0, chi_star_1_approx]))
        ds["chi_star"].attrs.update(
            {
                "units": "none",
                "description": "the actual truncated phase matrix moments",
            }
        )
        if th_f_bis is not None:
            ds["theta_f"] = xr.DataArray(np.rad2deg(th_f_bis))
            ds["th_f"] = xr.DataArray(np.rad2deg(th_f_bis))
        else:
            ds["theta_f"] = xr.DataArray(np.rad2deg(theta[id_approx]))
            ds["th_f"] = xr.DataArray(None)
        ds["theta_f"].attrs.update(
            {"units": "degrees", "description": "the truncation angle"}
        )
        ds["th_f"].attrs.update(
            {
                "units": "degrees",
                "description": "the th_f parameter value "
                "(to force truncation angle)",
            }
        )
        if th_tol_bis is not None:
            ds["th_tol"] = xr.DataArray(np.rad2deg(th_tol))
        else:
            ds["th_tol"] = xr.DataArray(None)
        ds["th_tol"].attrs.update(
            {"units": "degrees", "description": "the th_tol parameter value"}
        )
        ds.attrs = {"truncation method": "GT"}
        date = datetime.now().strftime("%Y-%m-%d")
        ds.attrs.update({"date": date})
        ds.attrs.update({"phase_moments_1": chi_1})
        ds.attrs.update({"integration method": method})
        if method == "lobatto":
            ds.attrs.update({"lobatto_optimization": lobatto_optimization})
        ds.attrs.update({"pytrunc_version": VERSION})
        return ds
    else:
        return pha_approx, f, pha_star
