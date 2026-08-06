A python package for the truncation of scattering phase functions in
radiative transfer applications

Mustapha Moulana  
[HYGEOS website](https://hygeos.com/en/)

-----------------------------------------

# Features
- Analytic scattering phase functions: Henyey-Greenstein, two-term
  Henyey-Greenstein and Fournier-Forand
- Phase function Legendre moments, computed numerically for an arbitrary
  phase function or analytically for the Henyey-Greenstein family
- Truncation of the forward peak with the delta-m method of Wiscombe
  (1977) and the geometrical truncation (GT) method of Iwabuchi and
  Suzuki (2009)
- Truncation results returned as xarray datasets, gathering the
  approximated and truncated phase functions, the truncation factor,
  the truncation angle and the exact and truncated moments
- Numerical utilities: Legendre polynomials and their derivatives,
  Bessel functions of the first kind, and a Lobatto quadrature with
  cached abscissas and weights, for accurate integration of strongly
  peaked phase functions on a limited number of angles

# Quickstart
Truncate a Henyey-Greenstein phase function with the delta-m method and
get the result as an xarray dataset:
```python
>>> import numpy as np
>>> from pytrunc.phase import henyey_greenstein
>>> from pytrunc.truncation import delta_m_phase_approx
>>> theta = np.linspace(0., 180., 1801)  # scattering angles in degrees
>>> phase = henyey_greenstein(theta, g=0.85, normalize=2)
>>> ds = delta_m_phase_approx(phase, theta, m_max=20)
>>> ds['f'].values
array(0.03874944)
```

The same phase function truncated with the GT method, using the last
moment as the truncation fraction:
```python
>>> from pytrunc.phase import calc_moments
>>> from pytrunc.truncation import gt_phase_approx
>>> chi = calc_moments(phase, theta, m_max=20, normalize=True)
>>> ds = gt_phase_approx(phase, theta, trunc_frac=chi[20])
>>> ds['f'].values, ds['theta_f'].values
(array(0.03874944), array(7.1))
```
More complete walkthroughs are available in the
[Notebooks](notebooks.rst) section.

# Documentation contents

## Notebooks
Two notebooks, from the first steps to the validation against the
reference publication:

- [Examples](01_examples.ipynb) — truncate a realistic water cloud phase
  function (Mie calculation at 500 nm, effective radius of 8 µm) with
  both the GT and the delta-m methods, and use the Lobatto quadrature to
  keep an accurate phase function and accurate moments with fewer angles
- [Validation Iwabuchi](02_validation_iwabuchi.ipynb) — reproduce the
  figures of Iwabuchi and Suzuki (2009): exact and approximated phase
  functions, a zoom on the forward peak, and the exact and approximated
  phase moments with their ratio

## Pytrunc package
The [API reference](pytrunc.rst), organized by module:

- [truncation](pytrunc.truncation.rst) — the `delta_m_phase_approx` and
  `gt_phase_approx` functions, the two truncation methods
- [phase](pytrunc.phase.rst) — the analytic phase functions
  (`henyey_greenstein`, `two_term_henyey_greenstein`,
  `fournier_forand`) and the moment calculations (`calc_moments`,
  `calc_hg_moments`, `calc_tthg_moments`)
- [utils](pytrunc.utils.rst) — the Legendre polynomials, the Bessel
  functions and the Lobatto quadrature and integration

## Releases
The [Releases](changelog_link.rst) section lists the versions of pytrunc
and the changes they introduced.

# Installation
The installation can be performed using one of the following commands:
```shell
$ conda install -c conda-forge pytrunc
```
```shell
$ pip install pytrunc
```
```shell
$ pip install git+https://github.com/hygeos/pytrunc.git
```
