<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/hygeos/pytrunc/refs/heads/main/pytrunc/img/pytrunc-logo-horizontal-dark-bg.png">
  <img alt="pytrunc" src="https://raw.githubusercontent.com/hygeos/pytrunc/refs/heads/main/pytrunc/img/pytrunc-logo-horizontal-light-bg.png" width="450">
</picture>
</p>

------------------------------------------------

<p align="center">
<a href="https://pypi.python.org/pypi/pytrunc"><img alt="pypi" src="https://img.shields.io/pypi/v/pytrunc.svg"></a>
<a href="https://anaconda.org/conda-forge/pytrunc"><img alt="conda-forge" src="https://img.shields.io/conda/vn/conda-forge/pytrunc.svg"></a>
<a href="https://github.com/hygeos/pytrunc"><img alt="github" src="https://img.shields.io/github/v/tag/hygeos/pytrunc?label=github&amp;color=blue"></a>
<a href="https://pepy.tech/project/pytrunc"><img alt="downloads" src="https://static.pepy.tech/badge/pytrunc"></a>
</p>

<p align="center">
<a href="https://github.com/hygeos/pytrunc/actions/workflows/tests.yml"><img alt="tests" src="https://github.com/hygeos/pytrunc/actions/workflows/tests.yml/badge.svg?branch=main"></a>
<a href="https://hygeos.github.io/pytrunc/"><img alt="docs" src="https://github.com/hygeos/pytrunc/actions/workflows/docs_github_pages.yml/badge.svg?branch=main"></a>
<a href="https://github.com/hygeos/pytrunc/blob/main/LICENSE.txt"><img alt="license" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg"></a>
<a href="https://pixi.sh"><img alt="pixi" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json"></a>
<a href="https://github.com/astral-sh/ruff"><img alt="ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
</p>

A python package for the truncation of scattering phase functions in
radiative transfer applications

Mustapha Moulana  
[HYGEOS website](https://hygeos.com/en/)  
[Documentation](https://hygeos.github.io/pytrunc/)

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

# Examples
Truncation of a realistic water cloud phase function (Mie calculation at
500 nm, effective radius of 8 µm) with the delta-m method:

<details>
<summary>Show figure</summary>

```python
>>> ...
>>> from pytrunc.truncation import delta_m_phase_approx
>>> m_max = 20  # stream / term number
>>> # phase_exact -> P11 of a liquid cloud at 500nm, effective radius of 8 micrometers
>>> # theta -> the phase angles in degrees
>>> ds = delta_m_phase_approx(phase_exact, theta, m_max)  # return an xarray dataset
>>> ...
```

<img src="https://raw.githubusercontent.com/hygeos/pytrunc/refs/heads/main/pytrunc/img/truncated_phase_dm.png" width="800">

</details>

The same phase function truncated with the geometrical truncation (GT)
method:

<details>
<summary>Show figure</summary>

```python
>>> ...
>>> from pytrunc.truncation import gt_phase_approx
>>> from pytrunc.phase import calc_moments
>>> m_max = 20  # stream / term number
>>> # phase_exact -> P11 of a liquid cloud at 500nm, effective radius of 8 micrometers
>>> # theta -> the phase angles in degrees
>>> chi = calc_moments(phase_exact, theta, m_max=m_max, normalize=True)  # the phase moments
>>> f = chi[m_max]  # the truncation factor
>>> ds = gt_phase_approx(phase_exact, theta, f)  # return an xarray dataset
>>> ...
```

<img src="https://raw.githubusercontent.com/hygeos/pytrunc/refs/heads/main/pytrunc/img/truncated_phase_gt.png" width="800">

</details>

# Documentation
The complete documentation is available at
[hygeos.github.io/pytrunc](https://hygeos.github.io/pytrunc/). It
includes example notebooks (truncation of a realistic water cloud phase
function, Lobatto quadrature, and the validation against Iwabuchi and
Suzuki (2009)) and the full API reference. The docstrings are also
available from the built-in `help` function, e.g.
`help(delta_m_phase_approx)`.

# License
Pytrunc is licensed under the Apache License 2.0, see
[LICENSE.txt](https://github.com/hygeos/pytrunc/blob/main/LICENSE.txt).
