"""
Pytrunc
=======

A python package for the truncation of scattering phase functions
in radiative transfer applications.

Provides
  1. Analytic scattering phase functions: Henyey-Greenstein,
     two-term Henyey-Greenstein and Fournier-Forand
  2. Phase function Legendre moments, computed numerically for an
     arbitrary phase function or analytically for the
     Henyey-Greenstein family
  3. Truncation methods: the delta-m method and the Iwabuchi GT
     method

How to use the documentation
----------------------------
Documentation is available in two forms: docstrings provided with
the code, and a standalone reference guide, available from `the
pytrunc homepage <https://hygeos.github.io/pytrunc/>`_.

Code snippets are indicated by three greater-than signs::

    >>> import numpy as np
    >>> from pytrunc.phase import henyey_greenstein
    >>> theta = np.linspace(0.0, 180.0, 1801)
    >>> phase = henyey_greenstein(theta, g=0.85, normalize=2)

Use the built-in ``help`` function to view a function's
docstring::

    >>> help(henyey_greenstein)
"""
