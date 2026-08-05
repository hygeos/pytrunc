# PYTRUNC CHANGELOG


## Unreleased

* Add the `fournier_forand` phase function to the phase module, following
  the same conventions as `henyey_greenstein` (theta_unit and normalize
  parameters), with a direct unit test suite (tests/test_phase.py)

* Docstring convention pass aligned with the geoclide package: 72-column
  docstring limit enforced by ruff (max-doc-length), numpydoc style with
  Examples sections on all public functions, unified References format,
  and module descriptions in every module including the test modules

* Modernize the license metadata: SPDX expression (license = "Apache-2.0")
  with license-files, rename LICENSE.TXT to LICENSE.txt, remove the
  deprecated license classifier and require setuptools >= 77


## v1.1.0
Release date: 31-07-2026

* Major speed-up of the `gt_phase_approx` function with method='lobatto'
  - The Lobatto quadrature abscissas and weights are now cached, avoiding
    the expensive node computation at each call
  - Faster interpolation in `integrate_lobatto`
  - The parameter lobatto_optimization is now True by default

* Fix scipy >= 1.14 compatibility (the x argument of the scipy simpson
  function became keyword-only)

* Fix an UnboundLocalError in `gt_phase_approx` when the truncation angle
  search does not improve on its starting point

* Add a pytest regression test suite (tests directory)

* Add a GitHub workflow for automatic publishing to PyPI on version tags

* Code modernization: type hints in all modules, ruff lint + format
  (line length 79), pyright clean, docstrings and comments reflowed to
  72 columns


## v1.0.1
Release date: 06-02-2026

* Modify python minimum version to 3.11

* VERSION constant automatically read from pyproject toml file

* Add missing information to pyproject toml file


## v1.0.0
Release date: 06-02-2026

* First release
