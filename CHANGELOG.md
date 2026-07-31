# PYTRUNC CHANGELOG


## Unreleased

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


## v1.0.1
Release date: 06-02-2026

* Modify python minimum version to 3.11

* VERSION constant automatically read from pyproject toml file

* Add missing information to pyproject toml file


## v1.0.0
Release date: 06-02-2026

* First release
