# PYTRUNC CHANGELOG


## Unreleased

* Fix the `bessel_j1` large-x branch: it indexed the wrong mask and
  raised a ValueError for any input containing x > 35. The branch now
  uses Hankel's asymptotic expansion (Abramowitz & Stegun eq. 9.4.6),
  accurate to ~1e-10; the x <= 35 series is unchanged

* Fix `delta_m_phase_approx` ignoring its `method` parameter: the
  moments were always computed with the calc_moments default 'lobatto',
  whatever the caller asked. The moments now honor `method`, so the
  default ('trapezoid') results change slightly (the docstring example
  f value moves from 0.27248273 to 0.27250572)

* Replace the bare print of `calc_moments` on an out-of-range theta
  with a `warnings.warn` (UserWarning)

* Performance pass, bit-identical results (verified bitwise against a
  122-array reference of gt_phase_approx, delta_m_phase_approx,
  calc_moments and quadrature_lobatto outputs):
  - Advance the three-term Legendre recurrence incrementally across the
    moment loops of `calc_moments` and `delta_m_phase_approx` instead
    of restarting it from degree 0 at each degree (O(m_max²) → O(m_max)
    recurrence work): calc_moments is ~30x faster at m_max=256 and
    delta_m_phase_approx ~18x at m_max=128
  - Hoist the loop invariants out of the `gt_phase_approx` truncation
    angle search (phase×sinθ product, min/max reductions on monotone
    slices, argsorts replaced by reversal views, a reused scratch
    buffer for the candidate phase, sortedness detection so that
    integrate_lobatto skips its per-call argsort): the search is
    ~1.2-1.4x faster depending on the method
  - Fuse the first and second derivative Legendre recurrences of the
    Lobatto node solve into one loop (`_legendre_dp_d2p`)

* Remove dead code and redundant work: an always-true condition in the
  gt_phase_approx winner test, a duplicate allocation, defensive copies
  that protected nothing (`.copy()` on freshly allocated arrays and on
  multiplication results), a duplicated phase-size check, scalar
  conditions expressed as np.where, and a wrong error message in
  bessel_jn ("x must be >= 1" for a x < 0 check)

* Deduplicate repeated blocks: the theta_unit validation (4 copies →
  `_theta_to_rad`), the Dirac normalization of the two truncation
  methods (→ `_dirac_delta_part`, with its wrong "sin(pi) = 0" comment
  corrected to sin(0) = 0), the Henyey-Greenstein kernel of the
  two-term variant (→ `_hg_from_mu`, converting theta and sorting mu
  once), `calc_tthg_moments` reusing `calc_hg_moments`, and the three
  copies of the truncation candidate evaluation in gt_phase_approx
  (forced th_f mode, search seed and search loop → one local closure).
  An unknown `method` now raises a ValueError instead of a KeyError

* Rename the `constant` module to `constants`, for consistency with the
  geoclide package. The imports become `from pytrunc.constants import
  VERSION` and `from pytrunc.constants import DIR_ROOT`

* Use lowercase names for the local mathematical variables, following the
  PEP 8 naming conventions: `P0`, `P1`, `Pn`, `Pnm1`, `Pnp1` and their
  derivative variants in the utils module, `Pf` and `Pf_tmp` in the
  truncation module, and `F_HG1` and `F_HG2` in the phase module. The
  mathematical notation of the docstrings is unchanged, and the ruff
  configuration now selects the pep8-naming rules

* Reflow the docstrings and comments of every module, filling the
  72-column budget instead of only enforcing it, add docstrings to the
  private integrator wrappers of the truncation module, and shorten the
  Ocean Optics Web Book references so that they fit in the limit

* Commit the pixi lock file, previously ignored: the tests and docs
  workflows install the environment from it, and failed at the pixi setup
  step without it

* Add the pytrunc logo, in a light and a dark variant, at the top of the
  README and in the documentation sidebar, both following the light/dark
  theme

* Add a documentation landing page (docs/introduction.md, included in
  index.rst) presenting the features, a quickstart and the contents of
  the documentation

* Improve the README: features, installation, quickstart, examples,
  documentation and license sections, and badges for the tests, the
  documentation, the license, pixi and ruff

* Add a GitHub workflow running ruff, and lint the documentation
  configuration and notebook sources accordingly

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
