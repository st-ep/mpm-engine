# Changelog

## Unreleased

### Added

- `ident/weakform/elastic_grid.py`: grid-consistent (Bubnov-Galerkin) weak-form
  assembly on the 3D quadratic B-spline grid basis, for laws linear in theta.
  The fixed-corotated elastic pair (mu, lambda) is the first client and names the
  module; `assemble_columns_timeweak` is the law-independent engine, taking the
  volume-weighted Cauchy stress columns of any linear law plus an optional known
  stress part (a pressure closure, for instance).
  - Instantaneous route: the exact per-node MPM momentum balance with a
    finite-difference acceleration load.
  - Time-weak route: the same rows integrated in time against a temporal weight
    that vanishes at both ends of the window, which removes the acceleration
    from the data entirely. On the elastic bounce this is the difference between
    a percent-level and a sub-percent recovery.
  - Smooth spatial windows over the kept nodes and exact row aggregation over
    node and frame blocks, both linear combinations of exact node rows.
  - Node gating on collider clearance, support mass, and per-particle validity.
- `tests/test_elastic_grid.py`: 16 tests, including a manufactured-acceleration
  patch test that recovers (mu, lambda) to solver precision, and order-one
  negative controls for the factor of two in the mu column, the load sign, and
  the current-versus-reference volume pairing.
- `ident/io/schema.py`: `KNOWN_LAWS`, extending the dump law tag with
  `corotated`, `vonmises`, `drucker_prager`, and `eos_fluid` so the
  NCLaw-matched comparison can write schema-valid dumps for its three
  non-granular materials. Gate code branches only on the two granular names, so
  no existing behaviour changes.

### Changed

- `solve_elastic_grid` works for any number of columns: it always returns
  `theta` and `theta_sd`, and adds the elastic names (mu, lam, E, nu) only when
  there are exactly two. A one-column law, the sand friction coefficient or the
  water bulk modulus, no longer trips over the elastic pair's names.

### Measured

- Elastic recovery on the TrackEUCLID drop dumps, time-weak route versus the
  radial-window reference: sphere E error 0.10 percent (radial 0.35 percent),
  mu error 0.21 percent (radial 3.89 percent); cube E error 0.07 percent
  (radial 13.4 percent), mu error 0.42 percent (radial 10.9 percent). The radial
  window's small sphere E error is a cancellation between its mu and lambda
  errors, not accuracy in either modulus.
