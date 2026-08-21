# Changelog

## Unreleased

### Added

- `experiments/nclaw/ingest.py`: the ingestion path for trajectories generated
  by NCLaw's OWN simulator, so their data feeds our identification and a
  cross-engine rollout without touching the solve code.
  - `read_nclaw_dir` turns their state folder (0000.pt, 0001.pt, ... holding
    dict(x, v, C, F, stress, sections, types) in their y-up frame) into one
    schema-valid npz: every vector and both legs of every two-point tensor
    rotated to z-up, their Kirchhoff stress divided by J to give Cauchy, the
    particle volume and density taken from a small json manifest (their state
    files carry neither), and their one-step stress lag realigned.
  - The velocity-gradient convention is MEASURED on the ingested data, not
    assumed: the acceleration-consistency probe runs on the arrays, the verdict
    and both residuals are stored in the dump metadata, and C is transposed only
    if the measurement says so.
  - Degradation tiers with per-channel provenance (measured, derived, absent):
    finite-difference velocities when v is missing, moving-least-squares
    velocity and deformation gradients over k reference-frame neighbours when C
    or F is missing, and an explicit refusal of the oracle pressure when the
    stress channel is missing.
  - `export_to_nclaw` writes one of our dumps in their exact format, including
    an option to reproduce their one-step stress lag, which is what lets the
    whole path be validated before any of their data exists.
- `experiments/nclaw/suite.py`: `cloud_from_dump` plus a `cloud` argument to
  `run_scene` seed a rollout from a PROVIDED particle cloud (frame-0 positions,
  reference volumes and velocities), so our rollout stands in 1:1 particle
  correspondence with the trajectory it is scored against. A new `cross` stage
  runs the whole chain in one command per material: ingest, identify from their
  trajectory, roll out from their frame-0 cloud, score in their metric.
  `stage_identify` takes an explicit dump and an output tag.
- `tests/test_nclaw_ingest.py`: 16 tests plus two opt-in slow ones. A
  manufactured trajectory is exported to their format and ingested back;
  positions, velocities, L, F, volume and mass come back bitwise identical, the
  probe verdict agrees with the dump probe, and the identified moduli agree to
  1e-9. Negative controls: a transposed C channel flips the verdict and is
  transposed back, and a folder read with the wrong stress lag misses by order
  one. One test runs the whole cross stage through the simulator on a 2197
  particle rollout; the two opt-in slow ones (NCLAW_ROUNDTRIP_DUMPS) run the
  same chain on the real suite dumps.
- `experiments/nclaw/`: the full NCLaw-matched comparison suite (scene
  generation at their published configs, weak-form identification, rollout,
  scoring in their metric, matplotlib and Blender side-by-side renderers,
  dump writer and L-convention probe), migrated from the research tree so the
  comparison reproduces from this repo plus a clone of NCLaw (set NCLAW_DIR
  or place the clone in a parent or sibling directory). Verified: the jelly
  identification reproduces bit for bit after migration.
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
- `experiments/nclaw/suite.py`: the NCLaw asset folder resolves on first use
  (`assets()`) instead of at import, so the ingest path imports the suite's
  rotation and its identify legs without a clone of their repository present.
- `identify_friction` refuses when the dump carries no oracle pressure, naming
  the dump's `pressure_source`, instead of reading a stress trace that is not
  there.
- `_wave_speed` refuses a non-physical law before the engine sees it (E <= 0, or
  nu outside (-1, 0.5)). Measured reason: such a pair makes lam + 2 mu negative,
  the time step NaN, and the first p2g2p writes outside the grid, which on this
  machine is a bus error rather than an exception.

### Measured

- Round trip of the ingestion path on the suite's own cube dumps (126 frames,
  35937 particles each), our dump out to their format and back:

  | dump | x, v, L, F, volume, mass | stress | theta agreement |
  | jelly_cube | bitwise identical | 8.0e-8 relative | E and nu exactly equal |
  | sand_cube | bitwise identical | 5.8e-8 relative | friction angle 2.6e-11 relative |

  The stress residue is the single float32 store of the Kirchhoff product
  J sigma, and it is the only lossy step in the path: declaring the channel
  Cauchy in the manifest makes the stress round trip bitwise as well. The
  L-convention probe returns the same verdict and the same residuals on both
  files (jelly frame 14: 0.0047 against L, 1.995 against L^T; sand frame 12:
  0.0019 and 1.907).
- Kinematics-only tier on the same jelly dump (positions in, velocities by
  finite difference, L and F by moving least squares over 24 reference-frame
  neighbours, no stress): E 1.0122e5 against the truth 1e5, 1.2 percent high,
  nu 0.1957, against 0.09 percent low for the measured-channel path. The sand
  leg on the same stripped folder refuses the friction angle for want of the
  oracle pressure and falls back to its prior, as it must.
- Elastic recovery on the TrackEUCLID drop dumps, time-weak route versus the
  radial-window reference: sphere E error 0.10 percent (radial 0.35 percent),
  mu error 0.21 percent (radial 3.89 percent); cube E error 0.07 percent
  (radial 13.4 percent), mu error 0.42 percent (radial 10.9 percent). The radial
  window's small sphere E error is a cancellation between its mu and lambda
  errors, not accuracy in either modulus.
