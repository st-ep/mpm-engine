# Changelog

## Unreleased

### Added

- Material 14, `composed`: one elasticity kind and one plasticity kind chosen
  independently per material type, through `set_parameters_dict("elasticity",
  ...)` and `("plasticity", ...)`. Elasticity kinds: `corotated`, `hencky`,
  `stvk`, `volume_taichi`, `volume_ziran`. Plasticity kinds: `identity`,
  `von_mises`, `drucker_prager` (with `cohesion`), `sigma`. Two of the
  elasticities are new to the engine, both transcribed from NCLaw's
  `nclaw/material/preset.py`: `stvk` is `2 mu F E_green + lam J (J - 1) I` (their
  StVKElasticity, and NOT the fork's retired `kirchoff_stress_StVK`, none of
  which is reused), and the two volumetric equations of state, `lam J (J - 1) I`
  and `kappa (J - J^(1 - gamma)) I` with `gamma` a parameter at their value 2.
  Existing materials are untouched: 14 is additive, and every kind it composes
  is either a new function or one an existing material already calls.
  This is what lets a cross-engine comparison run on the other engine's own
  constitutive pair. Measured: their water is a purely volumetric law with no
  deviatoric term, and rolling their water trajectories on it instead of our
  gamma = 1.1 power-law fluid moves the truth-parameter floor from 5.07e-3 to
  2.99e-12 position MSE. Their plasticine reproduces bit for bit through
  material 14 what material 1 gives (7.8411e-12 on the dataset scene, the same
  number to every digit), which is the check that composing is faithful.
- `tests/test_composed_material.py`: every kind against a reference formula
  transcribed from their preset.py with the class named, on a shared batch of
  random deformation gradients. The returned F is compared tightly (1e-5) and
  the stress loosely (1e-3), because the engine's Hencky stress from its own
  returned F differs from an exact float64 evaluation of the same formula by
  2.5e-4 relative: that is warp's fixed-iteration float32 `svd3`, measured and
  recorded rather than absorbed into a formula tolerance. A final test compares
  the transcriptions against their actual modules when NCLaw is importable, so
  the transcriptions are checked and not merely trusted.
- `experiments/nclaw/suite.py`: `run_scene(nclaw_law=...)` and per-material
  `nclaw_law` specs naming the elasticity and plasticity kinds NCLaw's own
  config uses for that material; `run_scene(substeps=...)` to fix the substep
  count per dumped frame instead of taking it from the CFL, which a cross-engine
  floor needs because the other engine's trajectory is its discrete solution at
  its own dt (their sand needs it: our CFL picks two substeps where they take
  one, and that alone held the sand floor at 2.0e-4 instead of 2.0e-11);
  `identify_eos(form="linear")`, the volumetric column `(J - 1) I` with lam as
  the single linear unknown, which recovers 5.8001e4 against their 5.7692e4 from
  their own water dataset scene.

- `MPM_Simulator_WARP.set_grid_semantics(...)`: five independent, opt-in grid and
  transfer options, all off by default.
  - `freeslip_bound`: freeslip walls applied inside the grid operator on the
    outer node layers of all six faces, zeroing the wall-normal velocity only
    where it points INTO the wall. This is what freeslip means, and it is what
    `add_surface_collider(..., "slip")` does NOT do: the collider projects the
    normal component out unconditionally inside its half-space, so a body
    leaving the wall is held against it. Prefer this option for domain walls.
  - `mass_eps`: grid velocity `mv / (m + eps)`, damping fringe nodes whose share
    of a stencil is a rounding error rather than a physical mass.
  - `empty_node_gravity`: zero-mass nodes carry `gravity * dt` into g2p.
  - `mls_transfer`: MLS-MPM transfer (Hu et al. 2018), stress on the affine
    channel and F advanced with the APIC matrix C, instead of the
    gradient-weight internal force and velocity gradient L.
  - `particle_clip_cells`: g2p clamps the advected position that many cells
    inside the box.
  Together they are NCLaw's grid operator and transfer (`nclaw/sim/mpm.py`), and
  what they buy is measured: on NCLaw-generated plasticine trajectories the
  cross-engine floor at truth parameters falls from 1.067e-3 to 7.8e-12 position
  MSE (dataset scene), a factor of 1.4e8, with the wall semantics alone worth
  141x. Asking for any of the three grid-operator options switches the
  normalization kernel, which forces dense full-grid sweeps, disables CUDA-graph
  capture and refuses the fused tick, because empty nodes have to be visited.
- `tests/test_grid_semantics.py`: the options are bit-identical-when-off
  (a stored digest of the pre-change engine's x, v and F on a standard scene),
  the freeslip clamp arrests an approaching particle exactly and lets a
  separating one leave at full speed on both the near and the far face, the
  collider slip plane demonstrably does not, and `mass_eps` softens what it
  should.
- `experiments/nclaw/suite.py`: `run_scene(nclaw_bc=...)` and
  `NCLAW_GRID_SEMANTICS`, so a cross-engine rollout can ask for NCLaw's grid
  semantics; the walls then come from the grid operator instead of the six
  collider planes. A dict overrides individual options, which is how the
  per-behavior attribution was measured.

- `experiments/nclaw/suite.py`: two rollout-only `MATERIALS` entries,
  `sand_table` (fork material 13, tabulated mu(I)) and `visc_table` (fork
  material 12, tabulated apparent viscosity), plus an `ENGINE_PASSTHROUGH` list
  that lets `engine_params` carry a recovered CURVE through to the engine as
  data (the table, its log-grid ends, and the grain scales that fix the inertial
  number the table is read at). This is what lets a function-encoder recovery be
  re-simulated on the NCLaw scenes without a parametric fit; the consumer is
  `sim/fe_ls_baseline.py` in the parent tree. The entries have no identify leg
  and no NCLaw published column, so the `--material` choices are now the four
  physical materials explicitly rather than every key of the dict.
- `ident/io/schema.py`: `tabulated_mu_i` and `tabulated_viscous` law tags, which
  the dumps those rollouts write carry. Gate code branches on `constant` and
  `pouliquen` and ignores the rest, so nothing existing changes.
- `tests/test_tabulated_mu_i.py`: the tabulated mu(I) material (13) at the
  Drucker-Prager cone constant reproduces material 2 on a column collapse to
  3.4e-6 m over a 0.21 m runout, and halving the tabulated friction spreads the
  column further. This is what makes a recovered-curve rollout comparable to a
  Drucker-Prager truth trajectory rather than a comparison across two return
  maps; measured at full scale on the NCLaw grid-20 cube as well, position MSE
  1.1e-13.

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
  `stage_identify` takes an explicit dump and an output tag, and the `identify`
  stage exposes both as `--dump` and `--tag`, so the least-squares row of the
  differentiable-simulation comparison is identified from exactly the grid-20
  trajectory that comparison uses.
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
- `tests/test_mu_i_return_map.py` and `tests/test_mu_i_phi_dilatancy.py`, plus two
  tests in `tests/test_tabulated_mu_i.py`, asserting what the four mu(I)
  verification probes in `video2sim/sim` only printed. The probes are archived
  with a pointer to the covering test. Two findings from writing them: the
  existing tabulated test locked material 13 to material 2 at a constant mu, not
  to the parametric material 9 the probe checked, and the probe's `corr(I, Phi)`
  statistic does not separate the dilatant material 11 from material 9 (material
  9 gives -0.452), so the new test asserts binned median Phi against Phi_c(I)
  instead, where material 11 tracks to 0.0022 and material 9 misses by 0.0245.
  `smoke_mu_i_phi.py` got no test: it checked that material 11 equals material 9,
  which stopped being true when material 11 gained its own pressure and stress
  branches.
- `ident/io/schema.py`: `KNOWN_LAWS`, extending the dump law tag with
  `corotated`, `vonmises`, `drucker_prager`, and `eos_fluid` so the
  NCLaw-matched comparison can write schema-valid dumps for its three
  non-granular materials. Gate code branches only on the two granular names, so
  no existing behaviour changes.

### Changed

- `experiments/` is now campaigns and an archive, nothing flat. Five campaigns
  are kept, each a package with a stages CLI, artifacts under `out/<name>/`, a
  test, and a row in the rewritten `experiments/README.md`, which reads as the
  index of what each one established: `nclaw`, `elastic`, `robotics`, `diffsim`,
  `fe_ls`. Everything else moved to `experiments/archive/` or
  `../sim/archive/` rather than being deleted, so paths in the writeups keep
  resolving.
  - `experiments/elastic/`: the eight-script elastic drop family from
    `video2sim/sim` (elastic_drop, elastic_grid_gate, elastic_identify_sequential,
    plastic_drop, plastic_identify_sequential, hyperelastic, hyperelastic_fe,
    sample_complexity) behind one CLI. `core.py` holds the single version of the
    drop scene, the dump reader, the radial interior test functions, the
    rotation-invariant validity filter, the FCR and Hencky stress bases, the row
    assembly and the grid-consistent recovery; those had two, and in four cases
    eight, drifting copies. Every headline number was reproduced from the
    consolidated code on the dumps in the tree before the originals were deleted:
    sphere radial E error 0.0034888047331791405, cube radial 0.134340294155481,
    grid gate sphere timeweak 0.0010229504340479867 and cube 0.000692258814248059
    with all four acceptance flags true, Mooney gentle probe C10 / C01 / Kbulk
    0.22101 / 0.36008 / 0.08380 at cond 713.42 over 5376 rows. The table is in
    `experiments/elastic/README.md`.
  - One behaviour change inside that move: the two sequential scripts filtered
    particles on `||F - I||`, which is not frame-objective, while the rest of the
    family had moved to `||log sigma(F)||`. `sequential.py` uses the
    rotation-invariant filter, which moves the streamed sphere posterior E from
    2.011713e5 to 2.006977e5, the latter being the batch least-squares estimate
    to seven digits.
  - `experiments/diffsim/` and `experiments/fe_ls/`: promoted from
    `video2sim/sim/{diffmpm,diffsim_identify,fe_ls_baseline}.py` with their three
    tests. Artifact directories `out/diffsim_baseline` and `out/fe_ls_baseline`
    are unchanged, and both packages resolve them in the staging tree while the
    runs live there.
  - `experiments/robotics/`: the Franka dough chain, identify force then predict
    then plan, with `common.py` holding the press scene, the lobedness metric and
    the CoTracker wrapper that three or two scripts each had duplicated. The
    press-scene extraction was checked to reproduce the inline version with zero
    particle-position difference before substitution. The other thirteen flat
    scripts are in `experiments/archive/`.
- `experiments/nclaw/suite.py`: plasticine maps to fork material 1 (`metal`,
  Hencky elasticity with the von Mises return) rather than material 5
  (fixed corotated with the same return). NCLaw's plasticine dataset config is
  `SigmaElasticity` + `VonMisesPlasticity`, and `SigmaElasticity` is Hencky
  (`nclaw/material/preset.py`). Measured effect on the cross-engine floor: small,
  1.078e-3 to 1.067e-3, because the two elasticities differ at second order
  below the 1.7 percent yield strain.
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
