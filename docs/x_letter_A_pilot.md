# Material A: rounded-letter X pilot

Historical experiment. The active paper figure now uses the fixed rounded-X
comparison in `docs/x_letter_study.md`; use its sources and commands to
regenerate the current figure. Results below retain their original scope.

This adaptive feasibility study is separate from the notched-square experiment
in `docs/x_shaping.md` and from the active paper figure. It uses the true
parameters of material A: Hencky/von Mises, E = 80 kPa, nu = 0.30,
yield stress = 1 kPa, density = 1000 kg/m³. It is not an identification result
or a physical Franka experiment. Material B is not run in this pilot.

A subsequent target-design step is documented in `docs/x_letter_target_redesign.md`.
The results below retain their original target and numerical meaning.

## Source and data map

- `experiments/robotics/x_letter_target.py`: the approved rounded capital X
  and an alternative with thinner arms.
- `experiments/robotics/x_letter_pilot.py`: initial two/four-pinch trials,
  using the original cylindrical-finger executor.
- `experiments/robotics/x_letter_refine.py`: explicit control over pinch
  order and repeated pinches; separately frozen target candidates.
- `experiments/robotics/x_letter_report.py`: reconstructs and renders saved
  released particle positions. It does not modify simulation data.
- `out/x_letter_A_20260909/`: first ten exploratory trials, including
  unsuccessful deeper pinches and smaller fingers.
- `out/x_letter_A_refine_20260909/`: subsequent order, repetition and
  gap-search trials, including their unsuccessful candidates.

Each output directory contains source snapshots and SHA-256 hashes, target
meshes and hashes, a protocol, environment records, commands, logs and raw
rollouts. Each rollout has a configuration, complete tool-center/force/time
history, phase endpoints, intermediate pressed and released particle states,
and the final particle positions and velocities. Source guards reject changed
code. Run drivers are retained with their batch names in the output folders.

The inherited `x_shaping.CONFIG` still contains `target_width` and
`target_notch_radius` fields for the older paper experiment. They do not define
this pilot's target. Use `x_letter_target` / `x_letter_refine.TARGETS`, the
protocol's explicit target dictionary and the hashed target meshes. Calling
`x_shaping.score` directly would score the wrong geometry.

## Geometry and measurement

The first pilot and refinement batches start from the same 60 × 60 × 25 mm
block in every run (90 mL). A separately archived initial-shape variant is
described below.
The approved target is a rounded capital X with nominal width 78 mm, height
86 mm, 20 mm-wide horizontal arm ends, 14 mm inward fillets, and 2 mm outward
fillets. Its extrusion thickness is 26.698874 mm to conserve the reference
volume. `x_letter_target.outline` defines all twelve vertices and fillets.

The initial pilot also scores a thinner-arm version. The refinement batch
declares two smaller alternatives before its simulations: `compact` and
`compact_thin`. These are different tasks. An error decrease obtained by
changing a target must not be presented as improved control for a fixed target.
All target-specific scores are saved, including scores for rejected targets.

The objective is symmetric, area-weighted mean 3D surface distance, with
triangle-centroid quadrature and closest-triangle distances. Reference particle
volumes are deposited on 1.25 mm voxels, smoothed with sigma = 1.3 voxels,
and extracted at density 0.5. No rigid alignment, rescaling, clipping or
reconstruction-bias subtraction is used. The supplementary top-view IoU is
measured from orthographic silhouettes in common world coordinates.

## Motion and numerical settings

All pinches are centered on the initial specimen center. Two vertical cylinders
approach while open, close symmetrically to the commanded physical gap, reopen
to 72 mm, and withdraw. The pair then turns 90 degrees while raised before the
next pinch. Finger diameter and final gaps vary across declared trials; contact
coefficients, material, object, and speeds remain fixed. Smaller gaps mean
deeper pinches. No tool-work cap, force stop or shape feedback is used.

Each finger closes/opens at at most 50 mm/s; approach/withdrawal is at at most
100 mm/s. A raised quarter-turn takes 1 s. Finger height is 45 mm; the bottom
stops 0.5 mm above the floor and lifts to 60 mm clearance. The final observation
is 1 s after full withdrawal. Friction is separable Coulomb, 0.3 on the fingers
and 0.5 on the floor. The collider SDF spacing is 1 mm and contact band 0.2 mm.

Baseline simulation uses a 64³ grid in a 160 mm domain, dt = 50 microseconds,
4 ms tool-command intervals and particle seed 0. All runs use GPU 0. Repeating
the same physical case can differ slightly because of parallel GPU reductions.
The extended executor reproduces the original two-pinch tool trajectory and
timestamps exactly; its final particle RMS difference was 0.000480 mm.

The simulated colliders represent cylindrical fingers. The full Franka arm is
not simulated. New repeated/order-reversed motions are not automatically
covered by the older paper experiment's Panda kinematic audit.

## Adaptive search history

The first ten runs vary cylinder diameters of 28, 24 and 20 mm, pinch depths,
and two versus four actions. Subsequent batches test reversed orientation,
four/six pinches, and appropriately assigning different gaps to the vertical
and horizontal directions. The drivers `order_trials.py` and
`directed_trials.py` are retained in the refinement output directory.

The `four_independent_balanced` search then optimizes four independent final
gaps for directions y, x, y, x, with 28 mm cylinders and the approved target.
It uses bounded Nelder–Mead, 20 objective calls, initial gaps [43, 26, 43, 26]
mm and four additional simplex vertices obtained by reducing one gap by 3 mm.
Bounds are 16–54 mm. This is an adaptive best-observed result, not a global
optimum or an equal-budget comparison of methods. To replay its recorded
configuration, including cache hits:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_letter_refine search --tag four_independent_balanced --target balanced --gaps 43 26 43 26 --passes 1 --radius 14 --first-angle 90 --budget 20
```

`target_options.py` is a separate, explicitly post-hoc design exploration:
it evaluates shorter X targets with wider ends against a saved four-pinch
result. Its numbers are not controller improvements, and those alternatives
do not enter the `four_independent_balanced` search.

### Initial-shape variant

`out/x_letter_A_initial_shape_20260909/run.py` tests a 60 × 70 × 21.428571 mm
initial block, still 90 mL. It inherits the frozen material, contact, speed,
solver and approved X target. The change is the physical starting specimen,
not a transformation of the rendered result. The purpose is to test whether
starting with a longer, thinner block gives longer arms and a lower final top
surface without adding actions. Its output directory has a separate protocol,
  source snapshot, hashes and raw trajectories. The driver itself is hashed.

The first trials transfer [43, 26, 43, 26] mm and the selected square-block
plan without changing their gaps. They do not constitute a controlled comparison
of identification methods, or an equal-budget comparison of starting shapes.
The 70 mm initial extent fits within the 72 mm geometric opening; any approach
contact is included by the continuous collider simulation.

The transferred optimized plan gives 2.105145 mm mean surface error and
72.2047% footprint IoU on this longer block, compared with 2.117957 mm and
71.2024% on the original square block. The difference is small. The square
block is retained for the final pilot result; the alternate initial shape is
not adopted, and these two tests do not establish its best achievable result.

## Selected square-block result

The selected case is `rollouts/e2d1cafe06d4ab52.npz` in the refinement output
directory. It is the lowest observed approved-target error among the 20 calls
in `searches/four_independent_balanced/`. The control parameters are:

| Pinch | Closing axis | Final physical gap (mm) |
| --- | --- | ---: |
| 1 | y | 20.090210 |
| 2 | x | 30.726318 |
| 3 | y | 34.580444 |
| 4 | x | 20.448608 |

Cylinder diameter is 28 mm. The original 60 × 60 × 25 mm specimen and approved
X target are retained. Baseline released mean surface error is 2.117956702 mm;
top-view IoU is 71.202356%. The first trial against this same target had an
error of 3.573176 mm. The final shape is recognizably an X, but its arms remain
blunter and its top less flat than the target. This is a feasibility result,
not a claim of a precise final shape or of optimal controls.

The selected plan can be replayed directly with:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_letter_refine run --gaps 20.090210 30.726318 34.580444 20.448608 --radius 14 --first-angle 90
```

`validate_selected.py` reruns these same commands with dt = 25 microseconds,
then with an 80³ grid at dt = 50 microseconds. It also prepares a separate
output directory and independently repeats the baseline case. It never
replans. `finish_report.py` verifies the raw data, audits the selected cases,
renders `material_A_result.png` and `material_A_result_actions.png`, and writes
an artifact hash manifest. Both drivers are in the refinement output folder.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m out.x_letter_A_refine_20260909.validate_selected
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m out.x_letter_A_refine_20260909.finish_report
```

Those commands reuse existing raw rollouts when their exact configuration is
already cached. To execute a fresh baseline simulation, choose a new output
directory:

```bash
.venv/bin/python -m experiments.robotics.x_letter_refine prepare --out out/x_letter_A_fresh_replay
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_letter_refine run --out out/x_letter_A_fresh_replay --gaps 20.090210 30.726318 34.580444 20.448608 --radius 14 --first-angle 90
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_letter_refine verify --out out/x_letter_A_fresh_replay
```

Preparation reads the original pilot's source/environment manifest, so retain
`out/x_letter_A_20260909/`. Use the recorded source versions and Python packages;
the guards intentionally reject modified experiment or solver code. Source
snapshots should be restored in an isolated checkout, not over unrelated work.

### Numerical replay results

All rows below execute the selected gaps without replanning, against the
same target and with the same 90 mL initial square block.

| Check | Grid | dt (microseconds) | Mean surface error (mm) | Top-view IoU |
| --- | ---: | ---: | ---: | ---: |
| Selected baseline | 64³ | 50 | 2.117956702 | 71.20% |
| Smaller timestep | 64³ | 25 | 2.094312532 | 72.15% |
| Finer grid | 80³ | 50 | 2.052917305 | 74.38% |
| Independent baseline repeat | 64³ | 50 | 2.117970861 | Not rendered |

The independent repeat has a final particle RMS difference of 0.001351 mm
from the selected baseline. These checks support numerical repeatability at
the tested settings; they are not a convergence proof or hardware validation.

All three audited surfaces are connected. Peak force per finger is about
1.2 N, with zero force during the initial approach and raised turns. The
surface error changes by less than 0.03 mm when reconstruction voxel spacing
is varied between 1.0 and 1.5 mm. The full audit is `selected_audit.json`.

Plane contact is enforced on grid velocities. The baseline has a maximum
particle penetration of 1.55 mm below the floor (2.58% of particles, with
volume-weighted mean depth 0.0144 mm). The finer grid reduces these values
to 1.32 mm, 2.01%, and 0.0093 mm. Particles are not clipped or corrected in
the saved data or distance calculation. This is a numerical contact limitation.

`material_A_result.png` shows the baseline. `material_A_fine_result.png` shows
the finer-grid replay, with its own 2.053 mm / 74.4% metrics. Both use exactly
the same selected controls. To regenerate the finer-grid image:

```bash
.venv/bin/python -m experiments.robotics.x_letter_report --out out/x_letter_A_refine_20260909 --refinement --case out/x_letter_A_refine_20260909/rollouts/8d4624c8e588bb9d.json --target balanced --stem material_A_fine_result --sequence
```

## Verification commands

Run from the repository root with the existing `.venv`:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_letter_pilot verify
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_letter_refine verify
.venv/bin/ruff check experiments/robotics/x_letter*.py
```

Verification rechecks frozen sources, raw data finiteness, zero inverted
particles, total reference volume, recorded gaps, all phase endpoints,
zero contact force during raised turns, final wait duration and every saved
target score. Completed searches also check that the selected case is the
lowest observed error for their declared target and evaluation budget.
