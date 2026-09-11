# Cylindrical-finger X shaping

Historical experiment. The active paper figure now uses the fixed rounded-X
comparison in `docs/x_letter_study.md`; use its sources and commands to
regenerate the current figure. Results below retain their original scope.

This is a separate simulation study for the proposed Franka shaping experiment.
It uses position-controlled pinches and explicit approach, opening, lifting,
and rotation. It does not use tool-work limits, force stopping, shape feedback,
or online replanning. Hardware execution and identification from real images
are not claimed. The study was adopted in `paper/icra2027/paper.tex` on
2026-09-09. Results and detailed checks are recorded in
`paper/evidence/x-shaping.md`.

## Source map

All experiment modules are under `experiments/robotics/`.

| Module | Role |
| --- | --- |
| `x_shaping.py` | Cylinder SDF, fixed X geometry, specimen sampling, recorded motion execution; CLI for isolated exploratory runs |
| `x_shaping_identification.py` | Separate simulated press calibration and held-out force prediction |
| `x_shaping_study.py` | Frozen equal-budget planning, true-material execution, numerical repeats, and verification |
| `x_shaping_run.py` | Subprocess orchestration and retained logs |
| `x_shaping_retime.py` | Execute the selected plans with a slower raised turn; preserve and compare original results |
| `x_shaping_audit.py` | Contact geometry, connected surfaces, reconstruction sensitivity, and independent replay checks |
| `x_shaping_report.py` | Exploratory views, full comparison, and four-panel paper preview; explicit destination required to replace an asset |
| `x_shaping_franka.py` | Proposed mount geometry, Panda kinematic checks, hand meshes, and offline joint trajectories |

Primary outputs are `out/x_shaping_identification_20260909/` and
`out/x_shaping_study_20260909/`. Earlier `hex_shaping_*` studies remain separate.
The generic isolated-run CLI reads the older hexagonal study's model database
unless a modulus/yield override is explicitly passed. The canonical study
instead imports the new calibration folder specified below. Do not substitute
the similarly named scripts or their default input models.

## Materials and calibration

Both true materials have E = 80 kPa, nu = 0.30, and density = 1000 kg/m³.
Their von Mises yield thresholds are 1 kPa (A) and 10 kPa (B). These match
the earlier rectangular experiment's definitions. They differ from the
subsequent hexagonal experiment, where B used E = 240 kPa. The common nominal
model has E = 80 kPa, nu = 0.30, and yield = 5.5 kPa, the midpoint of the two
true thresholds. No model is changed during planning or execution.

The calibration is regenerated, not silently inherited from the 240 kPa B.
A 14 mm downward press at approximately 80 mm/s on a 120 × 80 × 60 mm block
provides noise-free particle positions, velocities, elastic deformation
gradients, masses, and reference volumes. The weak-balance fit estimates the
elastic coefficients, followed by the existing elastic-strain plateau yield
estimator. Geometry, contact, density, and the Hencky/von-Mises family are known.
These observations include simulator state that RGB-D cameras do not directly
provide. Neither measured stresses nor plate forces enter these fits.

The regenerated parameters are:

| Material | E (Pa) | nu | Yield (Pa) |
| --- | ---: | ---: | ---: |
| A | 78795.494049 | 0.295881420 | 988.039678 |
| B | 77730.623474 | 0.300387036 | 9713.390265 |

The held-out press uses 24 mm displacement at 120 mm/s. Relative L2 force
errors are 1.420370% and 2.842983% for the identified models, and 426.340140%
and 40.327520% for the nominal model. The full data, fits, source hashes, and
force-error calculation are retained in the calibration directory.

## Fixed task and control

The shaping specimen is smaller than the calibration block: 60 × 60 × 25 mm,
with exact initial volume 90 mL. The target footprint is a 70 mm square with
a 14 mm-radius inward semicircle centered on each side. It has four corner
arms and four rounded notches. It is extruded to match the initial volume;
height is computed from the polygon area, approximately 24.5 mm. Each
semicircle has 64 segments. The target is common to both materials and all
models, and is fixed before their planning searches.

Two vertical cylinders of diameter 28 mm and height 45 mm begin with a 72 mm
physical gap. The first pinch closes along x and the second along y. Both
are centered on the original specimen center. The ONLY optimized variables
are these two final physical gaps. Centers, directions, height, and speeds
are fixed. All planners use 24–52 mm bounds, initial [36, 36] mm, and the
same simplex [36, 36], [28, 36], [36, 28] mm. Each bounded Nelder–Mead search
has 32 objective evaluations; convergence tolerances are zero so all models
receive the same budget. Commands are rounded to 0.000001 mm for caching.
The best observed point is selected, without claiming a global optimum.

Each pinch includes:

1. Lower the open pair from a 60 mm bottom clearance to 0.5 mm above the floor
   at at most 100 mm/s.
2. Close symmetrically to the planned physical gap at at most 50 mm/s per finger.
3. Open back to 72 mm at the same per-finger speed.
4. Lift to the original clearance at at most 100 mm/s.

The reported execution rotates the pair by 90 degrees in 1 s while raised
between the two pinches, with angular speed pi/2 rad/s. This is the gripper's rotation about the
specimen center; each circular cylinder has no commanded axial spin, which
does not affect its circular collision geometry. The current contact model
uses translational tool velocities; axial rotational friction at the cylinder
surface during this raised, noncontact rotation is therefore immaterial.
Final shapes are evaluated 1 s after the second full withdrawal. This is a
specified observation time, not a claim of mechanical equilibrium.

The 4 ms command interval quantizes each phase duration upward and reduces
its constant translational speed to reach the exact endpoint. Each MPM
substep still advances the moving collider. Rotations use samples on a
circular arc, with a linear chord within each command interval. Tools are
never removed, teleported, or disabled during this experiment.

The frozen planning searches used a 0.5 s raised turn. The subsequent Panda
check found that its pi rad/s angular speed exceeded the documented 2.5 rad/s
Panda Cartesian limit. `x_shaping_retime.py` executes every selected plan and
numerical repeat with the 1 s turn, retaining the original rollouts separately.
It checks that the gaps, phase endpoints, and all other phase durations are
unchanged, and records the resulting changes in surface and particle errors.
The figure must explicitly select `hardware_timing/`; that name denotes a
simulation timing adjustment, not a hardware experiment. The raised turn has
zero simulated contact force. The selected plans are not reoptimized after
this timing adjustment.

Contact is separable Coulomb friction: coefficient 0.3 for the fingers and
0.5 for the floor. There is no central anchor rod, glued boundary, or adhesion.
The cylinder signed distance is sampled at 1 mm spacing, with a 0.2 mm contact
band. Reported gaps are between geometric cylinder surfaces; the effective
contact boundaries are 0.4 mm closer together. Contact parameters are supplied
simulation assumptions, not quantities identified from the press.

Each model plans using only its own parameters. Its chosen gaps are then
executed in the true material without adjustment. The one nominal plan is
applied unchanged to both A and B. Two additional searches with the true
parameters provide finite-budget references.

## Geometry metric and numerical protocol

The objective and reported error are the symmetric mean of the two directed,
area-weighted mean surface distances, in millimetres. Triangle centroids are
quadrature points; closest-point distances use VTK's implicit-distance
evaluator. There is no alignment, rescaling, or subtraction of reconstruction
bias. The target mesh has consistent outward normals and three linear
subdivisions. Its volume and closedness are checked independently.

Specimen surfaces deposit reference particle volumes on 1.25 mm voxels,
apply Gaussian smoothing with sigma = 1.3 voxels (1.625 mm), and use marching
cubes at density 0.5. The same reconstruction serves the objective and figure.
This resolution differs from the larger hexagonal experiment; their absolute
error values do not describe the same task or reconstruction.

Baseline MPM uses a 64³ grid in a 160 mm domain, a 50 microsecond timestep,
and nominally two particles per cell direction. Particle counts are rounded
per dimension to preserve the exact 60 × 60 × 25 mm bounds and total volume.
Seed 0 adds jitter of at most 0.2 particle spacings. The floor is fixed at
10 mm in every numerical setting. Selected gaps are executed again with a
25 microsecond timestep and separately on an 80³ grid, without replanning.
The physical geometry, floor, target, and volume remain fixed.

Mean surface errors are 1.802059 mm for the nominal plan, 1.641764 mm for
identified plans, and 1.641696 mm for true-parameter references. Identification
reduces A's error by 15.19% and B's by 2.69%. Both improvements persist in the
numerical and reconstruction checks; B's improvement is modest. Plane contact
acts on grid velocities, so a small particle fraction penetrates the floor
by less than half a grid cell. The audit records this without clipping particles
or correcting output surfaces; it is a numerical contact limitation.

## Proposed Franka mount

The Panda visualization uses the locally installed MuJoCo Menagerie Panda
model through `robot_descriptions`. Original finger meshes are hidden and
replaced visually by the simulated cylinders and schematic adapters. The
proposed cylinder axes are 10 mm outward from each carriage, with their centers
85 mm along the hand's local z axis. With these offsets, stock gripper travel
equals physical gap plus 8 mm. The 72 mm opening maps to 80 mm travel; all
planned closing gaps remain within this range.

For the offline kinematic check, the specimen is centered at (0.43, 0) m in
robot coordinates and supported at z = 0.05 m. The hand stays downward, with
yaw determined by the pinch direction. Inverse kinematics checks sampled
poses against joint position limits and exports the resulting joint and tool
trajectories. With `--check-speeds`, it also checks sampled joint velocities,
hand translation, and hand angular velocity against the manufacturer's Panda
limits. Piecewise constant simulation commands have ideal starts and stops;
acceleration, jerk, torque, collisions, actuator dynamics, and hardware
execution are not validated. Actual adapters, TCP, maximum opening,
support friction, speed tracking, and the real identification measurements
must be established before reproducing the experiment on the physical arm.
The robot XML, meshes, and asset license are copied into the study's
`franka/panda_model_snapshot/` directory. The report uses that frozen copy.
For an exact reconstruction of the reported robot illustration in a fresh
study, copy the archived `hardware_timing/franka/panda_model_snapshot/` into
the same relative destination before running the Franka audit. Otherwise the
audit freezes the locally installed Menagerie model, which may differ.

## Exploration retained separately

Before freezing this task, `out/x_shaping_pilot_20260909/` tested the preceding
hexagonal material definitions, two identical gaps of 36 or 28 mm, and a
64 mm-wide volume-matched target. Both materials formed X-like shapes, but
their released responses were close. `out/x_shaping_tick4_20260909/` checked
the coarser command interval. The 70 mm target was chosen during feasibility
work to reduce the height increase demanded by the narrower target.

`out/x_shaping_material_pilot_20260909/` tested E = 80 kPa and yield = 20 kPa
for B, plus a 10.5 kPa midpoint model. The deep 24 mm pinch produced separated
lobes, a negative result that is not interpreted as validated physical fracture.
`out/x_shaping_b10_pilot_20260909/` tested the 10 kPa material subsequently
used in the frozen comparison. Each rollout retains its exact configuration
and source hash. Reconstructed source versions matching every exploratory
hash are in `out/x_shaping_exploration_20260909/sources/`.

A subsequent bounded test in `out/x_shaping_repeat_pilot_20260909/` repeats
the two identified openings once, without another optimization. Under the
original turn timing, A's surface error changes from 1.518 to 1.336 mm and B's
from 1.766 to 1.729 mm. This is a four-action pilot, not an equal-budget
comparison of four-action planners. The main study retains two pinches.

## Reproduction

Use the repository `.venv` and fresh output directories. GPU identifiers can
be changed explicitly. On the original run, three independent pipelines share
GPU 0; GPU 1 is left available for other project work.

```bash
X_CALIBRATION=out/x_calibration_reproduction
X_STUDY=out/x_study_reproduction
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_shaping_identification --out "$X_CALIBRATION" --device cuda:0
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_shaping_study prepare --out "$X_STUDY" --source "$X_CALIBRATION"
.venv/bin/python -m experiments.robotics.x_shaping_run --out "$X_STUDY" --devices cuda:0 cuda:0 cuda:0
.venv/bin/python -m experiments.robotics.x_shaping_study verify --out "$X_STUDY"
.venv/bin/python -m experiments.robotics.x_shaping_retime run --out "$X_STUDY" --device cuda:0
.venv/bin/python -m experiments.robotics.x_shaping_retime verify --out "$X_STUDY"
.venv/bin/python -m experiments.robotics.x_shaping_audit --out "$X_STUDY/hardware_timing" --planner "$X_STUDY"
.venv/bin/python -m experiments.robotics.x_shaping_franka --out "$X_STUDY/hardware_timing" --check-speeds
.venv/bin/python -m experiments.robotics.x_shaping_report --out "$X_STUDY" --execution "$X_STUDY/hardware_timing"
.venv/bin/pytest -q tests/test_x_shaping.py tests/test_hex_shaping_surface.py
.venv/bin/ruff check experiments/robotics/x_shaping*.py tests/test_x_shaping.py
```

The runner executes planning, baseline evaluation, and numerical repeats for
each model. It saves separate subprocess logs and explicit GPU assignments.
Every planning evaluation retains particle positions, stage states, complete
tool-center trajectories, phase endpoints, reaction-force diagnostics, and
its objective. Saved rollouts can be reused when resuming a search, but cached
evaluations still count against its fixed budget. Source and imported-data
hash guards prevent mixing changed inputs with existing results.
GPU floating-point reductions need not be bit-identical between fresh runs;
small changes can affect a finite-budget search on a nearly flat objective.
The saved raw rollouts support exact regeneration of the reported metrics
and figures independently of a new optimizer run.

To adopt the checked four-panel figure, provide an explicit destination and
then rebuild and inspect the active paper:

```bash
.venv/bin/python -m experiments.robotics.x_shaping_report --out "$X_STUDY" --execution "$X_STUDY/hardware_timing" --dest paper/icra2027/figs/identification_plastic_shaping
./paper/icra2027/build.sh
```
