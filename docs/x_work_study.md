# X shaping with commanded compression work

**Superseded on 2026-09-10.** The active paper figure now uses the common
cylindrical identification experiment documented in
[x_common_identification.md](x_common_identification.md), with new fits and
new plans. The account below and its rendering commands describe the historical
experiment; its archive is preserved unchanged.

The paper figure uses the common **12 mm minimum-opening guard**. Its
authoritative entrypoints are `experiments/robotics/x_work_guard_study.py`
and `experiments/robotics/x_work_guard_report.py`; the archive is
`out/x_work_guard12_study_20260909/`. They configure and reuse
`x_work_control.py`, `x_work_study.py`, and `x_work_report.py`.
Use the guarded entrypoints for this archive, with one protocol per process.
The unguarded entrypoints reproduce the earlier 4 mm experiment instead.

## What the controller commands

The four control inputs are positive compression-work targets, in joules,
for centered pinches along y, x, y, and x. Both fingers close at 50 mm/s.
At each 4 ms control tick, the controller sums the positive work supplied
by the two fingers:

`increment = sum_fingers(max(-reaction_force dot finger_velocity, 0)) * 0.004`.

It stops closing on the first tick when accumulated work reaches the target,
or when the physical cylinder-surface opening reaches 12 mm. The fingers
then open, withdraw, and turn through a raised quarter turn in one second.
There is no shape feedback or online replanning. The ending openings and
closing durations follow from the measured forces and work targets.

Work is summed over **both** fingers and only over the closing phase.
It includes elastic storage, motion, and dissipation; it is not a measurement
of plastic dissipation alone. The controller observes simulated contact
reactions and finger motion. It does not regulate an instantaneous force
profile. Threshold crossing can overshoot by at most one control tick of
work. The last tick's force, displacement, and work remain in the data.

The opening guard is a common software constraint, not a measured Franka
hardware travel limit. The earlier 4 mm guard left 1.6 baseline grid cells
between fingers and produced detached reconstructed pieces when B's work
commands were applied to A. The 12 mm guard provides 4.8 baseline cells and
lies below the intended stopping openings for both matching plans. This is
a resolution precaution, not proof of contact convergence. The material
model has no fracture law.

## How the commands are obtained

The original two identified-model gap plans are imported from
`out/x_letter_study_20260909/`. Each used 32 bounded Nelder-Mead objective
evaluations against the same rounded X, with the same initialization and
16–54 mm gap bounds. Earlier true-A feasibility work informed the target and
initialization; both were frozen before those searches.

For each model, the selected gap trajectory is replayed to predict positive
closing work. Closing speed is exactly 50 mm/s except for a possible shorter
last tick. The original gap executor slightly reduced the speed over the
whole closing phase to land exactly on a control tick. The new replay differs
only in that quantization detail. A separate model replay then executes the
predicted work targets to check that work-based stopping recovers the planned
shape. These two prediction runs and two model replays are additional to the
inherited gap searches. There is no new optimization against true-material
outcomes and no material or target adjustment.

Approximate work commands, in mJ for both fingers, are:

| Planning model | First y | First x | Second y | Second x |
| --- | ---: | ---: | ---: | ---: |
| Identified A | 44.70 | 45.05 | 15.09 | 13.69 |
| Identified B | 280.15 | 408.23 | 27.47 | 113.25 |

The complete unrounded commands are recorded in `plans/A/selected.json` and
`plans/B/selected.json`. Each vector is applied unchanged to both true
materials at every numerical setting. A guard stop means the requested work
was not reached and must be reported as such.

## Physics and identification

Initial specimens are 60 by 60 by 25 mm, with 90 mL reference volume and
density 1000 kg/m³. The fixed target is a rounded X, 64 by 80 mm in plan and
24.946587 mm high. Its SHA-256 is
`6e3e29fb5d3a50cb3466d7ec30d3ec07cca6a339fc09f13510931c8bd7710784`.

Both true materials have Hencky elasticity with E = 80 kPa and nu = 0.30,
and von Mises yield thresholds of 1 kPa (A) and 10 kPa (B). The unchanged
identified models have:

| Model | E (Pa) | nu | Yield (Pa) |
| --- | ---: | ---: | ---: |
| A | 78795.494049 | 0.295881420 | 988.039678 |
| B | 77730.623474 | 0.300387036 | 9713.390265 |

Identification uses separate 14 mm presses on 120 by 80 by 60 mm blocks,
with noise-free supplied particle positions, velocities, elastic deformation
gradients, masses, and reference volumes. The constitutive family, density,
geometry, and contact are known. Elastic moduli come from the weak-balance
fit and yield from the elastic-strain plateau. This comparison does not test
constitutive-family selection or perception from images.

Cylinders are 28 mm in diameter and 45 mm tall, with bottoms 0.5 mm above
the support. Their open gap is 72 mm. Tool and support friction are 0.3 and
0.5, with separable contact, a 1 mm SDF grid, and a 0.2 mm contact band.
The domain is 160 mm wide. Lowering and withdrawal use 100 mm/s; rotations
take one second. All preparation and withdrawal contacts are simulated.
Observation occurs one second after full withdrawal, without assuming exact
equilibrium. The Panda hand is a kinematic illustration with proposed mounts;
arm and gripper actuator dynamics, manufactured adapters, and hardware
force/work measurement are not validated here.

## Comparison and figure

The twelve true-material executions combine two actual materials, two plans,
and three numerical settings: 64³ with 50 microsecond timesteps; 64³ with
25 microsecond timesteps; and 80³ with 50 microsecond timesteps. Physical
geometry, work commands, reference volume, and control period remain fixed.
Particle placement uses seed 0. These are numerical sensitivity checks,
not statistical repetitions or a convergence proof.

The metric is symmetric area-weighted mean closest-triangle surface distance,
in mm, without registration, rescaling, particle clipping, or bias subtraction.
Reference-volume reconstruction uses 1.25 mm voxels, Gaussian sigma 1.625 mm,
and isovalue 0.5. Audits vary the voxel size, refine surface quadrature, inspect
connectivity and floor penetration, verify the work stopping condition, and
check sampled Panda joint ranges and velocities. They do not validate
acceleration, jerk, torque, or full-arm collision feasibility.

Panel (a) is the separately qualified 1.25 N-per-finger, two-second force-command
pilot from `out/x_pinch_force_pilot_20260909/`. It shows the same initial
geometry and different released responses; it is not the identification press
or the work-controlled shaping sequence. Its late-plateau mean forces are
1.1414 and 1.2727 N, so the common command does not mean identical measured
force. Panel (b) retains the original held-out 24 mm, 120 mm/s press curves
without modifying their arrays. Panel (c) shows the last two work-controlled
pinches of true A. Panel (d) uses the same camera, scale, target, and material
color for all four executions. A dagger marks a minimum-opening stop.

## Recorded results

Mean 3D surface errors in mm are:

| Numerical setting | A, plan A | A, plan B | B, plan A | B, plan B |
| --- | ---: | ---: | ---: | ---: |
| 64³, 50 microseconds | 1.044658 | 1.620778 | 2.896043 | 1.324430 |
| 64³, 25 microseconds | 1.038804 | 1.643685 | 2.895595 | 1.328243 |
| 80³, 50 microseconds | 0.941976 | 1.733867 | 2.829209 | 1.273118 |

Matching plans have lower 3D error in both materials at all three settings.
All matching runs complete every work command. Applying B’s commands to A
reaches the 12 mm guard on the first, second, and fourth pinches in every
setting. Those runs do not deliver the requested work on those pinches.
The opposite swap reaches the smaller work commands with little permanent
deformation of B. No claim of equal realized work is made for guard stops.

The A model replay has 0.060073 mm RMS particle-position difference from its work-prediction trajectory.
The B model replay has 0.120584 mm RMS particle-position difference from its work-prediction trajectory.

The 12 mm protocol replays the same inherited gap plans to calculate its
commands. Tiny differences from the 4 mm archive are floating-point replay
variation; the full command vector used in each comparison is frozen.

## Reproduction

From the repository root, use the same fresh `--out` directory in every
command. The commands below use `out/x_work_guard12_reproduction`; choose
another fresh directory if that path already exists. Preparation requires
an empty archive:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_work_guard_study prepare --out out/x_work_guard12_reproduction
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_work_guard_study predict --material A --device cuda:0 --out out/x_work_guard12_reproduction
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_work_guard_study predict --material B --device cuda:1 --out out/x_work_guard12_reproduction
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_work_guard_study selfcheck --material A --device cuda:0 --out out/x_work_guard12_reproduction
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_work_guard_study selfcheck --material B --device cuda:1 --out out/x_work_guard12_reproduction
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_work_guard_study evaluate --material A --device cuda:0 --out out/x_work_guard12_reproduction
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_work_guard_study evaluate --material B --device cuda:1 --out out/x_work_guard12_reproduction
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_work_guard_study verify --out out/x_work_guard12_reproduction
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_work_guard_report audit --out out/x_work_guard12_reproduction
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_work_guard_report diagnostics --out out/x_work_guard12_reproduction
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_work_guard_report report --dest paper/icra2027/figs/identification_plastic_shaping --out out/x_work_guard12_reproduction
./paper/icra2027/build.sh
```

The A and B workflows may run concurrently once both predictions exist.
Skip simulation stages to audit or render an existing archive. Source, input,
configuration, and asset hashes reject incompatible cache reuse. Archives
include full force/velocity traces, stopping reasons, model predictions,
numeric repeats, figure sources, and provenance. The two GPUs run separate
material workflows; no runtime comparison is claimed.

The original position comparison, force-hold trials, gain-scaling pilot, and
4 mm work study remain separate. In particular, their errors must not be
reassigned to the final guarded comparison. Table II and its pending timing
discrepancy are unrelated and remain unchanged.

## Completed audit and scope

All twelve executions pass the finite-state, work-stopping, speed, travel,
reference-volume, and non-inversion checks. Matching plans have lower 3D
error at each tested reconstruction voxel size (1.0, 1.25, and 1.5 mm), as
well as with refined surface quadrature. Baseline top-view silhouette IoU
is 89.04% versus 86.82% for A and 82.11% versus 62.19% for B, comparing
matching to swapped plans. All outcomes fit the same camera and scale.

Baseline and half-timestep reconstructions each have one connected surface.
The finer-grid A execution with B's plan has three reconstructed components;
the largest contains 99.573% of surface area. The two small
pieces are retained in the metric, and are not validated fracture behavior.
The largest sampled final floor penetration is 1.640 mm. The maximum
work overshoot on completed commands is 6.392%, within one control
tick. These findings limit claims of convergence or exact contact.

The four baseline trajectories pass sampled Panda inverse-kinematics, joint
range, and joint-velocity checks. Raised turns and the final observation
phase have zero simulated finger reaction. These checks do not validate
acceleration, jerk, torque, full-arm collisions, actuator dynamics, or hardware.
See `additional_audit.json`, `franka/audit.json`, and
`work_stopping_diagnostics.png` in the final archive.

## Paper adoption

The work-controlled figure is adopted in `paper/icra2027/paper.tex` and
`paper/icra2027/figs/identification_plastic_shaping.pdf` (with a PNG preview).
The results, caption, and introductory description now match this protocol.
The rebuilt `paper/icra2027/paper.pdf` remains eight pages; pages 2 and 5–8
and the figure PDF were visually inspected for layout. Compilation succeeds
with seven pre-existing missing bibliography entries and the pre-existing
0.6606 pt overfull equation at TeX line 498. The build log, before/after
manuscripts, diff, figure provenance, page renders, and final checksums are
in the experiment archive. Table II and all unrelated source text are unchanged.

## Panel (a) layout revision, 2026-09-10 UTC

The active paper figure now uses `experiments/robotics/x_work_probe_panel.py`
for panel (a), through the same guarded report entrypoint. The three views
have equal image areas and a continuous white background. Their orthographic
cameras use the same magnification, verified by projecting a 10 mm reference
segment, and align the initial specimen center. The action view retains its
oblique angle and geometric foreshortening. Dashed lines on the top views
show the analytic initial 60 mm square. Specimen surfaces, recorded states,
commands, and errors are unchanged; panels (b), (c), and (d) are pixel-identical
to the previous rendering.

The rendering revision is archived separately in
`out/x_work_panel_a_layout_20260910/`, with before/after paper files,
`figure_provenance.json`, `renders/probe_view_calibration.json`, source copies,
build log, PDF inspection images, and checksums. All 338 files in the original
work-study archive retain their recorded hashes. Reproduce the current figure:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_work_guard_report report --out out/x_work_guard12_study_20260909 --render-out out/x_work_panel_a_layout_20260910 --dest paper/icra2027/figs/identification_plastic_shaping
./paper/icra2027/build.sh
```

The rebuilt paper remains eight pages. The figure PDF and paper page 7 were
visually inspected; the manuscript source and text on all other pages are
unchanged. Existing build warnings remain unchanged.

## Panel (a): equal visible heights, 2026-09-10 UTC

This supersedes the preceding layout revision. The three pictures now occupy
a common visible-height band, with Pinch/A/B labels in a separate header row.
Vertical crops remove the excess gripper context and empty image margins;
all specimen geometry is retained. All three views preserve the same physical
magnification. A and B retain a shared crop; the action view uses a separate
vertical crop, so the previous common initial-center alignment is superseded.
No recorded states, numerical results, or manuscript text change. Panels
(b), (c), and (d) remain pixel-identical.

The current rendering archive is `out/x_work_panel_a_equal_height_20260910/`.
Its source copies, crop/calibration metadata, visible-height check, before/after
paper files, build log, PDF inspections, and hashes record the revision.
The PDF compiles, remains eight pages, and page 7 and the figure were visually
checked. Reproduce using:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_work_guard_report report --out out/x_work_guard12_study_20260909 --render-out out/x_work_panel_a_equal_height_20260910 --dest paper/icra2027/figs/identification_plastic_shaping
./paper/icra2027/build.sh
```

## Squeezing motion cues, 2026-09-10 UTC

Panel (a) now labels the action “Squeeze” and overlays two inward blue arrows
with white outlines. Their directions are projections of the actual horizontal
finger-closing axis. Their lengths are illustrative, not force or displacement
measurements. The three underlying image renders, their crops and physical
scale, and panels (b)–(d) are pixel-identical to the preceding revision.
The manuscript source and numerical results are unchanged.

The current figure revision is `out/x_work_panel_a_motion_20260910/`, which
archives its sources, projected arrow endpoints, before/after paper files,
build log, PDF inspection images, and checksums. The eight-page PDF was rebuilt
and page 7 and the figure PDF were visually checked. Reproduce with:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_work_guard_report report --out out/x_work_guard12_study_20260909 --render-out out/x_work_panel_a_motion_20260910 --dest paper/icra2027/figs/identification_plastic_shaping
./paper/icra2027/build.sh
```

## Refined motion-arrow placement and color, 2026-09-10 UTC

The current panel (a) uses charcoal (#41434a) for its motion arrows and
“Squeeze” label, separate from the blue/orange material colors. Arrowheads
are anchored outside the fingers with 10 mm illustrative clearance; their
heights (47 and 39 mm above the world origin) use the exposed shaft regions.
Both directions follow the projected horizontal closing axis. Arrow lengths
and offsets are illustrative, not measurements. All three base images and
panels (b)–(d) remain pixel-identical; manuscript text and results are unchanged.

The current rendering archive is `out/x_work_panel_a_motion_refined_20260910/`.
It includes source copies, cue coordinates, before/after paper files, build
log, PDF inspections, and checksums. The rebuilt PDF remains eight pages;
the figure and page 7 were visually checked. Reproduce with:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_work_guard_report report --out out/x_work_guard12_study_20260909 --render-out out/x_work_panel_a_motion_refined_20260910 --dest paper/icra2027/figs/identification_plastic_shaping
./paper/icra2027/build.sh
```

## More gripper context and simpler arrows, 2026-09-10 UTC

The current panel (a) uses a separate zoom for the action view: its material
is 0.733 times the released views' screen scale, allowing more of the Panda
hand to appear within the same visible image height. Released A and B retain
their common scale and are pixel-identical to the preceding revision. The
caption explicitly distinguishes the action and released display scales.
Longer, slender, open-headed charcoal arrows replace the heavy outlined arrows.
Their direction follows the closing axis; they remain illustrative motion cues.
All experiment data, numerical results, and panels (b)–(d) are unchanged.

The current archive is `out/x_work_panel_a_gripper_context_20260910/`, containing
source copies, view calibration, before/after manuscript files, caption diff,
build log, PDF inspections, and checksums. The rebuilt PDF remains eight pages;
the figure and pages 7–8 were visually checked. Reproduce with:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_work_guard_report report --out out/x_work_guard12_study_20260909 --render-out out/x_work_panel_a_gripper_context_20260910 --dest paper/icra2027/figs/identification_plastic_shaping
./paper/icra2027/build.sh
```
