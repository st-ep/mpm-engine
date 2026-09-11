# Force-command planning for the rounded X

**Status:** exploratory force-hold study, not the active paper result. A’s
force plan was successful, but B’s search remained poor. The nine coarse
B trials were completed; the subsequent refinement was stopped after 9 of
16 objective calls to test work-based stopping. `study_status.json` records
the completed work. No B force plan was executed on the true materials,
and the commands below describe the full proposed workflow, not completed
validation. The replacement study is documented in `docs/x_work_study.md`.

This study replaces prescribed final finger openings with four commanded
force amplitudes. It uses the same material laws, specimen, target, and
cylindrical contacts as the preceding position-controlled X study.

The physical executor is `experiments/robotics/x_force_control.py`.
Planning and execution use `experiments/robotics/x_force_study.py`; audits
and figure generation use `experiments/robotics/x_force_report.py`.
The main archive is `out/x_force_study_20260909_v2/`.
The earlier `out/x_force_study_20260909/` contains five completed
initialization trials that overcompressed the material. Those trials are
not the final comparison, and configurations without completed NPZ files
are not results.

## Task, inputs, and controls

The frozen target is the same rounded X, 64 by 80 mm in plan, with
90 mL volume. Its SHA-256 is
`6e3e29fb5d3a50cb3466d7ec30d3ec07cca6a339fc09f13510931c8bd7710784`.
Earlier true-A feasibility trials informed this target. No new target
adjustment is made for force control.

Initial specimens are 60 by 60 by 25 mm, with density 1000 kg/m³.
Both true materials have Hencky elasticity with E = 80 kPa and nu = 0.30;
their von Mises yield thresholds are 1 kPa (A) and 10 kPa (B).
The separately identified models are unchanged:

| Model | E (Pa) | nu | Yield (Pa) |
| --- | ---: | ---: | ---: |
| Identified A | 78795.494049 | 0.295881420 | 988.039678 |
| Identified B | 77730.623474 | 0.300387036 | 9713.390265 |

These parameters came from separate 14 mm compression experiments on
120 by 80 by 60 mm blocks, with noise-free supplied particle positions,
velocities, elastic deformation gradients, masses, and reference volumes.
The known-family weak-balance fit and elastic-strain plateau estimator are
unchanged. Shaping outcomes are not identification data. The original
held-out 24 mm press force curves are copied exactly for figure panel (b).
Their original files and the extracted force/displacement arrays are hashed.

The four pinches are centered on the specimen and alternate y, x, y, x.
Each uses the PI feedback law qualified in `docs/x_pinch_force_pilot.md`:
the mean signed compressive force per finger drives symmetric inward speed,
with Kp = 0.050 m/(s N), Ki = 0.040 m/(s² N), a 12 ms force filter,
a 4 ms control period, and conditional integration under saturation.
The force is the measured grid reaction impulse divided by elapsed time.
Opposing jaw forces are measured separately; their cancelling vector sum
is not used as gripping-force feedback.

Two additional regression runs exercise the generalized executor with just
one 1.25 N pulse. Relative to the qualified single-pinch archive, released
particle positions differ by 0.001150 mm RMS for A and 0.000357 mm for B;
per-finger force traces differ by less than 0.000044 N RMS. Initial particles
and commanded pulse samples match exactly. These are implementation checks,
not additional planning outcomes. Their script, raw states, and measurements
are retained as `controller_equivalence.py`, `equivalence_*.npz`, and
`controller_equivalence.json` in the study archive.

Each pinch lowers the open fingers, approaches from 72 to 64 mm opening
at 10 mm/s per finger, waits 0.2 s, and applies a two-second force command.
Its profile has a 0.4 s raised-cosine rise, a 1.2 s plateau, and a 0.4 s
raised-cosine fall. Only the four peak amplitudes are optimized. The controller
can retract as well as close. Fingers then open and withdraw, and the hand
makes a one-second raised quarter turn before the next pinch. Observation
is one second after final withdrawal. Preparation, opening, withdrawal, and
their possible contacts are included in the simulation.

The 28 mm diameter cylinders remain 45 mm tall, with bottoms 0.5 mm above
the support. Tool and support friction are 0.3 and 0.5. The SDF uses 1 mm
cells and a 0.2 mm band. Per-finger speed is bounded by 50 mm/s and its
force-phase acceleration by 0.5 m/s². Physical opening remains between
4 and 72 mm. The lower bound is a common travel guard, not a target opening.
The nominal Panda adapter maps this range to 12 to 80 mm stock gripper
travel. The arm and actuator dynamics are not simulated.

## Planning and comparison

Each identified model plans independently against the same fixed target.
The initial force vector is 0.8 times the mean signed per-finger reaction
over the last 0.1 s of each pinch in that model's previously selected
position-controlled trajectory. This is a model-predicted initialization,
not a measurement from the corresponding true-material execution.
The previous position searches used 32 evaluations per model; their cost
is additional to the new force-planning budget.

An initial multiplier of 1.1 caused travel-limit activation in three
completed A and two completed B trials. Those trials were stopped and
archived before beginning the revised comparison. The multiplier of 0.8
and the revised feasibility penalty were then frozen for both models.

Each initial force search uses 32 bounded Nelder-Mead objective evaluations.
Variables are normalized by the initial force vector, and the initial
simplex adds 10% in each coordinate in turn. All physical force amplitudes
have the same bounds, 0.05 to 14 N per finger. Selection uses the lowest
surface error among feasible evaluated candidates. A fixed evaluation
budget does not establish optimizer convergence.

The objective is symmetric area-weighted mean closest-triangle surface
distance to the target, in mm. Reconstruction uses reference particle
volumes, 1.25 mm voxels, Gaussian sigma 1.625 mm, and isovalue 0.5.
There is no registration, rescaling, particle clipping, or bias subtraction.
A feasibility penalty adds 100 mm times the sum of per-pulse travel-limit
fractions, plus 100 mm times relative height excess above 45.5 mm.
The selected plans must have no travel-limit activation or sampled height
excess in their planning model. Failed solver trials receive an objective
of 1000 mm and retain their failure record.

B's initial search remained near weak deformations, with a best predicted
error of 2.740309 mm. The separate `x_force_refine.py` protocol therefore
broadens the first two force amplitudes in B's identified model. It tests a
3 by 3 grid: first-pinch multipliers 0.95, 1.00, and 1.03; second-pinch
multipliers 0.95, 1.05, and 1.15, applied to the unscaled model-predicted
initialization forces. The last two amplitudes stay at B's initial selected
values. Sixteen further Nelder-Mead evaluations start from the best feasible
candidate, with relative simplex steps 0.02, 0.04, 0.08, and 0.08.
The physics, target, objective, feasibility rules, and force bounds are
unchanged. These additional trials use no true-material shaping outcomes.
The initial selection is preserved as `refinement/B/selected_initial32.json`;
the expanded selection records the full evaluation count. Planning effort
therefore differs between A and B and is not a controlled runtime comparison.

Both selected force vectors are executed in both true materials. The force
commands, low-level controller, geometry, and contact settings are unchanged
between matching and exchanged plans. Measured force trajectories and
finger motions can differ. There is no shape feedback or online replanning.
Any travel-limit activation in a mismatched execution must be reported;
it is not successful tracking of the requested force.

The baseline is 64³ with a 50 microsecond MPM timestep. Each of the four
true-material executions is repeated at 25 microseconds and on an 80³
grid. Physical geometry, total reference volume, force commands, and
controller period stay fixed. Particle placement uses seed 0.
These checks assess numerical sensitivity, not convergence or repeated
hardware performance.

## Figure and reproduction

Panel (a) uses the completed single-pinch 1.25 N-command pilot. It shows
A under the cylinders and both released specimens with a common view and
scale. This is a material-response demonstration, separate from the press
used for identification. Panel (b) retains the held-out press force curves.
Panel (c) uses the last two pinches of the matching A execution, at their
minimum recorded gaps. Panel (d) compares both plans in both true materials.
All displayed material surfaces have the same color; labels identify the
material and planning model. The Panda hand is a kinematic illustration
with proposed adapters, not evidence of hardware force regulation.

From the repository root, use a fresh `--out` directory for reproduction,
passing it to every command:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_force_study prepare
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_force_study plan --material A --device cuda:0
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_force_study plan --material B --device cuda:1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_force_refine prepare --material B
for index in 0 1 2 3 4 5; do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_force_refine probe --material B --index "$index" --device cuda:1
done
for index in 6 7 8; do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_force_refine probe --material B --index "$index" --device cuda:0
done
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_force_refine finish --material B --device cuda:1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_force_execute_plan --plan A --device cuda:0
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_force_execute_plan --plan B --device cuda:1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_force_study verify
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_force_report audit
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_force_report diagnostics
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_force_report report
```

The two planning commands and, later, the two evaluation commands can run
concurrently. Both GPUs are used. Another account's GPU job could not be
stopped with the available OS privileges, so GPU 1 is shared during part
of this work. No runtime comparison is claimed.

The equivalent `experiments.robotics.x_force_execute_plan` driver groups
executions by planning model. Run `--plan A --device cuda:0` after A's search
and `--plan B --device cuda:1` after B's search. Each command runs both true
materials at all three numerical settings, so A's execution checks can begin
while B is still planning. This changes scheduling only. The driver source
and device assignment are archived. Use one scheduling scheme per archive:
the resume guards require the recorded device for an existing rollout.

The archive contains frozen sources, model parameters, target geometry,
initialization provenance, optimization evaluations, full control-rate force
and motion records, selected plans, numerical repeats, and figure provenance.
`source_sha256.json` and `input_sha256.json` prevent silent changes during
reproduction. The previous position-control and single-pinch archives remain
separate. Table II and its unresolved timing discrepancy are unrelated.
