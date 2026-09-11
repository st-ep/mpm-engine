# X shaping with consistent planning and execution timing

Status: all 64 planning simulations, six identified-model checks and twelve
true-material executions are complete. The reference-identity, controller,
geometry and sampled Panda kinematic audits pass. Results are recorded in
`paper/evidence/x-consistent-force-study.md`.

The entrypoint is `experiments/robotics/x_consistent_force_study.py`.
The study directory is `out/x_consistent_force_study_20260911`.
This revision removes the change of closing speed after gap optimization
in `out/x_force_profile_study_20260910`. Historical archives are retained.

## Planning and force references

Both identified models receive the same starting geometry, rounded X target,
initial search point `[20, 31, 35, 20]` mm, and evaluation budget. The target
and common initialization were informed by earlier A feasibility trials.
No material-specific warm starts or new material/target adjustments are used.

Bounded Nelder–Mead searches four nominal final openings in `[16, 54]` mm,
with 32 objective evaluations per material. Each evaluation simulates the
complete four-pinch sequence along y, x, y, x on a fresh specimen. During
each pinch, both fingers start at 72 mm opening and follow smooth closing
motions with 40 ms ramps and at most 20 mm/s per finger. The endpoint and
fixed timing rule determine the duration. Opening, withdrawal, raised
quarter turns, and the final one-second release use the existing protocol.

The objective is the symmetric, area-weighted mean distance between the
released specimen surface and the unchanged analytic target. Reconstruction
uses reference particle volumes, a 1.25 mm voxel grid, Gaussian smoothing
with sigma 1.625 mm, and isovalue 0.5. Every component is included, without
alignment, rescaling, clipping, or baseline subtraction. The optimizer
retains the best observed candidate at its evaluation limit; convergence
or global optimality is not claimed.

Every planning simulation prescribes the motion with force feedback
disabled and records the contact reactions. The selected candidate's
archive is copied byte-for-byte to `reference_A.npz` or `reference_B.npz`.
Its recorded velocity and force arrays become the execution profiles.
There is no additional reference simulation, retiming, or force optimization.
The associated filtered force samples are extracted from the same record
with the original causal alignment to the control interval.

## Execution and comparison

Execution retains the audited `x_force_profile_control.py` controller. Every
4 ms, a PI correction based on the difference between predicted and measured
mean outward force per finger is added to the baseline closing velocity.
Both fingers move symmetrically. Actual velocity can differ from the
baseline, and actual final opening is not constrained to the planned value.
The squeeze ends at its profile duration or the 12 mm minimum-opening guard.
Execution receives no shape feedback and performs no replanning.

The baseline speed limit is 20 mm/s per finger. The execution speed ceiling
is 50 mm/s per finger, leaving room for force corrections. Controller
updates limit acceleration to 0.5 m/s²; phase transitions and hard guard
stops remain kinematic. The 12 ms force filter, conditional anti-windup,
12–72 mm opening limits, and common gain-scaling rule are unchanged.
The previous common gain multiplier of 0.5 is fixed before this search.

Both identified models undergo baseline (64³, 50 microseconds), finer-grid
(80³, 50 microseconds), and baseline-repeat checks. All four profiles must
finish with no guard stop, at most 15% raw-force relative L2 error per
profile, and sampled specimen height at most 45.5 mm. Repeat differences
must remain below 0.10 mm in surface score and 0.5 mm RMS particle position,
with identical stop reasons. Failure prevents freezing and true execution.

After both models pass, complete plans and the controller are frozen.
Each plan executes in both true materials at baseline, half timestep
(64³, 25 microseconds), and finer grid: twelve true-material executions.
Swapping exchanges the entire baseline motion, force history, and duration.
Guard stops and force-tracking errors remain part of the reported outcomes.
The comparison evaluates material-dependent planning and execution together;
it does not isolate a benefit of force feedback over trajectory replay.

## Scope and provenance

Materials, supplied-state identification records, held-out validation,
target geometry, contact, density, gravity, scoring, and rendering are
inherited unchanged. Identification is not rerun in this revision. Its
noise-free positions, velocities, and elastic deformation gradients and
known constitutive family remain qualifications of the experiment.

Force feedback assumes ideal simulated finger-force measurements. Sampled
Panda kinematic checks do not establish physical gripper force control,
collision-free arm dynamics, or hardware feasibility. Grid/timestep checks
are sensitivity checks; subcell floor penetration is retained and exact
contact or numerical convergence is not claimed.

Each candidate saves its configuration, particle states, force and tool
histories, phase records, score, and checksum. Selected reference identity
and profile extraction are checked in `planning_consistency_audit.json`.
The existing independent controller/contact/geometry audit is applied to
all twelve executions. Package versions, source snapshots, input hashes,
logs, and prior manuscript assets are retained with the study.

## Reproduction

Run from the repository root using `.venv/bin/python`. Set
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` for each command.

To construct the study from the preceding archived identification inputs:

```bash
.venv/bin/python -m experiments.robotics.x_consistent_force_study prepare --out NEW_STUDY
.venv/bin/python -m experiments.robotics.x_consistent_force_study plan --out NEW_STUDY --material A --device cuda:0
.venv/bin/python -m experiments.robotics.x_consistent_force_study plan --out NEW_STUDY --material B --device cuda:1
.venv/bin/python -m experiments.robotics.x_consistent_force_study model_checks --out NEW_STUDY --material A --device cuda:0
.venv/bin/python -m experiments.robotics.x_consistent_force_study model_checks --out NEW_STUDY --material B --device cuda:1
.venv/bin/python -m experiments.robotics.x_consistent_force_study freeze --out NEW_STUDY
.venv/bin/python -m experiments.robotics.x_consistent_force_study evaluate --out NEW_STUDY --material A --device cuda:0
.venv/bin/python -m experiments.robotics.x_consistent_force_study evaluate --out NEW_STUDY --material B --device cuda:1
.venv/bin/python -m experiments.robotics.x_consistent_force_study audit --out NEW_STUDY
.venv/bin/python -m experiments.robotics.x_consistent_force_study report --out NEW_STUDY --render-out NEW_FIGURE
```

The A and B commands for each stage can run concurrently on separate GPUs.
Existing cached candidates are checked before reuse when resuming a search.

After the completed study has its final checksum manifest, reproduce both
searches and all later stages from that archive alone:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  -m experiments.robotics.x_consistent_force_reproduce \
  --source out/x_consistent_force_study_20260911 \
  --out NEW_REPRODUCTION --render-out NEW_REPRODUCTION_FIGURE
```

`--prepare-only` verifies and stages the inputs without launching simulations.
The reproduction imports frozen identification and target inputs, but reruns
both gap searches. GPU reductions and resulting optimizer paths need not be
bit-identical; differences are reported without changing the original data.
Neither command writes the active manuscript or figure.
