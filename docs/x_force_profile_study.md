# Force-profile X shaping

Superseded on 2026-09-11 by `docs/x_consistent_force_study.md`, which reruns
the gap searches with the final smooth motion and retains each selected
rollout's force history directly. The retimed-reference experiment and
results below are historical; its controller remains in use.

Status: frozen-plan execution and audit complete. All 12 true-material
executions are retained; all matched executions pass the force and stopping
checks. The figure and manuscript adoption are documented in
`paper/evidence/x-force-profile-study.md`.

This study replaces compression-work stopping with force feedback. Its code
is `experiments/robotics/x_force_profile_control.py` and
`experiments/robotics/x_force_profile_study.py`, with common gain selection
and final execution in `x_force_profile_gains.py`; the archive is
`out/x_force_profile_study_20260910`. Similarly named force-pulse and
force-and-hold studies are different, unadopted experiments. The opening
docstring of `x_force_profile_study.py` is inherited from an earlier study;
its protocol, implementation, and the description below define the actual
force-profile experiment.

## What is controlled

The planner supplies four time histories of desired mean outward force per
finger, together with nominal closing velocities. Pinches alternate along
y, x, y, x. Every 4 ms, the controller adds a PI correction based on measured
force error to the nominal velocity. The two fingers move symmetrically.
Actual final opening is free to change with material response. A pinch ends
at the end of its planned profile, or at the common 12 mm travel guard.
No accumulated work, shape observation, or replanning enters execution.

Nominal motion is feedforward, not a position constraint. It supplies motion
through free space and through the plastic regime, where reaction force can
decrease as compression continues. Force feedback corrects that motion.
This distinction matters: a constant-force hold can settle on different
deformation branches and proved sensitive to grid resolution in our pilots.

Let e be desired filtered force minus measured filtered force, in newtons.
Before saturation, the per-finger closing velocity is

`v = v_nominal + g * s * 0.050 * e + integral(g * s * 0.040 * e dt)`,

where `s = 1.25 / max(1.25, predicted_peak_force_in_N)` and the selected
common gain multiplier is `g = 0.5`. This single rule
applies to both plans and both actual materials. Gains have units m/(s N)
and m/(s² N), respectively. Conditional integration prevents windup during
speed, acceleration, and travel saturation. The limits are 50 mm/s per
finger, 0.5 m/s² per servo update, and 12–72 mm opening. Hard guard stops
and transitions between motion phases remain kinematic.

Force is the signed mean of the two outward contact reactions, computed
from contact impulse over each 4 ms interval. An exponential filter has a
12 ms time constant. The reference uses the corresponding model-predicted
filtered force available before each control tick. The audit compares raw
measured interval forces against raw model-predicted interval forces, so
filtering does not hide tracking error. Force feedback uses ideal simulated
measurements; sensor noise and delay are not modeled.

Reported profile durations include the nominal free approach after lowering
the fingers. Forces are time-varying references; a listed peak is not a
constant force held for the whole listed duration.

After each pinch, the fingers reopen and withdraw. A raised quarter turn
takes 1 s. The final specimen is scored 1 s after the last withdrawal.

## Planning and fixed inputs

Both material definitions, their common identification action, fitted
parameters, held-out validation, specimen geometry, rounded X target,
contact geometry, friction, and surface metric are unchanged from
`out/x_common_identification_study_20260910`.

The 90 mL specimens have density 1000 kg/m³ and gravity 9.81 m/s².
Separable contact uses friction coefficients 0.3 on the cylindrical fingers
and 0.5 on the floor. The cylinder radius is 14 mm and height is 45 mm.

The original identified-model gap searches used 32 objective evaluations
per material, bounded Nelder–Mead, and four final openings between 16 and
54 mm. Here those openings define nominal trajectories with smooth 40 ms
acceleration/deceleration ramps and at most 20 mm/s per finger. A new
simulation in each identified model predicts its four force histories.
The earlier work commands and earlier faster force traces are not reused.
There is no new shape optimization or true-material tuning at this stage.

The force histories and nominal velocities are replayed with feedback in
each identified model at 64³/50 μs and 80³/50 μs, followed by a fresh baseline
repeat. The predeclared feasibility checks require all profiles to finish,
at most 15% relative L2 raw force error per profile, and sampled material
height no greater than 45.5 mm. Repeats must retain stopping reasons and
differ by less than 0.10 mm in shape error and 0.5 mm RMS particle position.
Both profile hashes are frozen before executing either plan in either true
material. The same complete arrays are used at all numerical settings.

The initial gains pass A's checks, but B's final pinch has 21.98% force
tracking error on the finer grid. A subsequent, declared model-only sweep
first compares common PI multipliers 1, 1.5, 2, and 3, reusing the recorded
result for 1. None passes B's force check. A recorded extension then tests
0.5 and 0.25, still before any true-material execution. The earlier protocol,
source revision, and results remain in `gain_history`. Among multipliers
passing the same feasibility checks for both
identified models, selection minimizes the worse finer-grid shape error.
The selected multiplier then receives fresh baseline and repeat checks.
The multiplier affects both PI terms for every material and pinch. The
reference force and nominal velocity arrays remain unchanged. The sweep
and selection are recorded in `gain_protocol.json` and `gain_selected.json`.
The selected multiplier is 0.5. Its model finer-grid errors are 0.9455 mm
for A and 1.2946 mm for B; maximum per-profile force errors are 7.42% and
12.82%, respectively. The 0.25 multiplier also passes and gives 0.9325 and
1.3011 mm. The declared worse-material shape objective selects 0.5; these
small differences do not establish a uniquely optimal controller.

Four true-material comparisons are run at each of 64³/50 μs, 64³/25 μs,
and 80³/50 μs. Guards and tracking failures are retained. These are numerical
sensitivity checks, not a convergence study.

The target remains the 64 × 80 × 24.946587 mm rounded X with 90 mL volume.
Its design and the original common search initialization were informed by
earlier A feasibility trials. The metric is symmetric area-weighted mean
closest-triangle distance. Surfaces use reference particle volumes, 1.25 mm
voxels, Gaussian sigma 1.625 mm and isovalue 0.5. All components enter the
score, without alignment, rescaling, clipping, or subtracting a baseline.

## Reproduction

Run from `/geoelements/Stepan/mpm-engine` using `.venv/bin/python`, with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`. Each stage accepts `--out`.

1. `-m experiments.robotics.x_force_profile_study prepare --out NEW_DIRECTORY`
   stages the frozen identification, target, original plans, sources, and
   package/input hashes. It requires the original common-study archive.
2. Run `reference --material A --device cuda:0` and the corresponding B
   command on cuda:1, then `model_checks` for each material.
3. Run `-m experiments.robotics.x_force_profile_gains prepare --out DIRECTORY`,
   then its `sweep` stage for each material, `select`, `model_checks` for
   each material, and `freeze`. Finally run this module's `evaluate` stage
   for each material on separate GPUs. Do not use the earlier base-module
   freeze/evaluate stages for the gain-selected experiment.
4. Run `-m experiments.robotics.x_force_profile_audit audit --out DIRECTORY`,
   then its `diagnostics` stage.
5. Render with `-m experiments.robotics.x_force_profile_report --out DIRECTORY
   --render-out NEW_RENDER_DIRECTORY`.

After the archive is frozen, `x_force_profile_reproduce.py` provides a
single frozen-plan replay command. It verifies the archive and stages its
inputs, reruns all 12 true executions, audits them, and regenerates the
figure. It imports identification and planning records as fixed inputs;
it does not claim to rerun their estimators or optimizers. `--prepare-only`
checks staging without launching simulations. GPU reductions need not be
bit-identical, so the replay reports every error difference and stopping
comparison rather than replacing the original records.

## Hardware scope

This is simulation with supplied-state identification, ideal finger-force
sensing, and kinematic cylindrical tools. The Panda hand is an illustration
with proposed adapters, checked for sampled IK, joint range, and speed.
It is not a hardware experiment or a full-arm dynamics/collision check.

Stock Franka Hand operation at these low forces and with these time-varying
commands is not established. Its API accepts a grasping-force command, but
GripperState does not expose measured finger force. Manual R50010/1.2 lists
30–70 N adjustable continuous grasping force. Added sensors alone do not
establish low-force actuation. Net wrist force cannot generally recover
opposing finger forces that cancel. Hardware transfer needs verified force
sensing, low-force actuation, and a suitable feedback interface.

Sources: [gripper API](https://frankarobotics.github.io/libfranka/0.15.0/classfranka_1_1Gripper.html),
[gripper state](https://frankarobotics.github.io/libfranka/0.15.0/structfranka_1_1GripperState.html),
[Hand manual](https://franka.de/hubfs/Product%20Manual%20Franka%20Hand_R50010_1.2_EN.pdf).

## Earlier strategies

The time-pulse, contact-pulse, established-force hold, and normalized hold
archives preserve all attempted candidates, stopped searches, and model
checks. None contains a new true-material shaping execution or a paper
adoption. `docs/x_force_hold_study.md` and each archive's status records
document their limits. Good shape alone was insufficient: force tracking
and finer-grid sensitivity determined whether to proceed.

## Audited execution results

| Setting | A matched | A swapped | B matched | B swapped |
|---|---:|---:|---:|---:|
| baseline | 1.025124 mm | 1.504295 mm | 1.313165 mm | 2.905625 mm |
| half_dt | 1.003272 mm | 1.541393 mm | 1.305704 mm | 2.905625 mm |
| grid80 | 0.946754 mm | 1.516058 mm | 1.294684 mm | 2.930380 mm |

The force-profile and nominal-motion plan are swapped together. Swapped
executions also incur force-tracking errors; this comparison does not
isolate deformation under perfectly tracked identical forces. A under B's
plan reaches the minimum-opening guard on pinches 1, 2, and 4 in all three
settings. These failures remain visible in the figure and force-history
diagnostics. The 80³ grid was used during model-only controller selection;
the reported grid comparison is a sensitivity check, not an independent
held-out numerical validation.

A full frozen-plan execution replay can be launched from the repository:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  -m experiments.robotics.x_force_profile_reproduce \
  --source out/x_force_profile_study_20260910 \
  --out out/x_force_profile_replay \
  --render-out out/x_force_profile_replay_figure
```

Use fresh output paths. Add `--prepare-only` to verify the frozen archive
and stage inputs without rerunning simulations. The final staging check is
recorded with the publication archive.
