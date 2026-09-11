# Force-and-hold X shaping

Status: unadopted model-only development, frozen before true-material execution.
The current alternative is documented in `docs/x_force_profile_study.md`.
B completed 24 initialization and 24 optimization evaluations; final B
selection was not run. An early candidate had 2.60 mm finer-grid shape error
despite good force tracking. Procedures below describe this candidate study,
including the intended selection/execution stages that were not completed.

The current simulation source is `experiments/robotics/x_force_hold_damped_control.py`,
with `x_force_hold_damped_study.py`, `x_force_hold_damped_plan.py`, and
`x_force_hold_damped_select.py`. Its archive is
`out/x_force_hold_damped_study_20260910_v2`. The figure and audit entrypoints
are `x_force_hold_report.py` and `x_force_hold_audit.py`. Earlier similarly
named files implement different, unadopted controller variants; see the
history below. Source snapshots and hashes identify every revision.

## Controlled action

Each of four centered pinches commands a mean outward force per finger and
a dwell duration. Pinches alternate y, x, y, x. Finger velocities follow
measured contact force; final opening, work and specimen shape are not
execution stopping commands. There is no online shape feedback or replanning.

The fingers approach from 72 mm opening at 10 mm/s per finger until three
successive filtered force samples reach 0.05 N. The commanded force rises
smoothly over 0.2 s. The dwell begins after three consecutive samples lie
within 10% of the command, after the ramp. The controller maintains the
command for the planned dwell, then ramps down over 0.12 s before opening
and withdrawing. A 3 s force-attainment timeout and 12 mm minimum opening
terminate an unsuccessful pinch. The sequence continues after withdrawal;
guard and timeout events are recorded rather than hidden.

Feedback uses the signed mean of the two outward contact reactions, measured
as grid impulse divided by the 4 ms control interval, with a 12 ms EMA.
Base PI gains are Kp=0.050 m/(s N) and Ki=0.040 m/(s² N). Both are multiplied
by `1.25 / max(1.25, F_command_in_N)`. This common command-dependent rule
reduces the response to absolute force error at larger loads. It does not
use actual material identity. The controller uses conditional integration,
a 50 mm/s per-finger velocity bound and 0.5 m/s² acceleration bound during
servo updates. Hard travel stops and phase transitions are kinematic;
actuator dynamics are not simulated. Inherited `start_gap_m` and
`pre_pulse_wait_s` entries in CONTROL are unused; the action above is what
the code executes.

There are eight planning variables: four forces (0.05–14 N) and four dwells
(0.08–3 s, quantized to 4 ms). Raised quarter turns take 1 s. Shape is scored
1 s after final withdrawal. `durations_s` stores the commanded dwell, not
total action duration. The recorded completed-dwell counter excludes a final
interrupted guard tick; the force history includes that tick, and the audit
accounts for this convention.

## Fixed inputs and planning

The common cylindrical identification action, held-out validation, fitted
parameters, true material laws, initial specimen, target X, contact geometry,
friction and scoring are unchanged from the frozen
`out/x_common_identification_study_20260910` archive. No force observations
are added to identification. Planning receives identified-model parameters
and predictions; true-material executions and their shape errors are excluded.

B's normalized-controller search initializes forces from the 20 ms averaged
peaks in its previous identified-model position plan. At the minimum dwell,
force is reduced by 15%, at most three times, if compression already passes
that model's reference opening by more than 0.5 mm. Endpoint trials and at
most seven bisections then initialize dwell durations. Reference openings
are initialization objectives, never execution commands. A 24-call bounded
Nelder–Mead search refines all eight force/dwell controls on a 64³/100 μs grid.

A's commands are carried from the completed model-only two-grid refinement
in `out/x_force_hold_study_20260910/robust/A`. Selection is restricted to
commands no greater than 1.25 N, where the normalized controller's gain
multiplier is exactly one. These commands are still replayed and checked in
the current implementation. The older A search uses nine paired force/dwell
seeds and four local neighbors, assessed at 64³/100 μs and 80³/50 μs. It is
based on the shared identified model and contains no true-material results.

For each material, three distinct feasible candidates receive fresh model
checks at 64³/50 μs and 80³/50 μs. Selection minimizes worst model surface
error across planning and both checks, requiring completed holds, per-hold
raw mean-finger force RMS error no greater than 15% of the command, and
sampled height no greater than 45.5 mm in each setting. A fresh baseline
repeat must remain feasible, retain stopping reasons, and differ by less
than 0.10 mm in surface error and 0.5 mm RMS final particle position.

Both selected plans are frozen before true-material cross-execution at
64³/50 μs, 64³/25 μs and 80³/50 μs. These are numerical sensitivity checks,
not evidence of convergence. Full surfaces and every connected component
enter the unchanged symmetric area-weighted 3D metric, without registration,
rescaling or clipping.

## Development history

- `out/x_force_time_study_20260910`: fixed-start, total-duration force pulses;
  18 one-pinch model trials, not adopted.
- `out/x_force_contact_study_20260910`: contact-triggered total-duration
  pulses; 18 pilots, initialization and 22 A / 27 B local-search calls.
  Stopped because short pulses could yield good shapes while missing forces.
- `out/x_force_hold_study_20260910`: force attainment followed by dwell, with
  unnormalized PI gains. It includes 18 pilots, initialization, and 14 A /
  15 B local-search calls before numerical checks redirected the search.
  The A two-grid refinement completes 13 candidates. B stops after four
  complete paired candidates and an additional final-hold diagnostic; the
  longer hold retains large force oscillations. No true executions or paper
  adoption occur in this archive.
- `out/x_force_hold_damped_study_20260910`: two model replays with normalized
  gains, using unchanged initial B commands. All holds track within roughly
  3–6%, but shape error is about 2.6 mm because the old timings under-squeeze.
  This pilot is frozen. Its admissible dwell range was 0.08–1.5 s.
- The current `..._v2` archive expands the dwell domain to 3 s and replans
  with the normalized controller. Behavior for earlier admissible commands
  is unchanged; the original pilot's sources remain in its snapshot.

Interrupted searches are explicitly marked. Their configured evaluation
budgets are not claimed as completed. Evaluation ledgers distinguish partial
prefix trials, full rollouts, failed candidates and cached reuses. No control
or planning runtime comparison is claimed.

## Hardware scope

This is simulation with ideal finger-force measurements and supplied-state
identification. Panda geometry is a kinematic illustration with proposed
cylindrical adapters. Stock Franka Hand operation at these low forces is not
established. The gripper API accepts a grasping-force command, but
GripperState does not expose measured finger force. Manual R50010/1.2 lists
30–70 N adjustable continuous grasping force. Actual actuator range, force
convention, sensing and command capability require verification. Adding
sensors alone does not establish low-force actuation. Net wrist force does
not generally resolve cancelling opposing squeeze forces.

Sources: [API](https://frankarobotics.github.io/libfranka/0.15.0/classfranka_1_1Gripper.html),
[state](https://frankarobotics.github.io/libfranka/0.15.0/structfranka_1_1GripperState.html),
[manual](https://franka.de/hubfs/Product%20Manual%20Franka%20Hand_R50010_1.2_EN.pdf).

## Reproduction and paper update

Use each archive's exact source snapshots, input hashes and packages. The
current `x_force_hold_damped_study prepare` command requires a fresh output
directory. `x_force_hold_damped_plan plan` runs B planning; the separate
selection stage imports the completed model-only A candidates and checks
both materials. `x_force_hold_damped_plan evaluate` requires both selected
plan hashes to be frozen before running true materials. All commands accept
an explicit `--out` directory.

Complete the audit and render into a fresh directory before copying PNG/PDF
to the active paper figure. Finish all TeX writes before invoking
`paper/icra2027/build.sh`, then inspect the affected PDF pages. Simulation
and planning commands never write the active figure or TeX. The final
frozen-plan replay command and adoption hashes will be recorded when the
experiment passes its checks.
