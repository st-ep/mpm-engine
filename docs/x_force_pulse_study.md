# Timed force-pulse shaping study

Status: not adopted. The contact-triggered search was stopped after 22 A and
27 B optimization calls because the final pinches could have poor force tracking
despite low model shape error. No true-material cross-executions or paper update
were made from these pilots. See `force_tracking_review.json` in the archive.
The subsequent force-established dwell strategy is in `docs/x_force_hold_study.md`.
The selection and validation stages below describe the frozen intended protocol;
they were not executed for this unadopted strategy.

The current candidate uses `experiments/robotics/x_force_contact_control.py`
and `experiments/robotics/x_force_contact_study.py`, followed by the separate
planning stage `experiments/robotics/x_force_pulse_plan.py`. Its archive is
`out/x_force_contact_study_20260910`. The initial fixed-start pilot uses
`x_force_time_control.py` and `x_force_time_study.py`, with the distinct archive
`out/x_force_time_study_20260910`. Neither is the historical 2026-09-09
fixed-duration force-hold study. Source and protocol snapshots distinguish them.

## Fixed scientific inputs

The shared 32 mm identification probe, deeper/faster validation, fitted A/B
parameters, true material laws, initial specimen, rounded X target, geometry,
contact and scoring are inherited unchanged from
`out/x_common_identification_study_20260910`. That frozen archive remains intact.
Planning reads identified-model parameters and predictions. It does not use
true-material executions or their shape errors. Identification and validation
are not rerun, and no force observations are added to identification.

## Controls

Each of four centered pinches, alternating y, x, y, x, commands a mean outward
force per finger and a total pulse duration. A raised-cosine rise occupies the
first 20% of the duration; the force is constant for 60%, then follows a
raised-cosine fall for 20%. The controller uses signed per-finger reaction
forces, a 12 ms exponential filter, common PI gains, a 4 ms control period,
and 50 mm/s per-finger speed limits. The existing 0.5 m/s² acceleration limit
applies to servo updates; the hard travel stop and transitions between motion
phases are kinematic. Actuator dynamics are not simulated.

The contact-triggered variant approaches at 10 mm/s until three consecutive
filtered samples reach 0.05 N, then starts the pulse clock. A common 12 mm
opening guard ends an unsuccessful pulse early. Matching plans must avoid it.
The ending gap is an outcome, not an execution command. There is no work
budget, shape feedback, or online replanning. Fingers open and withdraw between
pinches; raised quarter turns take one second. Shape is measured one second
after final withdrawal.

## Planning and reproducibility

Both materials receive a nine-trial, one-pinch force/duration pilot. Force
amplitudes are fractions of the 20 ms averaged peak force in that model's
previously planned position trajectory. These are model predictions, not
true-material force measurements. The initial pilot uses fractions
0.95/1.05/1.20 and durations 0.4/0.8/1.2 s. The contact-triggered pilot uses
1.00/1.05/1.15 and 0.4/0.7/1.0 s. All trials, including guard stops, are saved.

The planning stage initializes force amplitudes using the smallest tested
factor that reached the model reference opening, or the closest feasible pilot
if none reached it. It then uses endpoint trials and at most seven bisections
per pinch to initialize durations near the existing model-plan openings.
These openings are initialization objectives only. A bounded Nelder-Mead
search subsequently varies all eight force/time controls for 48 objective
calls against the complete X target. Commands are quantized to 4 ms; force
bounds are 0.05–14 N and pulse durations 0.2–2.4 s. Selection uses the lowest
feasible identified-model 3D surface error among complete initialization and
search trials. Failed or guarded trials are retained.

Planning uses a 64³ grid and 100 μs timestep. The three strongest distinct commands receive identified-model checks at
64³/50 μs and 80³/50 μs before any true execution. Selection minimizes the
worst model error across those settings and the planning setting, requiring
no guard activation. A fresh baseline repeat must have the same stopping
reasons, less than 0.10 mm difference in surface error, and less than 0.5 mm
RMS difference in final particle positions. This selection protocol was frozen
before the checks or true-material runs.

True execution uses 64³/50 μs,
with 64³/25 μs and 80³/50 μs numerical sensitivity checks. Commands and physical
geometry remain fixed across settings. The previous 32-evaluation position
search, both pilot protocols, duration initialization and new optimization are
additional planning costs; no controlled runtime claim is made.

From the repository root, use `.venv` and fresh output directories. Run
`x_force_contact_study prepare`, then `pilot --material A --device cuda:0`
and `pilot --material B --device cuda:1`. Run `x_force_pulse_plan plan` for
each material, `x_force_pulse_select prepare` once, then `select` for each material.
Run `evaluate` for each true material only after both validated plans are frozen. Every command takes the same explicit `--out` directory.
`x_force_pulse_audit` checks the stored execution histories and produces force
tracking diagnostics. The pilot and planning stages retain their own source
and protocol hashes; archived source revisions must match for reproduction.

## Hardware scope

This is contact simulation with ideal simulated finger-force measurements.
The Panda geometry is a kinematic illustration. It does not establish stock
Franka Hand force tracking, actuator dynamics, full-arm collision avoidance or
camera-based state recovery. The stock gripper API accepts a grasping-force
command but does not expose measured finger force in GripperState. The Franka
Hand manual R50010/1.2 lists 30–70 N adjustable continuous grasping force; the
force convention and capabilities of the actual gripper and adapters must be
verified before a hardware implementation at this study's lower forces.

Sources: https://frankarobotics.github.io/libfranka/0.15.0/classfranka_1_1Gripper.html,
https://frankarobotics.github.io/libfranka/0.15.0/structfranka_1_1GripperState.html,
https://franka.de/hubfs/Product%20Manual%20Franka%20Hand_R50010_1.2_EN.pdf.
