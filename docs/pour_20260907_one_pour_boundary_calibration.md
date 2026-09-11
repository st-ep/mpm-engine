# One-pour boundary calibration, 7 September 2026

This is an explicitly authorized change from the earlier endpoint-as-check-only
protocol. The user chose to retain the original PDF two-pass weak spout-edge
estimate, **3.4392377844275503 Pa s**, and use the **same 60-degree identification
pour's 159 mL receiver measurement** to calibrate one effective simulator boundary
parameter. Initial liquid volume is 300 mL. The viscosity is not optimized by the
simulator. The six later outcomes and original 45/50-degree endpoint measurements
are excluded from parameter fitting and command selection.

The defensible claim is one pour supplies weak-form viscosity identification
**and effective simulator boundary calibration**. Agreement with 159 mL is an
in-sample calibration result, not evidence of transfer to another angle. No
claim of independently measured microscopic liquid–cup slip is made.

## Current result

The active model is the original separable/Coulomb source contact with effective
numerical coefficient **0.117**. It gives **158.9471 mL** at 160 cubed and
**161.3902 mL** at 192 cubed, with viscosity fixed at **3.4392377844275503 Pa s**.
The **2.4430 mL** difference passes the predeclared 3 mL grid-pair check. The
coefficient was fitted using only the same 60-degree pour's 159 mL endpoint.
It is not a measurement of microscopic liquid-wall friction.

With the contact coefficient frozen, the MPM-only angle search completed:
**48.37 degrees gives 60.8937 mL**, passing the 60 +/- 1 mL simulation target.
The command uses a 4.837 s inclination duration, 2 s hold, 2 s return, and 300 mL
initial fill. The selected replay has 0.0340 mL outside both cups and 0.0471 mL
tail variation. It uses 229,280 particles at 160 cubed. This new command has not
been validated on the robot or checked at the finer grid; the 160/192 check was
performed on the 60-degree calibration replay. No full target table or historical
handoff has been released.

The prior 60 mL command in `out/pour_weakform_recovery/philip/angle_table.csv`
was 47.20 degrees; the new prediction is 1.17 degrees higher. That comparison is
between commands only and is not a fit to measured outcomes. In particular, this
work does not establish a 7 mL real-robot accuracy claim.

Accepted prediction: `out/pour_navier_calibration/prediction_60ml_coulomb.json`.
Search-point plot: `out/pour_navier_calibration/coulomb_60ml_search.png`.
Active protocol: `out/pour_navier_calibration/coulomb_calibration_protocol.json`.
Active driver: `experiments/pour/pour_coulomb_calibrate.py`.

The sections below retain the diagnostic history, including the superseded Navier
route. Those exploratory settings are not the selected boundary model.

## Initial Navier protocol (superseded)

- Fix viscosity, density, measured geometry, grasp, and exact recorded joints.
- Fit only the source wall slip length, initially bounded to 0.25–4 mm.
- Start with 229,280 particles at 160 cubed in a 0.35 m domain.
- Match the calibration endpoint to within 1 mL, then compare the same parameter
  at 192 cubed (396,197 particles). Do not refit separately at each resolution.
- Require the endpoint difference to be at most 3 mL before moving to the
  60 mL target-angle search. This is a limited numerical check, not proof of
  convergence. No automatic finer-grid simulations are authorized by the driver.
- Search new angles using forward MPM outputs only. Freeze the calibrated
  boundary parameter before this search. A newly simulated 60 mL command is still
  a prediction requiring prospective robot validation.

The executable protocol is `out/pour_navier_calibration/calibration_protocol.json`.
`experiments/pour/pour_navier_reference.py` performs the forward simulations and
does not read measured endpoints. `experiments/pour/pour_navier_calibrate.py`
performs the bounded one-parameter calibration and subsequent simulation-only
angle search. Per-run provenance includes hashes of code, geometry, observations,
and the reference action/state records. No historical handoff is overwritten.

## Boundary development and rejected attempts

The original cup contact is separable Coulomb contact. These experiments install
an opt-in source-only Navier traction callback. The receiver contact remains
unchanged; no core solver files are changed.

A quadratic Robin velocity extrapolation improved analytical film-profile checks
but failed the first full cup simulation during initial settling: liquid escaped
the computational crop before pouring. That failed run is retained under
`out/pour_navier_calibration/replays/recorded60_b1.000000_n160_phase0_dt1/`.
It supplies no usable endpoint and was rejected.

The Navier implementation tested here uses implicit dissipative wall traction with
coefficient eta/b. A correction for partial wetting divides nodal mass occupancy
by the analytical quadratic-B-spline half-space occupancy before assigning wall
area. This corrects wall-area overcount at narrow-film edges; its derivation uses
analytical quadrature, not robot outcomes. The correction still assumes locally
planar walls and uniform liquid density and is a numerical approximation.

Tests of that implementation verify non-increasing relative grid kinetic energy and equal and
opposite wall impulse and torque, including moving and rotating walls. Analytical
film comparisons expose remaining spatial and temporal errors. In particular,
halving the film time step still changes the 1 mm-slip result by about 2.5%; this
does not satisfy the earlier strict 1% film time-step criterion. The full pour
therefore remains exploratory pending the explicit volume checks above.

Results and rejected cases are retained in `out/pour_navier_calibration/`.
The calibration status is updated atomically in `status.json`. A match is not
reported as a completed calibration until its specified volume check passes.

## Parameter-range review after the user's additional constraint

The user explicitly requires reasonable parameters with defensible bounds. The
0.25–4 mm interval above was an exploratory bracket, not a measured uncertainty
interval or a literature-supported physical range. Automatic fitting is paused;
the already-running 4 mm sensitivity probe may finish but cannot select a model.
`parameter_range_review.json` disables subsequent automatic fitting.

Physical and numerical slip must be distinguished. Experiments on glycerol and
Pyrex found no slip on hydrophilic surfaces and slip of order 100 nm on hydrophobic
surfaces ([Cottin-Bizonne et al.](https://arxiv.org/abs/cond-mat/0210154)). Our cup
model describes a plastic measuring cup; its surface material, treatment, and
wettability have not been established here. The Pyrex experiment is therefore
context, not a measured bound for this cup. It supplies no justification for
millimetre-scale physical slip.

Effective viscous wall forces can implement prescribed hydrodynamic boundary
conditions in particle methods ([Smiatek et al.](https://arxiv.org/abs/0712.3592)).
That supports testing a numerical boundary treatment; it does not validate our
parameter values or make endpoint fitting a substitute for numerical verification.
In particular, reproducing an analytical film with a prescribed 4 mm slip length
verifies implementation of that condition, not its suitability for the real cup.

The original PDF closure explicitly assumes zero wall velocity (equations in
the lubrication-closure section). Adding substantial *physical* slip only to MPM
would change the forward constitutive assumptions relative to identification.
For a planar film, Navier slip increases the flux by a factor 1+3b/h relative to
no slip at fixed viscosity. This must not be hidden by calling b "effective".

Before any continuation of endpoint calibration, a numerical correction must
be supported independently by film/geometry benchmarks and remain consistent
with the no-slip closure, or a changed physical model must be explicitly derived.
Any proposed geometric change must also be checked for its effect on the weak
identification; the same geometry enters that estimate. No geometry, viscosity,
or boundary parameter has been changed to fit the six validation outcomes.

## Measured-rim geometry probe

The completed fixed-geometry trials at 160 cubed produced 130.7698 mL for b=1 mm
and 146.3952 mL for b=4 mm. Neither matches 159 mL. The 4 mm film at dx=2 mm
has 2.262 times the analytical no-slip mean flow, so this is not a small correction
to the PDF's boundary assumption. No larger-slip trials were launched.

The existing collision padding widens the nominal 5.01 mm rim to 9.3225 mm at
160 cubed. A separate causal test restores the measured rim: outward padding
transitions smoothly from its original value at the existing spout-region start
to zero at the existing rim-blend start. The transition introduces no dimension
fitted to receiver volume. It leaves the entire measured solid and cavity intact;
receiver geometry is unchanged. The pre-existing rim-blend depth is a model
assumption, not a newly claimed caliper measurement.

`experiments/pour/pour_measured_lip_probe.py` runs this variant with the baseline
1 mm parameter and 3.4392377844275503 Pa s retained solely for controlled
comparison. It is not an accepted calibration or an angle prediction. The user
explicitly confirmed continuation of the geometry plan. Results and the
supplementary geometry provenance are in
`out/pour_navier_calibration/measured_lip_probe/`.

Independent film tests now also use the full pour's time-step restriction and
cell sizes. They examine a numerical rule b=C*dx, which tends to zero under
refinement, rather than claiming fixed microscopic slip. For a 10.9375 mm film at
dx=2.1875 mm, C=0.5 gives 0.9960 times analytical no-slip flow, but still 8.48%
relative velocity-profile error. Agreement of one mean flux does not establish
the full wall condition; phase and refinement diagnostics are retained separately
under `out/pour_navier_calibration/numerical_wall_range/`. No C value has been
selected against a real pouring endpoint.

The additional independent C=0.5 checks at film depths 8.75 and 13.125 mm and at
a 45-degree planar slope have mean no-slip flux errors from -3.5% to +3.1%.
These are analytical synthetic films, not the original 45-degree robot data.
The full-pour boundary is still not accepted based on these limited tests.

## Original contact comparison

The user recalled the earlier 174.8691 mL prediction and asked whether the original
separable/Coulomb wall may have been more appropriate. That earlier run used
eta=3.397435009739053 Pa s, a 384-cubed grid in the larger domain, and 396,197
particles. It cannot be compared directly with the new 160-cubed 3.4392 Pa s runs
to isolate a wall effect.

A matched original-contact probe is queued after the geometry test. It keeps
eta=3.4392377844275503 Pa s, 160 cubed, the same compact domain and particle count,
the original geometry padding, and the original Coulomb coefficient 0.05. No
coefficient is fitted. `experiments/pour/pour_original_wall_probe.py` retains a
comparison with the b=1 mm traction baseline and the measured-rim geometry probe.
The old forward-runner source was saved by SHA256 under `code_snapshots/` before
adding the explicit original-contact option.

The completed measured-rim probe predicts **126.9151 mL**, versus **130.7698 mL**
with the original collision padding at the same b=1 mm. Its change is **-3.8547
mL**, with 0.3271 mL outside both cups and 0.1871 mL receiver variation over the
final 0.5 s. It does not repair the underpour. The result is retained as evidence
against rim thickening being the primary explanation for the shortfall in this
boundary configuration; it does not validate either wall implementation.

The matched original-wall run produced **170.1657 mL** at eta=3.4392377844275503
Pa s and 160 cubed, versus 130.7698 mL with the 1 mm traction wall. This isolates a
large wall-treatment effect. The original wall remains the practical reference;
neither its closer endpoint nor the traction wall's better planar-film behavior
establishes reliable transfer to new angles.

## Bounded calibration of the original contact approximation

Following the user's suggestion to reconsider the original wall, a sensitivity
probe changes only its existing source contact coefficient from 0.05 to 0.20.
The numerical interval is declared before that probe's endpoint is available.
It limits the tangential collision impulse to 5–20% of normal impulse. This is
a deliberately limited engine-parameter exploration, **not** a measured or
literature-supported liquid-wall material-property interval.

If the two simulated outcomes bracket 159 mL, the bounded driver may calibrate
this one effective contact parameter against the same 60-degree pour's endpoint.
Viscosity stays fixed at the original weak estimate. The driver then requires
the same 3 mL 160/192 check before predicting a 60 mL command with MPM. It cannot
extend the coefficient interval or launch a finer grid automatically. The exact
protocol is `coulomb_calibration_protocol.json`; the driver is
`experiments/pour/pour_coulomb_calibrate.py`.

The interpretation in the paper must remain **one-pour weak-form effective
viscosity identification plus numerical contact calibration**. The PDF states
that the effective viscosity absorbs drawdown and transverse-shear approximations;
it must not silently be promoted to an independently measured bulk viscosity.
The original contact's deviation from no-slip remains a model limitation.
Endpoint calibration does not demonstrate new-angle accuracy; the six previously
seen outcomes are not calibration inputs and are not reused as blind validation.

The completed coefficient-0.20 replay gives **149.3510 mL**, with 0.1806 mL outside
both cups and 0.0340 mL tail variation. Together with 170.1657 mL at coefficient
0.05, it brackets the same-pour 159 mL target. The first bounded secant proposal is
coefficient **0.130**. This is being verified by a new full replay; interpolation
between parameter probes does not replace that replay or supply target angles.

The coefficient-0.130 replay completed at **156.7686 mL**, with zero outside
particles and zero tail variation. The next bounded secant trial is coefficient
**0.117**. The viscosity, geometry, motion, initial volume, and grid remain fixed.

The coefficient-0.117 replay completed at **158.9471 mL**, an in-sample error of
**-0.0529 mL** relative to the 159 mL measurement. There are 0.0183 mL outside both
cups and 0.0497 mL variation over the final 0.5 s. This is a successful coarse-grid
endpoint calibration at fixed weak viscosity. The same coefficient is now being
checked at 192 cubed; no new-angle command is released while that check is pending.

A forward-motion audit on the same recording reproduces its 60-degree joint
trajectory to 2e-15 rad and checks continuous phase joins for the planned angle
range. The recorded acknowledgement overhead (0.3287 s), dwell (2.0039 s), and
return acknowledgement duration (2.3962 s) are retained. The source cup's fitted
mounting offset remains fixed: a 60-degree command corresponds to 61.9119 degrees
of cup tilt at acknowledgement; a planned 45-degree command corresponds to
46.9185 degrees. Commands must therefore retain the robot-angle convention rather
than silently treating it as absolute cup tilt. This is a construction check,
not evidence that the robot will execute an unseen planned trajectory exactly.
The audit reads no additional recording or measured endpoint and is retained as
`out/pour_navier_calibration/planned_motion_audit.json`.

The unchanged-coefficient 192-cubed replay completed at **161.3902 mL** with
396,197 particles, 0.0015 mL outside both cups, and 0.0326 mL tail variation. Its
**2.4430 mL** difference from the 160-cubed match passes the predeclared 3 mL
comparison. This is a successful check on this grid pair, not a proof of continuum
convergence or a bound on real-robot error. The measured solid remains fixed;
the existing grid-dependent collision padding is part of the discretization.

The driver froze eta=3.4392377844275503 Pa s and contact coefficient 0.117 at the
calibrated 160-cubed production resolution, then started the planned 45-degree
MPM replay to bracket a 60 mL command. Angle selection uses forward MPM outputs
only. The numerical check and selection are recorded in `coulomb_grid_check.json`
and `selected_coulomb_boundary.json` under `out/pour_navier_calibration/`.

With these settings frozen, the planned 45-degree replay produced **39.6171 mL**
and the first candidate, 47.56 degrees, produced **54.6288 mL**. Both passed the
capture and settled-tail checks. The first candidate is therefore not accepted
as a 60 mL command. The safeguarded bracket search next tests 49.43 degrees;
these are simulated search points, not a fit to measured outcomes at other angles.

The full angle search completed with the following MPM-only outcomes. Intermediate
search points remain visible and are not relabelled as accepted commands.

| Command angle (degrees) | Simulated receiver (mL) | Role |
| --- | --- | --- |
| 45.00 | 39.6171 | Initial lower bracket |
| 47.56 | 54.6288 | First candidate, below tolerance |
| 49.43 | 70.8261 | Upper bracket refinement |
| 48.18 | 58.9088 | Candidate, just below tolerance |
| 48.37 | 60.8937 | Accepted simulated 60 mL command |

The selected replay is stored under
`out/pour_navier_calibration/replays/original-separable/planned48.37_bNA_n160_phase0_dt1_mu0.117000/`.
The detached driver finished normally after its four allowed candidate trials.
No viscosity or contact parameter changed during the angle search. The initial
60-degree match is an in-sample calibration; the new-angle result is an unvalidated
forward prediction. Prospective robot validation is needed before claiming
transfer accuracy. The six previously observed outcomes did not enter this search.

## Subsequent retrospective diagnostic requested by the user

After freezing and completing the 48.37-degree prediction, the user asked how far
off it might be based on the earlier real experiments. The actual tested commands
are retained in `out/pour_validation_20260906/frozen_commands.csv`: 47.06 degrees
gave approximately 63 mL and 50.14 degrees gave approximately 80 mL. The previously
mentioned 47.20 degrees belongs to an earlier simulation table, not that real test.

Linear interpolation between these two measured points at the frozen 48.37-degree
command gives **70.2305 mL**, or **+10.2305 mL** relative to the 60 mL target. A
shape-preserving interpolation of the six previous points gives **68.2320 mL**,
or **+8.2320 mL**, as an interpolation-sensitivity comparison. This spread is not
a confidence interval, and measurement/repeatability uncertainty is unavailable.
The diagnostic assumes unchanged fill, liquid, temperature, mounting, and timing.

Consequently, the numerical successes above do not demonstrate the desired 7 mL
real-robot accuracy. The lower-angle discrepancy likely remains. No viscosity,
contact coefficient, geometry, or command was changed using this diagnostic.
It explicitly uses earlier experimental outcomes for the user's requested
retrospective estimate and must not be reported as an independent one-pour
prediction. Details: `out/pour_navier_calibration/posthoc_60ml_expected_validation.json`.

## Overnight target search requested by the user

The user authorized unattended overnight simulations to determine the remaining
target commands. The last requested set is **60, 80, 100, 120, 140, 160 mL**. The
model remains frozen at the original PDF weak viscosity and source contact 0.117;
the retrospective measured-volume estimate above is not read by the new driver.

`experiments/pour/pour_coulomb_overnight.py` searches production-grid MPM outputs,
verifies each selected command to within 1 mL, fills a one-degree simulation curve
from 48 through 61 degrees, and checks every command at 192 cubed and at a half-cell
grid-origin displacement. The 60, 120, and 160 mL commands also receive half-time-step
checks. Declared comparison limits are 3 mL for resolution and origin, and 1 mL for
time step. These are numerical screens, not bounds on real-robot accuracy. The
driver does not refit the contact or viscosity or change commands based on these
checks. Failed checks remain in the report and prevent a ready handoff ZIP.

The explicit angle-search bound is extended to 62 degrees to bracket 160 mL. This
changes a planning bound only; the original runner's numerical implementation is
preserved, with its prior exact source archived by SHA256. A dry motion audit at
48, 61, and 62 degrees verified source-mesh crop clearance above 47 mm and joint-limit
margin above 0.90 rad, with the measured mounting fixed. The actual robot execution
of extrapolated commands is still a prediction. No geometric quantity was fitted.

The detached service is `pour-coulomb-overnight-20260907.service`; controller log is
`/tmp/pour_coulomb_overnight_20260907.log`. The user service manager has lingering
enabled, so laptop disconnection does not stop the queue. Only GPU1 is used; GPU0
remains occupied by others. The driver has an 8-hour planning budget and a 60-new-run
cap. Frozen source/data hashes and the protocol hash are checked between runs.

Outputs are under `out/pour_navier_calibration/overnight_targets/`: `status.json`,
`state.json`, `predictions.json`, `SUMMARY.md`, and the evolving `angle_table.csv`
and `angle_table.png`. A completed passing `philip_handoff.zip` contains only the
three-column table, graph, and short instructions. Historical ZIPs stay separate.
The unattended workflow passed 18 relevant tests and a frozen-input preflight
before launch. Case failures are retained; no unfavorable measurement is hidden
and no prior validation outcome enters command selection.
# Pilot release after the time-step review

The user subsequently requested completing the table for Philip using the
existing calibrated simulation. The separate package
`out/pour_navier_calibration/philip_pilot_20260907/philip_pilot_handoff.zip`
contains only the unchanged CSV, graph and brief pilot instructions. Each
selected command's full MPM result and provenance were verified before copying.
There is no parameter refit, outcome-based correction, or additional simulation.

The pilot explicitly states that numerical sensitivity remains unresolved and
7–10 mL robot accuracy is unestablished. A fixed numerical model calibrated on
one pour can be prospectively tested at other angles; the calibration match
alone does not validate that transfer. Failed numerical checks and original
acceptance criteria remain recorded unchanged. The paper description must retain
both weak-form viscosity identification and same-pour numerical contact
calibration, together with the numerical limitations. Package provenance is saved
outside the ZIP. No external message or upload was sent to Philip.

# Completed follow-up on 2026-09-07: recorded-60 time-step diagnostic

The half- and quarter-time-step replays completed successfully. At fixed
viscosity3.4392377844275503, source contact0.117, grid160 and the exact recorded
motion, final receiver volumes are158.947,152.189,141.566mL for time-step
factors1,0.5,0.25. Successive changes are-6.758 and-10.623mL. They grow rather
than contract, so this sequence does not establish temporal convergence.
The calibrated159mL match is strongly time-step dependent. These results do
not establish the smaller-step prediction as physical ground truth or identify
the underlying numerical cause. Fresh settling is recomputed at each time step.

Post-run provenance and identical-configuration checks passed. Every endpoint
has less than0.02mL outside the cups and less than0.05mL tail variation. Neither
parameter fitting nor validation measurements entered these two replays.
The separate review and graph are in
`out/pour_navier_calibration/recorded60_timestep_followup/REVIEW.md` and
`comparison.png`. The controller is inactive; no further simulations or new
handoff ZIP were issued. An implementation audit is recommended before any
further full replay, recalibration or angle sweep.

## Historical launch record

The user authorized the two recommended replays at half and quarter time steps.
The experiment is frozen separately in
`out/pour_navier_calibration/recorded60_timestep_followup/protocol.json`.
The existing recorded60 full-step endpoint is 158.9471388695 mL.
Both new runs retain viscosity 3.4392377844275503 Pa s, source contact 0.117,
grid160, phase0, 229280 particles, the recorded joints and fixed geometry.
Fresh settling uses the smaller step with the unchanged stopping rule.
No parameter fitting or validation measurements enter this diagnostic.

The runner change adds only the CLI choice `--dt-scale 0.25`; its previous source
is archived by SHA256. A detached controller waits for an idle GPU without
interrupting other users' jobs. It writes an endpoint comparison and time-series
figure after each completed replay. Preparation and baseline checks passed;
this entry records launch, not completion. Consult the follow-up `status.json`
and `REPORT.md` for the actual results. The previous target table remains under
numerical review. The declared adjacent time-step tolerance remains 1 mL;
contracting changes alone do not prove convergence or real-robot accuracy.
