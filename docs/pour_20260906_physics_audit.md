# One-pour physics audit after the failed reference replay

**Updated user accuracy requirement:** aim for absolute errors no larger than
7 mL, with 10 mL as a hard limit. This supersedes the earlier 5 mL goal below;
historical checks are retained. It does not authorize fitting the six validation
outcomes or accepting the reduced model's 15 mL diagnostic error.

The user authorized this audit on September 6, after the first weak-form/MPM table
failed its own 60-degree reference check. That table is retained as a historical
prediction; it is not recommended for precision pouring pending this audit.

## Data and model discipline fixed before new experiments

- Identification may use only the replacement `pouring_real_data/09-04-60-2s`
  recording, its joint/action logs, the stated 300 mL initial fill, and independently
  measured geometry. Viscosity must be estimated by a time-weak spout-edge balance.
- The six later receiver measurements must not select or fit viscosity, wall
  friction, geometry, optical thresholds, correction factors, or commands. No
  angle-volume regression on those outcomes may supply the controller.
- The 159 mL amount is a check of the same identification pour, not an objective
  for a simulator inverse fit or an optical volume rescaling. A same-pour check is
  not independent validation of the identified law.
- The prior 45/50-degree recordings can be examined only after a candidate is
  frozen, using their actual recorded motion. They and the six later outcomes
  have already been seen; future robot pours are needed for prospective validation
  after model development. Do not describe this history as blind development.
- Every candidate change needs an explicit physical, numerical, or image-based
  rationale independent of matching receiver endpoints. Retain rejected results
  and compare identical inputs when changing the simulator. No favorable grid,
  time window, or optical threshold may be selected for endpoint agreement.
- The existing identified artifacts and production simulations remain unchanged.
  New audit results live in `out/pour_physics_audit`, with input/code hashes.

## Order of work

1. Reconstruct the reference timing and compare inclination, dwell and return.
   Keep observed receiver volume, simulated receiver volume, and simulated source
   depletion distinct. Do not compare a source-depletion trace to a receiver trace
   without accounting for liquid in flight.
2. Audit weak-form mass conservation and flight handling, optical sensitivity and
   window consistency using this one recording. Any correction to the estimator
   needs a manufactured-data test before being applied to real data.
3. Benchmark the MPM wall and viscous-film flow against the analytical solution
   `u(z) = rho*g*sin(alpha)*(h*z-z*z/2)/eta`, with a no-slip base and a shear-free
   surface. First separate material/transfer behavior from cup geometry using a
   periodic plane film. Compare separable/slipping and sticky boundary treatments.
   For a proposed accurate implementation, seek less than 10% flow/profile error
   and less than 5% change on the next spatial refinement, retaining grid-phase
   checks. These are numerical diagnostics, not physical volume guarantees.
4. Once independently justified, replay the exact 60-degree motion with the
   candidate model. Seek agreement with the trustworthy time trace and roughly
   159 +/- 7 mL final volume, with 10 mL the hard rejection limit. Failure is evidence against release; do not fit a
   compensating viscosity, friction or volume offset to make this check pass.
5. Freeze a supported candidate, run diagnostic checks on other recorded motions,
   then create a prospective command table only if the evidence supports it.

The direct spout-law endpoint of 160.9 mL is insufficient evidence on its own.
Its existing synthetic 50-degree trajectory gives 94.3 mL while the earlier real
50-degree recording was reported as 79 mL. That is a warning, not a matched-motion
validation result. A predictor switch must not be used to hide the MPM failure.

## Findings from this audit

### Identification: conservation and measurement resolution

The original implementation subtracts an estimated in-flight inventory when
recovering source head but equates receiver gain to source discharge without
the corresponding change in that inventory. The consistent relation is
`d(V_receiver + V_flight)/dt = Q_source`. The candidate transport implementation
instead integrates releases whose arrival times fall inside the weak test
interval. It still estimates one coefficient by the time-weak spout balance;
it takes no derivative of the observed receiver curve and performs no MPM
inverse fit. Tests with independently specified analytical receiver histories
recover a known viscosity for constant and varying flight times and verify
final mass conservation.

With unchanged optical observations this correction changes eta from 3.3974
to 3.5827 Pa s. Its ballistic transit model assumes zero initial downward
velocity, so it is a documented approximation rather than a measured flight
history. This candidate is not automatically accepted for deployment.

The optical calculation also used 64 samples per ray and 0.5 mm height-search
steps. Keeping geometry, colour threshold, strip and time window unchanged:

| Ray samples | Height step (mm) | Conservative weak eta (Pa s) |
| --- | --- | --- |
| 64 | 0.5000 | 3.5827 |
| 256 | 0.1250 | 3.2149 |
| 1024 | 0.03125 | 3.5422 |
| 4096 | 0.0078125 | 3.5186 |

The final refinement changes eta by 0.67%; the 256-sample result was not selected
for any favourable physical prediction. Extracting every available frame at
the finest resolution, instead of every third frame and then interpolating,
gives 3.6443 Pa s. The historical 50 fitting samples included interpolation
from 33 extracted frames within the window; smoothing adds further temporal
correlation. The quoted regression standard error is therefore not a physical
confidence interval and must not be reported as identification precision.

At full video rate, the two half-window estimates remain different (3.36 and
6.39 Pa s). The ray geometry near the fitted boundary maps one vertical image
pixel to a median 4.51 mL (10th--90th percentile 2.99--4.89 mL). This is an image
scale calculation, not an uncertainty interval, but it explains why a roughly
9 mL half-window gain is a weak constraint on viscosity.

Plateaus and jumps persist in the optical curve after numerical refinement.
At 4096 ray samples with the original extraction stride, separate first/second
half-window fits give 3.01 and 6.32 Pa s, with about 21 and 9 mL observed gain,
respectively. These are diagnostics, not alternate estimates to choose between.
The smaller second-half volume change is particularly sensitive to image noise.
The images and overlays are retained; no threshold was adjusted to endpoints.

### MPM: the source wall is inconsistent with the assumed film

The PDF explicitly assumes zero liquid velocity at the wall for the brink
closure. The production MPM uses a separable Coulomb contact, which imposes no
such condition on a tangentially sliding film. In an independent periodic film
benchmark it continues accelerating instead of approaching the no-slip solution.
At 0.12 s its mean velocity is about 14.7 times the analytical no-slip mean.
That factor belongs to this benchmark and is not a correction factor for pouring.

Sticky contact converges toward the analytical solution but depends substantially
on grid placement. For SDF contact with its production half-cell band, at 16
cells across the film the flux ratios are 0.952 and 0.826 at two grid phases;
at 32 cells they are 0.990 and 0.921. The actual 384-cubed scene resolves the
inferred maximum film thickness with only about 5.7--6.7 cells. In addition,
the collider thickens the physical cup wall outward to three grid cells.
These are numerical/model discrepancies, not evidence for a different viscosity.

An experimental ghost-velocity extrapolation was tested only on a flat film.
It passes some aligned-grid cases but fails other grid phases (about 16% excess
flux at 16 cells across the film and phase 0.75). It is rejected for deployment;
no core solver code or production collider has been changed to enable it.

Separate full reference replays isolate (a) sticky source contact at the original
eta and (b) the conservative transport eta with the original separable contact.
They use exact recorded joints, fresh settling and unchanged geometry. Results
are retained in `out/pour_physics_audit/replay`; neither is an endpoint fit.

Both replays have now finished. The sticky source with the original 3.3974 Pa s
gives 114.55 mL in the receiver, 117.64 mL source depletion and 3.09 mL outside.
The conservative transport estimate (3.5827 Pa s) with separable contact gives
174.41 mL in the receiver, versus 174.87 mL before the transport correction.
Neither passes the same-pour consistency check. Changing the source boundary
condition is physically motivated, but the coarse sticky result is not a fix.

### Full-scene compact-domain verification and refinement

The swept CAD bounds of both cups fit inside a 0.35 m cube with at least 35 mm
clearance. The current crop test retains the full receiver, gravity, world-frame
motion and fluid; it only translates the fixed computational domain. It uses
192 cells across 0.35 m, exactly the original 384 cells across 0.7 m physical
cell size, and translates by an integer number of original grid cells. Both
the original sticky result and this crop test use the original viscosity and
SDF resolution. Every frame checks fluid clearance from the grid boundary;
coming within three cells rejects the crop.

Before reading its completed result, acceptance limits are fixed to 1 mL for
both final receiver amount and source depletion, and 1.5 mL receiver-curve RMS
difference against the full-domain sticky replay. Particle counts, timestamps,
physical cell size and shared input hashes must also agree. This compares two
numerical solutions, not either solution with an experimental endpoint.

Only after crop acceptance, refine the full-scene MPM at 320 and 384 cells in
the 0.35 m cube (1.094 and 0.911 mm cells), at each of the two grid phases 0 and
0.5 cell in x/z. All four runs use the same corrected, spatially refined,
full-video-rate weak estimate, 3.6443097761 Pa s; source sticky and receiver
separable; the same geometry; and SDF resolution 384. These are fixed numerical
choices independent of the 159 mL check and the six validation outcomes.
Both grid phases must change final receiver amount by no more than 3 mL on
refinement, and the finer-grid phase difference must be no more than 3 mL.
Passing these checks is a numerical screen, not proof of convergence or physical
accuracy. Reject nonconservation, changing tails or fluid hitting the boundary.
Do not pick whichever grid phase happens to predict the measured endpoint.
Further geometry-resolution or time-step checks may still be needed before any
new command table. No automatic table export is authorized by this batch.

The protocol was frozen at 2026-09-07 00:23:42 UTC, before the crop replay
finished. `pour_compact_launch.py` waits for that replay, verifies the declared
crop limits and input hashes, and only then starts one persistent systemd worker
per GPU. Each worker runs its phase at 320 then 384 cells. Failed cases stop the
worker and are retained. `pour_compact_verification.py --collect` reports both
numerical differences and the separately labelled same-pour physical check;
it cannot produce a robot table. The crop-verification, weak-transport and
handoff-guard test suites currently pass all 35 tests.

The crop comparison completed successfully: final receiver difference 0.3559 mL,
source-depletion difference 0.3203 mL and receiver-curve RMS difference 0.2419 mL.
The closest fluid approach to the domain boundary was 52.8 mm. The compact run
took 883 s, versus 998 s for the original full-domain sticky replay. This verifies
the crop at unchanged physical resolution; its 114.90 mL receiver prediction is
still physically inconsistent with the 159 mL observation and is not a repair.

Both persistent refinement workers are now running the predeclared 320-cell
cases, one phase on each GPU, then proceeding to their 384-cell cases. Each
320-cell case has 1,834,244 particles and a 10.92 microsecond time step. The
viscosity is fixed at 3.6443097761 Pa s. Refinement results are pending.

**Compute review hold (2026-09-07 00:38:43 UTC):** the user questioned committing
5--6 hours without evidence of likely accuracy. Both worker supervisors were
paused with SIGSTOP while their existing 320-cell simulation children continue.
This prevents either 384-cell case from starting automatically. The initial
protocol and scientific parameters remain unchanged; this is a resource-based
pause recorded before any refined endpoint is available, not selection of a
favourable numerical result. The estimated 5--6 hours covered reference
diagnostics only, not the further angle sweeps needed for a robot table.
See `compact/compute_review_hold.json` for the supervisor/child process record.
Review the completed 320-cell pair before resuming the supervisors; passing a
coarse result alone does not authorize a table or remove the finer-grid checks.

### Completed 320-cell pair: physical check failed

Both runs completed in approximately 103--105 minutes. The two grid placements
predict 122.5759 and 122.4513 mL in the receiver, respectively, against the
reported 159 mL: errors of -36.4241 and -36.5487 mL. Their final receiver amounts
differ by just 0.1246 mL; receiver-curve RMS difference is 0.1360 mL and the maximum
curve difference 0.4411 mL. This passes the grid-placement screen, but a pair at
one resolution does not establish spatial convergence.

Both final particle states are finite, the initial rest volume is conserved,
receiver tails vary by less than 0.12 mL, and outside volume is below 1.8 mL.
Consequently, unfinished settling or liquid still outside the receiver does not
account for the 36.5 mL discrepancy. The 384-cell cases remain on hold and the
GPUs are idle. This model must not supply the robot table.

### Synthetic recovery check from the saved MPM results

`pour_closure_recovery_audit.py` runs no new simulation. It applies the same
time-weak hydrostatic brink closure to the saved MPM source-particle inventory,
using the existing 60-degree geometry, poses and fitting window. Direct source
inventory removes optical liquid measurements and flight-time estimation from
this diagnostic. No experimental endpoint or any of the six validation outcomes
enters the calculation. The diagnostic does not replace the real-video estimate.

The MPM input viscosity is 3.6443 Pa s. The weak closure applied to its simulated
flow returns 6.6604 and 6.6407 Pa s for the two grid placements, about 1.82 times
the input value, with fitting RMS residuals of 0.43 and 0.47 mL. During the window,
MPM releases about 30.65 mL; the hydrostatic brink forcing at the MPM input
viscosity predicts about 57.56 mL on that same source-state history.

Thus this weak closure and the current discretized MPM do not attach the same
viscosity to the same simulated flow. This discrepancy exists without video
volume errors. It does not isolate closure error from remaining MPM numerical
error, and the ratio must not be used as an empirical viscosity correction.
No coefficient or geometry has been changed and no new handoff has been made.
Results are in `compact/closure_recovery/`; `compact/pair_review.png` shows the
two completed forward histories.

### Frozen reduced-model check, explicitly separate from MPM

After freezing the 4096-ray/3.5186 Pa s candidate and its forward equations, the
same local brink law was integrated over the actual 60/45/50 joint histories.
The rigid hand-to-cup transform was fixed from the 60-degree video. No other
fluid observations entered its identification or prediction. This is a reduced
hydrostatic film model, not an MPM result or a replacement robot table.

| Recorded command | Predicted settled amount, assuming no spill (mL) | Previously reported amount (mL) |
| --- | --- | --- |
| 60 degrees | 159.13 | 159 |
| 45 degrees | 55.25 | 48 |
| 50 degrees | 94.34 | 79 |

Halving the 60-degree integration time step changes its result by less than
0.001 mL. The good same-pour endpoint does not establish transfer: the other
checks miss by about 7 and 15 mL. Those previously seen recordings are diagnostic
development data, not blind validation. The six later target-volume outcomes
are absent from all of these fits and model-selection objectives.

External-outline overlays on the other recordings check the fixed-grasp
assumption without fitting any liquid measurements. Whether the cup was
reinserted between recordings is a separate input uncertainty; it cannot be
resolved by adjusting geometry until a volume prediction agrees.

### Independent audit of the predicted robot motion

`pour_motion_transfer_audit.py` compares the synthetic joint-motion generator
with the original 45/50-degree joint logs, using the fixed 60-degree hand-to-cup
transform. It reads no liquid observations and fits nothing. The generated
60-degree motion reproduces the reference to floating-point precision.

At the common dwell, the predicted-minus-recorded cup tilt is only +0.007 degrees
for 45 and +0.012 degrees for 50; tip height differs by less than 0.2 mm. During
inclination, maximum orientation differences are approximately 0.91 and 1.31
degrees. The 50-degree acknowledgement is 0.128 s later and return command
0.125 s later than the template predicts. Its return trajectory consequently
has a transient orientation difference of 8.88 degrees, at a recorded cup tilt
of about 32 degrees. This is a timing/path diagnostic, not an 8.88-degree error
in the final pouring angle or evidence of a particular volume error.

Use exact measured joint histories for the original 45/50-degree diagnostic
replays. Do not assume that accurate dwell angles establish accurate full
motion transfer, or compensate for the timing discrepancy using liquid endpoints.
The fixed-grasp motion comparison cannot detect an actual change in cup insertion.
Artifacts are in `out/pour_physics_audit/motion_transfer/`.

### Export guard

The table exporter now refuses to write a handoff when the identification
command replay differs from the user-reported 159 mL by more than the updated
10 mL hard limit, and separately reports whether the 7 mL goal is met.
This is an acceptance gate only: it never changes
viscosity, geometry or interpolated simulation results. A passing reference is
necessary, not sufficient, for subsequent prospective validation.

An actual export attempt is blocked at +15.87 mL. The previous ZIP remains
byte-for-byte unchanged as a historical artifact and is not recommended for
precision pouring. Tests cover the transport mathematics, forbidden calibration
substitutions, and refusal to export a failed reference before writing files.

## What would constitute a defensible continuation

1. Keep the six later outcomes out of parameter estimation. Preserve the raw
   results and report their original errors when discussing that experiment.
2. Resolve the actual spout wall and crest, and verify no-slip flow across grid
   placements and refinements before using MPM to supply control commands.
   Merely increasing the full-scene grid or switching to sticky contact has not
   established this. The compact-domain test above retains both cups and exact
   world-frame motion, avoiding an artificial outlet or moving-frame change.
3. Improve optical observability independently of pouring outcomes: a closer,
   fixed view of the receiver boundary and spout, diffuse lighting, and an
   independently measured geometric scale. Weights can remain validation-only;
   using a scale trace for identification would require disclosing an instrumented
   pour instead of claiming video-only identification.
4. A single uninterrupted identification pour may contain more than one held
   angle. This can expose inconsistency inside one video without using multiple
   validation pours as calibration. It is still one experiment, and its complete
   trajectory and fitting rule must be specified in advance. It would not by
   itself repair an inaccurate simulator or validate the lubrication closure.
5. After these checks, freeze the complete pipeline and all commands before new
   prospective validation. Do not label the already-seen 45/50 recordings blind
   validation, and do not select the reduced model simply because its 60-degree
   endpoint happens to agree.
