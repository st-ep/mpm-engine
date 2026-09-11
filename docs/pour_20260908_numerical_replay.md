# Numerical replay revision, 8 September 2026

The viscosity identification is unchanged. The current candidate uses
**3.4392377844275503 Pa s**, reproduced by the PDF's two-pass integrated spout-edge
fit on `out/pour_three_video_identification/09-04-60-2s/observations.npz`.
Its fit window remains acknowledgement + 0.15 s to return send - 0.15 s.
No other video's viscosity estimate is pooled into this value. The directory
name records an earlier consistency comparison; each fit in that comparison
uses its own recording. The older recovery artifact with viscosity 3.3974 is
not the input to these runs.

The user separately authorized using the **same original 60-degree pour's
159 mL endpoint** to calibrate one effective source-contact coefficient, with
viscosity fixed. Therefore the claim is **one-pour weak viscosity identification
plus same-pour numerical contact calibration**. The 159 mL match is calibration,
not held-out validation, and the contact coefficient is not a measured
microscopic liquid-wall property.

## Input separation

- The source cup shape is the existing caliper-based parametric solid, including
  its documented shape assumptions. It is unchanged in this revision.
- Source mounting pose comes from exterior contours in two poses of the original
  60-degree recording. Receiver pose comes from its exterior rim/base contours;
  `endpoint_geometry.json` explicitly records that the liquid endpoint was not
  used. The frozen scene contains these rigid poses, not a fitted volume scale.
- The same original 60-degree video supplies the weak fit and the separate
  159 mL contact-calibration datum. Initial fill is 300 mL.
- Future motion timing uses two additional joint/action histories at 47.06 and
  52.92 degrees. Their liquid outcomes do not enter the timing model. These are
  additional robot-motion measurements and must be disclosed as such.
- The original 45/50 endpoints and the six later liquid outcomes are excluded
  from model parameter fitting and command selection. Previously inspected
  experiments are not described as new blind validation. Retrospective error
  forecasts are chat-only diagnostics and are not inputs to the forward solver.

## Why the numerical implementation changed

The old calibrated APIC replay changed substantially under timestep refinement:
158.9471, 152.1890 and 141.5658 mL at standard, half and quarter timesteps.
Matching the calibration endpoint at one timestep did not remove this error.

Separate analytical tests motivated stable log-volume/pressure arithmetic and
compensated particle-position updates. Both fixed demonstrated arithmetic
defects, but their combined original-60 timestep gap remained 6.2661 mL.
An isolated transfer prototype then retained a damped affine-FLIP velocity
residual while keeping grid-based advection and the APIC affine update. Its
residual factor is `alpha(dt) = exp(log(0.99) * dt / dt_ref)`, with `dt_ref`
given by the existing acoustic/viscous stability restriction. Time scaling is
an explicitly derived numerical variant. No blend parameter was optimized
against robot receiver outcomes.

The time-scaled blend passes the tested shear timestep-consistency checks;
the finer shear amplitude errors are 1.66–1.72%. The coarse shear amplitude
errors remain 6.46–6.58%, outside the declared 5% criterion. These limitations
remain in the record. The unblended variant failed the stationary-cup settling
check and was rejected. Existing sticky and quadratic no-slip wall alternatives
also failed their coarse analytical screens and were not adopted for target
planning.

At the old contact coefficient 0.117, the revised transfer gave 179.0592 and
179.3327 mL at standard and half timesteps: a 0.2735 mL gap, but no longer a
159 mL calibration. Contact was therefore recalibrated against that same pour.
The initial numerical exploration cap 0.20 failed to bracket 159 mL. A separately
disclosed revision increased the cap to the engine's default 0.40; this was not
an originally declared bound or a measured physical-friction interval. The old
bounded trials remain preserved. New trials at 0.249 and 0.272 used only the
original 60-degree calibration datum and forward MPM results.

## Current candidate and remaining checks

The candidate contact coefficient is **0.272**. The physical viscosity, geometry,
recorded calibration motion, fill, receiver contact and production particle
count remain fixed. The production grid is 160 cubed over a 0.35 m domain,
with 229,280 particles. The comparison grid has 396,197 particles at 192 cubed.

| Original 60-degree replay | Receiver volume (mL) | Check |
| --- | ---: | --- |
| 160 cubed, standard timestep | 159.5307 | Within 1 mL of same-pour calibration |
| 192 cubed, same contact and standard timestep | 160.3498 | Difference 0.8191 mL, within 3 mL criterion |
| 160 cubed, half timestep | 160.5722 | Difference 1.0415 mL; fails the 1 mL criterion |

The maximum difference over all 773 matching grid-comparison frames is 1.3743 mL.
The grid comparison figure and independent reviewer have been inspected.
Particle accounting, finite final state, initial settling and receiver tail
checks pass for those completed cases. These are limited numerical checks,
not proof of continuum convergence or real-robot accuracy.

Within the unchanged weak-fit window, the coarse MPM predicts 35.2613 mL of
relative receiver gain, versus 30.3876 mL observed and 32.2812 mL predicted by
the fitted weak law. MPM's relative-curve RMSE is 3.2851 mL. This additional
same-video diagnostic is retained without fitting a new offset, time shift,
window or viscosity.

The half-timestep result has 0.0772 mL outside both cups and 0.0419 mL tail
variation. Its maximum raw-frame difference from the standard timestep is
1.2653 mL. The timestep comparison figure has been inspected. The original
automatic sequence **stopped on the failed 1 mL check**, as intended; no
`all_review.json` accepting this model was produced.

A separate, explicitly documented diagnostic decision permits only the same
two previously chosen angles, 48.37 and 53.04 degrees, at unchanged parameters
and with the measured timing correction. The purpose is to assess whether the
candidate improves the target predictions before spending more compute. The
1.0415 mL discrepancy is not rounded into a pass, the original criteria are
unchanged, and these diagnostics do not constitute a bulk release. The requested
real-error forecast must be reported in chat after these probes and before
any bulk search. No further simulation is queued behind them.

Authoritative decision:
`out/pour_physics_audit/aflip_blend_time_20260907/candidate_mu0.272000/diagnostic_probe_decision.json`.
Diagnostic outputs are in `diagnostic_timing_probe_48.37` and
`diagnostic_timing_probe_53.04` below that candidate. The original sequence's
failed status, numerical implementation, protocols and reviews are preserved.
The prior Philip table and ZIP are preserved byte for byte.

## Completed diagnostic probes and target verification

The two diagnostic probes completed and passed the independent input-hash,
particle-accounting, finite-state, settling and capture/tail audits:

| Command | Revised MPM (mL) | Previous timing-corrected MPM (mL) | Simulation change (mL) |
| --- | ---: | ---: | ---: |
| 48.37 degrees | 68.5311 | 61.7376 | +6.7934 |
| 53.04 degrees | 106.5767 | 101.5339 | +5.0427 |

These changes include the revised numerical transfer and the same-pour contact
recalibration; they are not isolated timing effects or real-robot errors.
The requested pre-bulk error forecast was subsequently reported in chat and is
not stored in repository files.

Linear inversion/extrapolation of these two **simulated** points proposes
47.32 degrees for a 60 mL forward check and 52.23 degrees for a 100 mL forward
check. The direct MPM checks returned 61.0075 mL at 47.32 degrees and 100.1548 mL
at 52.23 degrees. The latter satisfies the unchanged 1 mL target tolerance;
the former narrowly exceeds it. MPM-only inversion proposes 47.18 degrees
for the next 60 mL check. That completed at 60.5535 mL and passed the unchanged
1 mL target tolerance. The 60 and 100 mL commands have therefore been directly
checked. No robot package is released yet. Neither the model nor the command proposals use the
historical liquid validation outcomes.

`target_model_protocol.json` freezes the model, retains the failed original
timestep criterion, and bounds the new target verification campaign to 20 cases
and three hours. Together with the two earlier timing cases and the two revised
diagnostics, that allows at most 24 correction/verification target cases. The
earlier campaign is not reset. `pour_transfer_target_case.py` enforces allocation
under a file lock, rejects overwrites and reads no real-error forecast. The final
requested table still has eight targets from 60 to 200 mL; it is not complete.


The detached target-search controller has a critical phase that verifies/refines
60 and 100 mL and stops for review before the remaining commands. Its subsequent
full phase obtains two higher-angle simulation anchors and directly checks each
selected target. It uses only simulated receiver counts, preserves every result,
and stops on any raw volume reversal. Control protocols and state live under
`candidate_mu0.272000/table_search/`. The original target budget is not reset.


## Action export check

The generated 47.18, 47.32 and 52.23 degree episodes were inspected directly.
Their tilt command durations equal angle/10 seconds, return command durations
are 2 seconds, and acknowledgement-to-return-send holds are 2.00188744 seconds
under the frozen measured timing model. The runtime log's `hold 0s` means no
additional hold was inserted into that already populated trajectory. It does
not mean the experiment's two-second hold is absent. These checks use action
and joint records only, without receiver outcomes.


After both critical checks completed, their raw outputs and frozen-input reviews
were inspected, and the full search was started as a detached service. It first
runs the planned 60 and 65 degree motions with the measured timing correction,
then selects and verifies the remaining commands from MPM outputs. The original
recorded 60 degree calibration and the planned timing-corrected 60 degree case
are distinct trajectories and retain separate outputs. Neither viscosity nor
contact is recalibrated during target search.


## Prepared recorded-motion validation

`pour_transfer_recorded_validation.py` is prepared for at most two further
retrospective diagnostics, replaying the original 45 and 50 degree joint tracks.
These have not been launched. The wrapper requires the complete command package
first and freezes its hash before any validation replay. It cannot update the
package or refit viscosity, contact, geometry or motion. These recordings were
previously inspected, so they are not described as unseen or blind tests.

Both CPU export preflights passed: recorded joints and action times are
preserved. An independent comparison against the raw joint logs found maximum
export differences below 1.6e-15 radians. These checks validate the interpolation
of measured samples, not unobserved motion between those samples. The 45 degree
pour used `execute_trajectory`; the 50 degree pour used `go_to_pose`. Their actual
recorded joint tracks, including their delays, are replayed; no average timing
correction is added to them.

The identification routine and its source observations still come exclusively
from the original 60 degree recording. The existing original-60 cup-to-gripper
transform, receiver geometry and contact remain fixed. Thus mounting variation
between recordings is not silently calibrated away. The wrapper reads no liquid
endpoint. `pour_transfer_recorded_validation_review.py` separately checks raw
joint/action agreement, frozen inputs, particle accounting and final-state
validity. Any comparison to measured liquid amounts is a subsequent diagnostic,
not a command-selection or parameter-fitting input.

These two replays are allowed only if the completed target cases plus allocated
validation cases stay within the existing 20-new-case budget and the original
three-hour deadline. The target search takes priority. A failed diagnostic must
be reported; it does not authorize retuning against those outcomes.


## Higher-angle target anchors completed

The planned 60 degree replay completed at 160.6036 mL (0.0563 mL outside both
cups; 0.0497 mL final-half-second variation). The planned 65 degree replay
completed at 194.3118 mL (0.0301 mL outside; 0.0236 mL tail variation). Both raw
outputs passed the independent reviews. The 60 degree command is accepted for
the 160 mL target under the unchanged 1 mL target tolerance.

The planned, timing-corrected 60 degree output is 1.0729 mL above the original
recorded-60 calibration replay. This compares distinct motion clocks and replay
windows with the same fixed material/contact/geometry; it is not a new calibration
residual. No parameter was retuned to force the planned 60 degree output to 159.

The MPM-only inverse search next proposes 49.77, 54.77, 57.35, 62.88 and 65.84
degrees for the 80, 120, 140, 180 and 200 mL targets respectively. These are
unverified proposals until their direct forward runs complete. The controller
has started the first two and queued the remaining three.


## First interior-target checks and held-pose diagnostic

The direct 49.77 degree run completed at 80.2547 mL and satisfies the 80 mL
target tolerance. The 54.77 degree run completed at 122.0464 mL and does not
satisfy the 120 mL tolerance. Both passed their independent simulation-output
reviews; only the latter command needs MPM-only refinement. These are simulated
amounts, not measured validation outcomes. The full controller has moved to
57.35 and 62.88 degrees, with 65.84 degrees still queued.

A separate forward-kinematics diagnostic sampled 100 commanded angles over
45–68 degrees and 25 held poses per angle. The largest difference between the
planned held robot pose and the nominal command was 0.0888 degrees in rotation
and 1.0389 mm in TCP position. At the current 200 mL proposal, 65.84 degrees,
the differences were 0.0772 degrees and 0.5888 mm. No parameter or command was
changed. These sampled endpoint differences do not establish physical cup
reinsertion, tracking between recorded samples, or an error bound on volume.
The measured timing wrapper preserves these spatial poses.

Reproducible diagnostic: `experiments/pour/pour_transfer_command_pose_check.py`;
outputs in candidate `table_search/command_pose_check/`. It reads no liquid
measurements and does not write real-volume forecasts.


The 57.35 degree case completed at 140.4248 mL, inside the 140 mL target
tolerance. The 62.88 degree case completed at 182.9911 mL, outside the 180 mL
target tolerance. Both independent output reviews passed; the latter angle will
be refined from simulated volumes only. The 65.84 degree check is now running.
Per-case `bulk_search_queued: false` fields describe the worker, which does not
launch other runs; scheduling is performed by the separate full-search controller,
whose protocol, live service and status are authoritative for the queue.


## User stopped refinement and removed the 200 mL target

The later explicit instruction was "don't do final refiment, just skip 200 mL".
The full-search controller was paused and terminated, together with its two
incomplete refinement runs at 54.54 and 62.50 degrees. Their partial outputs
remain in place, with cancellation records; neither has a completed-result
claim. The queued 65.64 degree run was not launched. The earlier completed
65.84 degree simulation (201.7712 mL) remains preserved but is omitted from the
new robot handoff. The service's terminal `Result=signal` reflects this deliberate
cancellation, not a new numerical failure.

`experiments/pour/pour_transfer_package_completed.py` creates the new seven-row
handoff using the frozen, independently reviewed simulation points. It never
launches a simulation and leaves the model and original search state unchanged.
Previously accepted direct-check rows are retained for 60, 80, 100, 140 and
160 mL. For 120 and 180 mL, the unchanged piecewise-linear MPM inverse proposes
54.54 and 62.50 degrees, respectively. These two commands are interpolated
predictions, not completed direct checks. Their bracketing simulations and
hashes are recorded in provenance. No measured liquid validation result enters
this selection. This exception to completing every forward check is recorded
explicitly; the original 1 mL target tolerance is not rewritten or claimed met
by an unperformed check.

The artifact is `out/pour_navier_calibration/philip_corrected_20260908/`.
Its 78,936-byte ZIP contains only `angle_table.csv`, `angle_table.png` and
`START_HERE.txt`. The graph uses the same monotone piecewise-linear simulation
curve as command interpolation and visually distinguishes the two unchecked
commands. Detailed provenance and the package review remain outside the ZIP.
The image was visually inspected; CSV/ZIP integrity and preservation of the old
ZIP passed. All previous simulation results remain available.

The final command rows (mL, degrees, tilt-command seconds) are:

| Target | Angle | Tilt duration |
| --- | --- | --- |
| 60 | 47.18 | 4.718 |
| 80 | 49.77 | 4.977 |
| 100 | 52.23 | 5.223 |
| 120 | 54.54 | 5.454 |
| 140 | 57.35 | 5.735 |
| 160 | 60.00 | 6.000 |
| 180 | 62.50 | 6.250 |

The one-video weak viscosity, same-pour contact calibration, geometry and
measured timing model remain unchanged. The original timestep screen still
fails at 1.041521284 mL. No claim of established robot accuracy within 10 mL is
made. The prepared original 45/50 validation replays were not launched. No
further simulations follow from this packaging step.


## Interpolation evidence after freezing the handoff

A CPU-only diagnostic, saved outside the ZIP as
`philip_corrected_20260908/interpolation_evidence.json`, compares linear and
shape-preserving cubic interpolation of the completed MPM points. At the two
unchecked commands, the cubic-minus-linear differences are 0.1489 mL (120 mL
target) and 0.4264 mL (180 mL target). Leaving out each interior completed
simulation and predicting it from its two neighbours gives a maximum absolute
difference of 2.9716 mL across the available simulations. These checks neither
run new MPM cases nor use measured robot volumes. The released commands and
archive are unchanged.

Interpolation-method agreement is not an error bound, and leave-one-out
brackets are wider than the specific unchecked-command brackets. Neither
diagnostic bounds physical model error or establishes accuracy on the robot.
Increasing a command by half a degree changes the interpolated MPM result by
roughly 3–4 mL near these commands; this also changes the planned trajectory
duration and is not a separate measurement of cup mounting error.

The six historical validation outcomes have already been inspected during
development and retrospective forecasting. They are excluded from parameter
and command fitting, but must not be described as a fresh blind validation of
this revised handoff. Fresh robot outcomes, with the commands frozen beforehand,
are needed to establish the claimed real accuracy. Real-error forecast values
remain chat-only and are not stored in this diagnostic.


## User clarified that only the 200 mL target should be skipped

The user clarified that the earlier instruction meant skipping 200 mL only,
and explicitly requested completion of the 120/180 checks and an updated ZIP.
The two cancelled partial attempts were relocated intact to
`candidate_mu0.272000/cancelled_target_attempts/20260908_user_scope/`, with a
record of original paths and metadata hashes. The original worker can therefore
run fresh direct checks at 54.54 and 62.50 degrees without overwriting partial
evidence or modifying any frozen source file. Cancelled attempts remain counted
in the original budget: 12 previous attempts plus two fresh checks equals 14
of 20; the original deadline also remains unchanged.

Both direct checks started under detached user services at UTC 02:51:10 on
8 September. They use the exact frozen viscosity, contact, geometry, timing,
grid and particle count. There is no new identification or parameter fit. The
released angles themselves are held fixed. The 200 mL target remains excluded.
The new check protocol and launch records live in
`candidate_mu0.272000/direct_check_120_180_20260908/`. A separate packager is
prepared to update the graph and notes after independent reviews pass, while
preserving the earlier artifact. No completed direct-check claim is made until
the corresponding outputs exist and pass review.


## Direct checks and one MPM-only adjustment of the 180 mL command

The 54.54 degree replay finished at 120.6647 mL, within the unchanged 1 mL
simulation target tolerance. The 62.50 degree replay finished at 178.0308 mL,
outside that tolerance for the 180 mL target. Both passed independent accounting,
finite-state, settling, tail and input-hash reviews. The 62.50 target error
remains a failed target check, distinct from its passed output-integrity review.

The same piecewise-linear inverse used by the original MPM-only search proposes
62.65 degrees using the completed 62.50 and 62.88 degree simulations. Its tilt
duration is 6.265 seconds. One additional direct replay started at UTC 03:02:44
under `pour-aflip-refine180-20260908.service`. The explicit follow-up protocol
is `direct_check_120_180_20260908/refine_180_protocol.json`. It permits one case
and keeps the original deadline and total attempt limit; including cancelled
attempts, this is attempt15 of20. It changes only the proposed 180 mL command,
not the material, contact, geometry, timing model or any identification input.
No measured validation volume enters the proposal. The other six commands stay
unchanged, and 200 mL remains excluded.

The previous linear/PCHIP agreement of0.4264mL at62.50 degrees did not bound its
actual interpolation error: the direct value is2.0064mL below the linear
prediction. This is why the earlier interpolation diagnostic explicitly
declined to provide an error bound. The direct result and prior prediction are
both retained in the record.


The62.65 degree follow-up completed at178.3029mL. Its independent output review
passed, but it still misses the unchanged1mL target tolerance. The same MPM-only
inverse then proposed62.73 degrees/6.273 seconds using the tighter completed
62.65/62.88 bracket. A single further replay started UTC03:14:35, with its
decision recorded in `refine_180_second_protocol.json`. This is attempt16 of
the original20, counting cancellations; the original deadline remains unchanged.
Neither prior failed target check is discarded or relabelled. No measured
validation outcome or parameter adjustment enters this second command proposal.


## Directly checked handoff published

The62.73 degree case finished at181.2967mL. It passed independent particle
accounting, final-state, settling, tail and frozen-input review, but its+1.2967mL
target residual still fails the original1mL target criterion. The closest
completed180 command was retained for the experimental handoff under the
explicit `experimental_release_decision.json`; further command search stopped.
This is a release exception, not a revised tolerance or a passing target check.
The180 target failure and original timestep failure remain false in provenance.
No fresh physical accuracy claim follows from these completed simulations.

The new table has direct MPM outputs for all seven commands. Six satisfy the
original1mL target criterion. Only the180 row changes:62.73 degrees,6.273-second
tilt command. The120 row remains54.54 degrees/5.454 seconds, now directly
simulated at120.6647mL. All other commands, the weakly identified viscosity,
contact, geometry and robot timing model remain unchanged.

The new artifact is `out/pour_navier_calibration/philip_corrected_verified_20260908/`.
Its three-file ZIP was atomically published to `final/philip_corrected_handoff.zip`
(73,828 bytes; SHA256
775e9ba4e1d0ebbe6cdcbf15ff07738c750c8de18af28ce66704edb22463b8cb).
The previous archive remains preserved. The PNG was visually inspected and
archive contents, CSV/direct-result agreement and provenance were checked.
All four continuation services completed successfully; no further job is queued.
The total16 attempts, including cancelled cases, remain within the original
20-attempt cap and original deadline. No validation outcome entered any command
proposal, and no real-error forecasts were stored.


## Plot labels added without changing the MPM predictions

The plot-only artifact `out/pour_navier_calibration/philip_corrected_labeled_20260908/`
adds the command angle above every selected point and marks intermediate raw MPM
runs. The same piecewise-linear curve remains, including its sharp bend near
180 mL: 62.50 degrees gives178.0308mL,62.65 gives178.3029mL, and62.73 gives181.2967mL.
The raw sequence is monotone. The underlying reason for the locally abrupt
slope change has not been established; the plot does not establish a physical
feature of the real pour. No smoothing or new simulation was performed.

CSV and START_HERE are byte-identical to the prior verified package. Predictions,
raw points, model parameters and all acceptance flags remain unchanged. The PNG
was visually inspected before atomically publishing the three-file ZIP to
`final/philip_corrected_handoff.zip` (95,654 bytes; SHA256
a06499e537fcbbe9d8fafc4fbed125c147684cc17ae4a89f373e716626b1ae8c).
The previous package is preserved, with linked provenance in the new artifact.
