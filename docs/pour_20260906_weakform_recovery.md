# September 6: restoring the weak-form pouring pipeline

The previous empirical angle table did not demonstrate the paper's method and is
superseded. The current workflow identifies one viscosity from the replacement
`09-04-60-2s` recording, freezes it, and runs full forward MPM at new angles. The
45°/50° measured endpoints and the six later validation results are excluded from
all fitting and planning calculations. They were seen during earlier development;
this is not a claim of retrospectively blind development. Future pours made after
these predictions are frozen can provide prospective validation.

## Identification

The algebraic, two-pass `brink-integrated` estimator in
`experiments/pour/pour_weakform_identify.py` is unchanged: reconstruct the source
head from volume conservation and logged cup pose, integrate the spout-edge film
forcing over time, and fit the coefficient `1 / eta`. There is no simulator inverse
fit and no use of final receiver measurements to rescale the optical observations.
The initial volume is 300 mL and density is 1260 kg/m³.

The receiver silhouette alone determines its pose in `endpoint_geometry.json`
(despite that historical filename). Two source-cup external contours determine
the cup-to-gripper pose in `grasp_two_poses.json`. Both come from this recording.
Fresh per-frame cup poses are reconstructed from the recorded joints; this also
fixes stale reference-pose metadata in an older diagnostic observation file.

For the geometry audit, `out/pour_wf/09-04-60-2s/fit_grasp_two_poses.py` fits
external rim/base points at two times in this same recording. The receiver
position comes from `geometry(cam, 60)` in
`out/pour_wf/angle_mismatch_check/compare_endpoints.py`: that function uses only
the 60-degree rim/base annotations and the fixed cup dimensions. Although the
surrounding historical diagnostic script also reports all three measured
endpoints, its geometry residual contains only projected contour coordinates;
those endpoint values do not enter the geometry fit.

The original whole-receiver colour segmentation was affected by reflections and
the moving source cup's shadow. The replacement readout uses a fixed clear strip
on the receiver's right side: upright image x=238..260, amber hue/value tests and
saturation above 0.55. It uses the existing first-intersection cavity observation
model. No volume gain, endpoint anchor, or monotonic projection is applied. The
other receiver regions and thresholds remain saved as diagnostics. The optical
algorithm was developed on this identification recording, not held out from it.

The fitting window is defined by logged actions: 0.15 s after tilt acknowledgement
to 0.15 s before return send, or [6.478658, 8.182549] seconds after the pour command.
The resulting effective viscosity is **3.397435 Pa·s**, with a 2.758 mL integrated
fit RMS, 28.975 mL observed volume change and all 50 frames retained. Its 0.071 Pa·s
statistical standard error excludes optical and geometric systematics. Nearby
windows give 2.81–3.40 Pa·s; changing the saturation threshold to 0.42 gives
2.579 Pa·s. This substantial optical sensitivity must accompany the estimate.

The frozen fit and hashes are in `out/pour_weakform_recovery/identified/identify.json`.
`identify.png`, `brink_fit.npz`, `observations.npz` and `geometry.json` contain the
fit diagnostics and actual inputs to forward simulation. Prior rejected optical
experiments remain outside the `identified` directory and are not simulation inputs.

## Forward prediction and numerical checks

`experiments/pour/pour_weakform_mpm.py` reads the frozen viscosity artifact. It
rejects an endpoint-fitted viscosity and records input/code hashes with each run.
The physical geometry, wall model, density and material settings are unchanged.
All angles share the reference pour's fixed grid translation. Planned joint paths
use only the 60° recording; the 60° mapping reproduces that reference trajectory.
Other angles use scaled progress on its recorded forward and return paths, which
is an assumed controller model rather than a measurement of future executions.

The 60° forward replay gives 162.609 mL at 192³ and 165.328 mL at 256³, a 2.719 mL
difference. Both conserve the particle ledger, leave no final spill and reach a
stable receiver plateau. This is a two-resolution comparison, not proof of grid
convergence. The existing slip boundary does not resolve the no-slip spout film;
the limitations already explained in the original PDF still apply. Further
numerical checks are saved in `out/pour_weakform_recovery/numerical_checks.json`.

The subsequent 256³ subcell-origin check **failed**: at 55°, shifting x/z by
1.367 mm changes the final volume from 117.989 to 127.849 mL (9.859 mL). Thus the
256³ candidates are withheld, including the initially promising 50°/80.81 mL
prediction. The 320³ comparison also failed: 127.593 versus 135.663 mL, an
8.070 mL difference under a 1.094 mm grid shift. The 384³ study uses the
same frozen viscosity and geometry. It tests both refinement from 320³ and grid
origin at the problematic 55° angle, rather than relying on the easier 60° case.
For that refinement comparison, use the mean of both origin placements at each
resolution. This was specified while the 384³ pours were still in progress:
comparing only one coarse origin would confound refinement with the known phase
artifact. The finer-grid origin difference must separately be at most 5 mL, and
the production origin remains the preselected zero shift. Two placements and two
resolutions are a numerical screen, not a proof of convergence or accuracy.
The export screen requires both its specified origin and resolution comparisons
to differ by at most 5 mL; that rule is not an experimental error guarantee.

The 384³ study **passed this screen**: 132.333 and 135.107 mL, a 2.774 mL origin
difference. Its mean is 133.720 mL versus 131.628 mL at 320³, a 2.092 mL refinement
difference. Comparing just the zero-shift runs also gives 4.741 mL, below the same
limit. Both 384³ runs conserve the particle ledger, end with zero spill and have
stable receiver tails. The selected production setting is 384³ at zero shift;
target runs completed on September 6 at 23:26 UTC. This screen is at 55°, not an origin sweep at
every target angle. The detailed results are in `accepted_grid_checks.json`.

The completed 384³ replay of the original 60° motion predicts **174.869 mL**,
versus the reported 159 mL: a **15.869 mL overprediction**. It conserves particles,
ends without spill and has a stable tail. This is a material physical discrepancy
despite passing the limited numerical screen above. It is not corrected using
the measured endpoint. The same recording's final optical estimate is 163.397 mL.
Over the actual weak-fit samples, the observed receiver gain is 28.975 mL, the
fitted brink-law gain is 30.261 mL, and MPM predicts a 36.915 mL gain.

An independent forward integration of the frozen brink law predicts 160.883 mL
for the same 60° motion. This suggests that transferring the brink coefficient
into the MPM flow/boundary model needs further investigation; endpoint agreement
alone neither proves that diagnosis nor establishes transfer to other angles.
This diagnostic does not supply operator commands. The full comparison is in
`reference_replay_check.json` and `reference_replay_production.png` under the
recovery output directory. The current MPM table is therefore a prospective
validation experiment, not evidence of demonstrated ±5 mL physical accuracy.

New target commands are obtained from forward MPM samples only. Interpolation
between simulated samples proposes an angle; a separate forward run must confirm
that angle before export. Every sample is retained. The exporter rejects changed
inputs, nonmonotonic curves, significant spill, unsettled volumes and targets
without direct forward checks. Matching a numerical target within 2 mL does not
establish physical accuracy within 2 mL or 5 mL.

Pure forward integrations of the brink law are saved separately as research
diagnostics (`brink_forward*_diagnostic*`). They have no measured endpoint input,
but they are not full MPM simulations and are not the operator handoff. Coarse-grid
MPM probes may help initialize an angle search; every released result must come
from its own run at the selected production grid.

## Reproduction and handoff

Use the project virtual environment with `MUJOCO_GL=egl`, `PYTHONPATH=.`,
`OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`.

```bash
# Reproduce the frozen optical arrays into a separate file for comparison.
.venv/bin/python -m experiments.pour.pour_receiver_side_strip \
  --output out/pour_weakform_recovery/reproduced_side_strip.npz

# Identification writes the frozen artifact directory. Do not rerun while a
# forward campaign is in progress; the exporter checks its exact input hashes.
.venv/bin/python -m experiments.pour.pour_weakform_recovery

# Representative full forward replay at the identified viscosity.
.venv/bin/python -m experiments.pour.pour_weakform_mpm \
  --angle 55 --grid 384 --device cuda:0

# Inspect target candidates, then simulate every proposed angle directly.
.venv/bin/python -m experiments.pour.pour_weakform_table

# Export only after all target checks pass.
.venv/bin/python -m experiments.pour.pour_weakform_table --export
```

The new operator archive is `out/pour_weakform_handoff.zip`: just instructions,
a three-column command CSV and a graph. Research inputs and diagnostics stay in
`out/pour_weakform_recovery`, including the complete table provenance. The old
`pour_angle_handoff` command now calls this checked workflow by default; historical
empirical exports require an explicit `--legacy` flag.

### Unattended server continuation

At the user's request, `pour-weakform-20260906.service` runs under Stepan's user
systemd manager, with user lingering enabled so laptop disconnection/logout does
not stop it. `out/pour_weakform_recovery/run_unattended.py` first watches the
existing scheduler without interrupting its GPU jobs. If that scheduler exits
early, the service adopts surviving simulations and resumes missing target
checks. On success it invokes the same strict exporter and checks the three-file
ZIP. Failed scientific/provenance checks stop release; the supervisor does not
relax thresholds or refit parameters to finish unattended.

State is in `out/pour_weakform_recovery/unattended_status.json`, with the server log
at `unattended.log`. Check the service with
`systemctl --user status pour-weakform-20260906.service`. The generated ZIP remains
`out/pour_weakform_handoff.zip`; the service records its SHA256 on completion.

### Completed handoff

All six commanded angles have their own completed 384³ forward run at the same
frozen viscosity. The final commands and numerical receiver amounts are:

| Target (mL) | Command angle (degrees) | Tilt duration (s) | MPM receiver (mL) |
|---:|---:|---:|---:|
| 60 | 47.20 | 4.720 | 61.8253 |
| 80 | 49.19 | 4.919 | 81.7760 |
| 100 | 51.73 | 5.173 | 100.4288 |
| 120 | 53.65 | 5.365 | 119.8994 |
| 140 | 55.65 | 5.565 | 138.0374 |
| 160 | 58.12 | 5.812 | 161.2660 |

All 13 completed production samples are retained in the monotone plotted curve.
The largest target error is 1.963 mL in simulation. The 86,537-byte ZIP contains
only `START_HERE.txt`, `angle_table.csv` (three columns) and `angle_table.png`.
Its SHA256 is `cdd3b571c61348f2e5f7343f84733784ae9726485b3ef92d1b389a65e723f50d`.
`handoff_verification.json` records independent archive, provenance, image and
target checks. The instructions explicitly disclose the 174.9 versus 159 mL
physical reference mismatch. No claim of experimental target accuracy is made.

The unattended service recovered the 55.65° and 53.65° runs after the original
session exited at 22:38 UTC. Their interrupted logs and recovery audit are retained;
only their subsequently completed runs enter the handoff. Both GPUs are free at
handoff completion.
