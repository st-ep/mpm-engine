# Surface-objective planning for work-limited hexagonal shaping

The active manuscript now uses the cylindrical-finger X experiment described
in [x_shaping.md](x_shaping.md). This document reproduces the preceding
work-controlled hexagonal study; its outputs remain archived.

This study tests whether a boundary objective and a larger, equal planning
budget improve the identified shapes from `hex_shaping_work_20260909`.
It retains both materials, the shared nominal model, the analytic target,
and the work-feedback execution policy. Historical studies are preserved.
The output directory is `out/hex_shaping_surface_20260909/`.

## Model information and control

Material A has E = 80 kPa, nu = 0.30, and yield = 1 kPa. B has E = 240 kPa,
nu = 0.30, and yield = 10 kPa. Both have density 1000 kg/m³ and use Hencky
elasticity with von Mises plasticity. The shared nominal model has E = 80 kPa,
nu = 0.30, and yield = 5.5 kPa. The identified parameters and held-out press
data are imported unchanged from `out/hex_shaping_study_20260908/`.
Their paths and hashes are in `imported_inputs.json`; the actual inputs are
copied into `inputs/`. This experiment does not refit the material to the
shaping target or test constitutive-family selection.

The target is an analytic regular hexagonal prism, 90 mm across flats and
81.666695 mm high. Its volume is fixed to the baseline initial particle
volume, 0.0005728760037726488 m³. The initial body is 120 × 80 × 60 mm.

Each model independently optimizes three final jaw gaps for directions
0°, 60°, and 120°, repeated once. All five searches start from [80, 85, 87]
mm, with an initial simplex formed by reducing each coordinate by 10 mm.
The bounds are 65–115 mm and the budget is 64 objective evaluations for
every model. Bounded Nelder–Mead uses zero convergence tolerances to expend
the fixed evaluation budget; the selected plan is the best observed point,
not a certified optimum. Objective inputs are rounded to 0.000001 mm and
raw rollouts are cached by the exact rounded control values. Cached calls
still count toward the 64-evaluation budget.

The selected gap plan is replayed in its own material model to predict
the positive tool work required by each of the six squeezes. Those six
work limits are then executed in the true material. Both jaws close at
100 mm/s per jaw, starting from a 160 mm opening. At each 0.8 ms tick,

```
delta_W = control_dt * sum_over_jaws(max(-F_reaction dot v_jaw, 0))
```

The jaws stop on the first tick reaching the stage's work limit, or at
the common 65 mm minimum gap. This is positive mechanical input from
both jaws, not net plastic dissipation. The force is the material reaction
on each jaw. The controller has force feedback and commanded travel, with
no shape feedback, online identification, or replanning. The nominal work
sequence is identical for A and B. Identified and true-parameter planners
use only their own models to generate their limits. Work limits are not
adjusted using true-material outcomes. The second pass generally requires
different work limits despite repeating the same three planned gaps.
Reported gaps separate the effective contact boundaries. Each box SDF has
a 0.2 mm contact band, so the geometric box faces are 0.4 mm farther apart.

The pair is removed immediately after each squeeze, followed by 0.30 s
of free evolution. Planning and final evaluation both observe the body
1.00 s after the last release. This specifies the observation time without
claiming mechanical equilibrium. Robot approach, withdrawal, and joint
motion are not simulated. The floor and box-SDF jaws use sticky contact.

## Metrics

The planning objective is the mean of the two directed, area-weighted mean
distances between the reconstructed specimen boundary and the analytic
target surface. It is evaluated in world coordinates, without registration,
shape rescaling, or a fitted target. The calculation is implemented in
`experiments/robotics/hex_shaping_surface.py`.

The specimen surface uses the established figure reconstruction: reference
particle volume deposited on 2.5 mm voxels, Gaussian smoothing with sigma
1.3 voxels, and marching cubes at density level 0.5. This recipe is fixed
across all methods and numerical settings. Triangle centroids, weighted
by triangle areas, approximate the surface integrals. Each distance is
the absolute closest-point distance to the opposing triangle mesh, using
VTK's native implicit-distance evaluator. The analytic target is linearly
subdivided five times to give 20,480 target quadrature triangles.

Surface reconstruction introduces smoothing and discretization effects.
For reference, reconstructing the regular target particle sample itself
scores approximately 0.684 mm against the analytic target. This is a
diagnostic, not an error floor or a quantity to subtract. Refining only
the target quadrature once changes the two preceding identified work
results by less than 0.002 mm. Analytic plane/area/prism tests check units,
area weighting, subdivision invariance for constant-distance patches,
target area and volume, and zero self-distance.

The preceding metric is retained separately as `particle_error_1s_mm`:
the mean of the two directed mean unsquared nearest-neighbor distances
between the full particle clouds. The new metric is always named
`surface_error_1s_mm`. Neither replaces the meaning of the other's field.
The two metrics have different discretization effects and their numerical
values are not directly comparable.

Raw particle slice diagnostics also compare convex hulls in bands ±3 mm
around 25%, 50%, and 75% of the fixed target height with the exact target
hexagon. These are 2D cross-section IoUs, not full-body overlap. They help
check that improved reconstructed geometry corresponds to actual shape
changes. All comparisons use the same origin and orientation.

## Results

All values below use true-material executions at 1.00 s after release.

| Material | Nominal surface (mm) | Identified surface (mm) | True-parameter surface (mm) |
| --- | ---: | ---: | ---: |
| A | 6.059852 | 1.116986 | 1.107447 |
| B | 4.421554 | 1.159648 | 0.884686 |
| Mean | 5.240703 | 1.138317 | 0.996066 |

| Material | Nominal particle (mm) | Identified particle (mm) | True-parameter particle (mm) |
| --- | ---: | ---: | ---: |
| A | 2.547360 | 1.602505 | 1.595927 |
| B | 2.073557 | 1.552935 | 1.548753 |
| Mean | 2.310459 | 1.577720 | 1.572340 |

Re-evaluating the preceding work-control study with the same surface
metric gives identified errors of 1.181998 mm for A and 1.305417 mm for B.
The new plans reduce these by 5.5% and 11.2%, respectively. The nominal
errors change from 6.059771 to 6.059852 mm for A and from 4.591579 to
4.421554 mm for B. The nominal model therefore benefits from the new
search too, while the visual separation remains clear. Nominal A reaches
the 65 mm stroke limit in all six stages; both identified policies stop
at their requested work limits in every baseline and numerical execution.

This is a modest geometric improvement. The identified mean particle
error is essentially unchanged, from 1.575423 to 1.577720 mm. A's particle
error increases by about 1%, while B's decreases by about 0.75%. A also has
a local cross-section tradeoff: its upper-slice IoU falls from 94.42% to
92.63%, while its total surface error decreases. B's three slice IoUs are
95.98%, 96.96%, and 97.12%. The result should not be described as an
improvement in every metric or every cross-section.

The surface improvement holds for reconstruction voxel sizes of 2.0,
2.5, and 3.0 mm, using the same sigma in voxel units and density level.
The updated identified errors over those choices are 1.121/1.117/1.135 mm
for A and 1.177/1.160/1.224 mm for B. These checks support the direction
of the improvement without treating reconstructed boundaries as exact.

The selected model-planned gaps, repeated twice, are:

| Planner | Gap 1 (mm) | Gap 2 (mm) | Gap 3 (mm) |
| --- | ---: | ---: | ---: |
| Nominal | 73.458523 | 77.832540 | 79.910675 |
| Identified A | 71.495235 | 82.687651 | 88.155323 |
| Identified B | 79.363496 | 81.662297 | 83.216148 |
| True A | 72.165332 | 82.645011 | 88.185214 |
| True B | 79.361289 | 81.422850 | 83.284537 |

The executed work limits are:

| Planner | Work 1 (J) | 2 | 3 | 4 | 5 | 6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Nominal | 1.916951 | 0.927842 | 1.021059 | 0.760263 | 0.613531 | 0.539151 |
| Identified A | 0.504393 | 0.251774 | 0.265417 | 0.248279 | 0.194549 | 0.173981 |
| Identified B | 2.832673 | 1.468591 | 1.643468 | 1.079568 | 0.887842 | 0.781191 |
| True A | 0.498930 | 0.253041 | 0.266961 | 0.245377 | 0.194095 | 0.173459 |
| True B | 3.029851 | 1.584856 | 1.760029 | 1.153802 | 0.950562 | 0.835264 |

The true-parameter references are finite-budget planning results. The
remaining error does not measure identification error alone. This study
evaluates model-planned work limits; it does not establish that work control
outperforms position control. The earlier position-controlled nominal
results remain valid and had lower particle errors than these nominal
work-controlled outcomes.

## Numerical checks and provenance

The baseline uses a 48³ MPM grid in a 0.30 m domain, dt = 0.1 ms, two
particles per cell direction, and seed 0. Selected work budgets are also
executed at dt = 0.05 ms and on a 64³ grid, without retuning. Target geometry
is fixed; the finer grid changes initial particle sampling and volume by
about 1.3%, as in the preceding studies. The floor remains three grid cells
above the domain bottom, so the floor and target move together when the
grid changes; target dimensions stay fixed. Compare methods within each
setting. These checks do not constitute a formal convergence study.

| Setting | A nominal surface (mm) | A identified surface (mm) | B nominal surface (mm) | B identified surface (mm) |
| --- | ---: | ---: | ---: | ---: |
| 48³, 0.10 ms | 6.059852 | 1.116986 | 4.421554 | 1.159648 |
| 48³, 0.05 ms | 6.340733 | 1.039833 | 4.458705 | 1.212238 |
| 64³, 0.10 ms | 5.757019 | 2.423656 | 3.754527 | 0.912119 |

The identified advantage persists in both settings, but absolute surface
accuracy is resolution-sensitive, especially for A. The 64³ true-parameter
reference for A also has a surface error of 2.463783 mm. The full particle
errors and all reference outcomes are retained in `summary.json`.

Preparation saves the protocol, source snapshots/hashes, package versions,
GPU information, input hashes, shared targets, and copies of the preceding
manuscript and figure. All later stages check these frozen inputs. Targets
are created only in `prepare`, avoiding the preceding driver's shared-file
write race when model jobs run concurrently. Source snapshots include the
unchanged simulation, gap replay, work controller, and geometry helpers.
Identification is reproduced through the preceding study's own guide;
this study explicitly imports its frozen fitted parameters and probe data.

Every search evaluation retains the complete selected-stage geometry and
final particle positions/velocities. Calibration and execution files also
retain every stage's force, velocity, gap, incremental and cumulative work,
requested work, and stopping reason. Calibration is checked against its
gap-planning rollout. Independent self-model work replays check that the
gap-to-work conversion remains accurate. Verification recomputes both
metrics, selected-plan choices, work integrals and stopping rules, checks
finite states and zero inversions, and writes `summary.json` and
`checksums.json`.

The controller removes a jaw pair by moving it outside the domain. The
resulting pose-jump warnings are expected; the calibration replay check
compares this implementation with the planner's timed deactivation.

The first nominal and identified-A searches were briefly interrupted to
rebalance independent jobs across the two GPUs. They resumed from their
saved rollout caches with unchanged models, controls, and protocol. Initial
logs are retained. Per-model device assignments and all subsequent stage
logs are in `logs/`. No experiment was selected by device or discarded
because of its shaping result.

## Reproduction

Use the repository environment and a fresh output directory. The historical
input study is documented in `docs/hex_shaping_study.md`.
To regenerate the identification inputs from scratch, run that driver's
`prepare` stage in a fresh directory and use it as `HEX_SURFACE_SOURCE`.
Its old gap-planning and execution stages are not inputs to this new study.

```bash
HEX_SURFACE_OUT=out/hex_shaping_surface_reproduction
HEX_SURFACE_SOURCE=out/hex_shaping_study_20260908
.venv/bin/python -m experiments.robotics.hex_shaping_surface_study prepare --out "$HEX_SURFACE_OUT" --source "$HEX_SURFACE_SOURCE" --device cuda:1
.venv/bin/python -m experiments.robotics.hex_shaping_surface_run --out "$HEX_SURFACE_OUT" --devices cuda:1
.venv/bin/python -m experiments.robotics.hex_shaping_surface_study verify --out "$HEX_SURFACE_OUT"
.venv/bin/python -m experiments.robotics.hex_shaping_surface_audit --out "$HEX_SURFACE_OUT"
.venv/bin/python -m experiments.robotics.hex_shaping_surface_report --out "$HEX_SURFACE_OUT" --previous out/hex_shaping_work_20260909
.venv/bin/pytest -q tests/test_hex_shaping_surface.py tests/test_hex_shaping_work.py
.venv/bin/ruff check experiments/robotics/hex_shaping_surface.py experiments/robotics/hex_shaping_surface_study.py experiments/robotics/hex_shaping_surface_run.py experiments/robotics/hex_shaping_surface_report.py experiments/robotics/hex_shaping_surface_audit.py tests/test_hex_shaping_surface.py
```

The runner executes `plan`, `calibrate`, `selfcheck`, `execute`, and `numerics`
in that order for each model. One device argument runs pipelines sequentially.
Five device arguments run all five concurrently; the study used
`--devices cuda:1 cuda:1 cuda:1 cuda:0 cuda:0`, in model order nominal,
identified A, identified B, true A, true B. A stage can also be rerun directly
with `hex_shaping_surface_study STAGE --out ... --model ... --device ...`.
Replaying a search regenerates its optimizer bookkeeping from cached raw
rollouts. Never run two writers for the same model simultaneously.

The figure renderer defaults to a preview in the output folder. Copying to
the paper requires an explicit destination, followed by compilation and
inspection:

```bash
.venv/bin/python -m experiments.robotics.hex_shaping_surface_report --out "$HEX_SURFACE_OUT" --dest paper/icra2027/figs/identification_plastic_shaping
./paper/icra2027/build.sh
```

The four-panel figure keeps the unchanged press and held-out force panels,
shows the first three actual squeezes of A's identified work policy, and
compares the common target with true-material executions. All outcomes
share the same camera and scale. Dashed target silhouettes are annotations,
with no transformation of the simulated shapes.
