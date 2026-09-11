# Identification for work-limited hexagonal shaping

This is the archived 32-evaluation work-control candidate. The active paper
now uses [surface-objective planning with 64 evaluations](hex_shaping_surface.md).
The controller and all data recorded here remain unchanged.

This experiment obtains a clearer nominal-versus-identified difference by
changing the squeeze stopping rule. It retains the current paper's material
properties, common analytic target, nominal model, six squeeze directions,
closing speed, and identification results. It is separate from the preceding
negative six-independent-gap test with original B at E = 80 kPa.

The manuscript was unchanged when this candidate was completed. Its four-panel figure
is saved as `out/hex_shaping_work_20260909/identification_plastic_shaping_work.pdf`
and `.png`. The larger comparison is `work_control_comparison.png`.

## Control and information

Both jaws close at 100 mm/s per jaw. The controller accumulates the positive
mechanical work supplied by the two jaws and stops each squeeze at a
model-predicted work limit. For each 0.8 ms control tick it adds

```
delta_W = dt_control * sum_over_jaws(max(-F_reaction dot v_jaw, 0))
```

The force is the material's reaction on the jaw, measured from the SDF
collider's grid impulse. The minus sign converts it to power supplied to
the material. Both jaws contribute; their forces must not be added as
vectors before computing work, since opposing forces would cancel.
The work is positive tool input, not a measurement of plastic dissipation.
The jaws translate during each squeeze, so no rotational power term is
needed. Measurement is noise-free in this simulation.

The six work limits are computed before execution. Directions are
0°, 60°, 120°, 0°, 60°, 120°. Each pair starts at a 160 mm opening.
The pair is removed after stopping, followed by 0.30 s of free evolution.
Reported outcomes are observed 1.00 s after final release. These are
specified observation times, not a claim of mechanical equilibrium.

Every method shares a 65 mm minimum jaw gap. A volume-matched 65 mm
hexagon would be approximately 157 mm high, within the 160 mm jaw height;
this sets a common stroke bound for the forming apparatus. The controller
stops at that bound if it cannot reach its work limit. The bound was fixed
before testing actual outcomes. Nominal A reaches it in all six squeezes;
this saturation is part of the reported result and is labeled in the figure.
The bound does not activate for either identified baseline execution.

There is force feedback during shaping, with jaw travel supplied by the
commanded motion. There is no shape feedback, material-parameter feedback,
online re-identification, or replanning. In particular, this is no longer
the previous experiment's open-loop position sequence. The earlier probe
still identifies the material from supplied simulation states and does
not use reaction-force or stress labels. Force is used separately by the
shaping controller.

## Planning and reused evidence

The five equally budgeted gap plans come from
`out/hex_shaping_study_20260908/`: nominal, identified A/B, and true A/B.
Each used the same initialization and 32 objective evaluations against
the common analytic hexagon, 90 mm across flats and 81.666695 mm high.
The loss was symmetric mean unsquared nearest-neighbor distance between
full 3D particle clouds, scored 0.30 s after final release. Every sampled
gap in these five searches is at least 65 mm.

For each selected plan, a new rollout in that planner's own material model
measures the work of each squeeze. Those six predicted work values become
its control parameters. They are not adjusted using true-material outcomes.
The shared nominal work sequence is applied unchanged to A and B. This is
a two-stage planning procedure, not direct optimization over six independent
work variables. The two passes generally have different work limits even
though the preceding geometric plan repeated the same three gaps.

The work limits, in joules, are:

| Planner | Squeeze 1 | 2 | 3 | 4 | 5 | 6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Nominal | 1.920442 | 0.907692 | 0.933148 | 0.744731 | 0.566315 | 0.510725 |
| Identified A | 0.473039 | 0.238400 | 0.256482 | 0.225959 | 0.180760 | 0.164813 |
| Identified B | 2.784240 | 1.424708 | 1.570553 | 1.048952 | 0.849969 | 0.746023 |

A has E = 80 kPa and yield = 1 kPa; B has E = 240 kPa and yield = 10 kPa.
Both have nu = 0.30 and density 1000 kg/m³. The nominal model remains
E = 80 kPa, nu = 0.30, yield = 5.5 kPa. The original identified parameters
and held-out press predictions are reused because neither material changes.
Both materials use the same known constitutive family. This does not test
selection between constitutive families.

`prepare` verifies the preceding study's simulation/source hashes, input
laws, search budgets, and sampled gap bounds. It copies the selected plans
and model parameters into the new folder, recording their original paths
and hashes. The five new work-measurement rollouts reproduce the preceding
gap-plan trajectories within 0.05 mm. Independent work-controlled replays
in their respective planning models recover all six gaps to within
0.134 mm, less than one 0.16 mm closing tick. Their final full-cloud
differences from the gap-controlled replays are below 0.100 mm.

## Results

Each set of work limits is independently executed using the true material.
The following full-cloud errors are measured 1.00 s after release:

| Material | Nominal (mm) | Identified (mm) | True-parameter reference (mm) |
| --- | ---: | ---: | ---: |
| A | 2.547362 | 1.586158 | 1.585998 |
| B | 2.120451 | 1.564688 | 1.545075 |
| Mean | 2.333906 | 1.575423 | 1.565536 |

The nominal work allowance overcompresses A until the stroke bound stops
the jaws. Its final cross-section is visibly distorted. For B, the nominal
limits stop closing earlier, leaving a wider, shorter body than intended.
The identified limits retain recognizable hexagonal profiles for both.

At half the target height, convex-hull cross-section overlap with the target
is 71.66% nominal versus 97.23% identified for A, and 84.84% versus 96.66%
for B. These are two-dimensional diagnostics of raw particle slices within
±3 mm of that fixed height, not whole-body accuracy. No rigid alignment or
rescaling is used. Three heights are plotted in `work_control_profiles.png`.

The previous position-controlled nominal plans had smaller errors than
these work-controlled nominal plans (1.673 mm for A, 1.715 mm for B).
That earlier evidence remains valid. This experiment demonstrates the
importance of the material model for planning effort-limited squeezes;
it does not show that work control dominates position control, or that
identification is necessary for every controller that can form a hexagon.

## Numerical and implementation checks

All six work policies are replayed without retuning their work limits at a
halved timestep and on a 64³ grid. The controller, force measurements, and
common stroke bound are identical across methods within each setting.

| Setting | A nominal (mm) | A identified (mm) | B nominal (mm) | B identified (mm) |
| --- | ---: | ---: | ---: | ---: |
| 48³, 0.10 ms | 2.547362 | 1.586158 | 2.120451 | 1.564688 |
| 48³, 0.05 ms | 2.660258 | 1.581060 | 2.137135 | 1.569905 |
| 64³, 0.10 ms | 2.193829 | 1.382099 | 1.581147 | 1.207894 |

Both checks preserve the identified improvement for both materials. Grid
refinement changes particle/target sampling and slightly changes initial
particle volume, as in the preceding study; compare methods within each row.
These checks are not a formal convergence study.

Verification recomputed all 168 stage work integrals in 28 complete new
rollouts: five work-prediction rollouts, five self-model checks, six baseline
executions, and twelve numerical executions. All states are finite with
zero inverted particles. All 18 execution errors and all work/stroke stop
labels were checked against the raw traces. Saved jaw centers reproduce
the reported gaps, and saved compressed/released states remain clear of
the lateral and upper domain boundaries. Imported source-study data match
their previously archived hashes. Sixteen focused tests and Ruff pass.

The baseline uses a 48³ grid, a 0.30 m domain, dt = 0.1 ms, two particles
per cell direction, and seed 0. Geometry, SDF/contact settings, and surface
rendering match the previous study. Removing each pair by moving it outside
the domain triggers a pose-jump warning; this implements the same idealized
instantaneous removal as the previous timed deactivation. The direct gap
replay checks confirm matching released trajectories. Robot approach,
withdrawal, and joint trajectories are not simulated.

## Reproduction and artifacts

The driver is `experiments/robotics/hex_shaping_work.py`; rendering is in
`experiments/robotics/hex_shaping_work_report.py`. Outputs are in
`out/hex_shaping_work_20260909/`. The folder retains work predictions, all
force/velocity/work traces, actual stops, independent executions, numerical
checks, source snapshots, imported-input hashes, environment, figures, and
verification results. The candidate's press panels reuse the unchanged
source-study data; their exact input paths and hashes are recorded separately.

Use a fresh output directory and the completed previous study as input.
If needed, reproduce that input study using `docs/hex_shaping_study.md` first.
Run these stages sequentially, since they write shared target artifacts:

```bash
HEX_WORK_OUT=out/hex_shaping_work_reproduction
HEX_WORK_SOURCE=out/hex_shaping_study_20260908
.venv/bin/python -m experiments.robotics.hex_shaping_work prepare --out "$HEX_WORK_OUT" --source "$HEX_WORK_SOURCE" --device cuda:1
for HEX_WORK_MODEL in nominal identified_A identified_B true_A true_B; do
  .venv/bin/python -m experiments.robotics.hex_shaping_work calibrate --out "$HEX_WORK_OUT" --model "$HEX_WORK_MODEL" --device cuda:1
  .venv/bin/python -m experiments.robotics.hex_shaping_work selfcheck --out "$HEX_WORK_OUT" --model "$HEX_WORK_MODEL" --device cuda:1
  .venv/bin/python -m experiments.robotics.hex_shaping_work execute --out "$HEX_WORK_OUT" --model "$HEX_WORK_MODEL" --device cuda:1
  .venv/bin/python -m experiments.robotics.hex_shaping_work numerics --out "$HEX_WORK_OUT" --model "$HEX_WORK_MODEL" --device cuda:1
done
.venv/bin/python -m experiments.robotics.hex_shaping_work_report --out "$HEX_WORK_OUT"
.venv/bin/python -m experiments.robotics.hex_shaping_work verify --out "$HEX_WORK_OUT"
.venv/bin/pytest -q tests/test_hex_shaping_work.py tests/test_hex_shaping_six_gaps.py tests/test_hex_shaping_control.py tests/test_hex_shaping_pilot.py tests/test_plastic_shaping_study.py
.venv/bin/ruff check experiments/robotics/hex_shaping_work.py experiments/robotics/hex_shaping_work_report.py tests/test_hex_shaping_work.py
```

An initial parallel numerical launch collided while writing a shared target
file, before A's identified numerical simulations began. That command was
rerun serially; both the failed launch and successful retry logs are retained.
The simulations and work budgets were not changed for the retry.
