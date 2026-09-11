# Separate forming and finishing gaps

This exploratory follow-up restores B's original E = 80 kPa and yield =
10 kPa, while A remains E = 80 kPa and yield = 1 kPa. It asks whether
independent settings for the two passes can retain a recognizable hexagon
and make the difference between nominal and identified plans clearer.
The current manuscript uses the preceding study with E = 240 kPa for B.

## Fixed setup

The common analytic target remains 90 mm across flats and
81.6666950594 mm high, with the baseline initial particle volume. The
initial block, density, Poisson's ratio, grid, time step, SDF geometry,
contact, jaw speed, and release durations are unchanged. The six closing
directions remain 0°, 60°, 120°, 0°, 60°, and 120°. Only the tying of the
first and second triplets is removed: all six final gaps are independent.

In the first setup, every planner starts at 80 / 85 / 87 / 80 / 85 / 87 mm. The initial simplex
subtracts 10 mm from one coordinate at a time. Bounded Nelder-Mead uses
40–115 mm bounds and a limit of 64 objective evaluations, twice the
previous budget for twice as many variables. The smallest observed objective
selects a plan; convergence or global optimality is not assumed. The loss
is still symmetric mean unsquared nearest-neighbor distance between full
3D particle clouds, scored 0.30 s after final release. The independent
executions are observed 1.0 s after release.

The shared nominal law is unchanged: E = 80 kPa, nu = 0.30, yield = 5.5 kPa.
Both materials still have nu = 0.30 and density 1000 kg/m³. Identification
uses new 14 mm simulated presses with supplied noise-free x, v, elastic F,
mass, and reference volume, under the same known constitutive family.
It does not use force or stress labels. The original B parameters are
restored explicitly; the fit for E = 240 kPa is not reused.

## Checks and assessment

Before optimization, the explicit six-gap implementation executes
67 / 73 / 75 / 67 / 73 / 75 mm for original B and compares every released
stage against the existing repeated-triplet implementation. The final
full-cloud replay difference is 0.00041289 mm, below the 0.05 mm check
tolerance. Every stage is checked, not only the last state.

The staged feasibility check first optimizes the true B model and the
unchanged nominal model using the same budget and initial simplex. A good
true-parameter hexagon is a prerequisite for adopting this experiment.
The remaining identified/true-model searches and numerical checks are
needed for a complete replacement of the current paper comparison.
A partial check must not be described as a full five-model benchmark.

Shape quality is assessed from both whole-body surfaces and particle-slice
convex hulls within ±3 mm of 25%, 50%, and 75% of the fixed target height.
Slice intersection-over-union alone cannot distinguish a good hexagon from
a similarly sized round body. A supplementary diagnostic compares mean
radial extent at the six target-corner directions with mean radial extent
at the six face-normal directions, about the fixed target center. The ratio
is 1 for a centered circle and 2/sqrt(3) = 1.15470054 for an exact regular
hexagon. Its analytic cases have dedicated tests. It is not the optimization
loss, a full 3D accuracy measure, or a replacement for visual inspection.

The experiment source is `experiments/robotics/hex_shaping_six_gaps.py`;
the renderer and geometric diagnostics are in `hex_shaping_six_report.py`.
Outputs are under `out/hex_shaping_six_gaps_20260909/`. Each setup records
its protocol, source snapshots and hashes, environment, GPU, fitted laws,
all trial trajectories, selected actions, independent executions, and
verification results. Existing study and manuscript files are not overwritten.

## Reproduction

Run from the repository root, with a fresh output directory. The two
independent plan commands may run concurrently:

```bash
HEX_SIX_OUT=out/hex_shaping_six_gaps_reproduction
HEX_SIX_DEVICE=cuda:1
.venv/bin/python -m experiments.robotics.hex_shaping_six_gaps prepare --out "$HEX_SIX_OUT" --device "$HEX_SIX_DEVICE"
.venv/bin/python -m experiments.robotics.hex_shaping_six_gaps plan --out "$HEX_SIX_OUT" --device "$HEX_SIX_DEVICE" --model true_B
.venv/bin/python -m experiments.robotics.hex_shaping_six_gaps plan --out "$HEX_SIX_OUT" --device "$HEX_SIX_DEVICE" --model nominal
.venv/bin/python -m experiments.robotics.hex_shaping_six_gaps execute --out "$HEX_SIX_OUT" --device "$HEX_SIX_DEVICE" --model true_B
.venv/bin/python -m experiments.robotics.hex_shaping_six_gaps execute --out "$HEX_SIX_OUT" --device "$HEX_SIX_DEVICE" --model nominal
.venv/bin/python -m experiments.robotics.hex_shaping_six_gaps numerics --out "$HEX_SIX_OUT" --device "$HEX_SIX_DEVICE" --model true_B
.venv/bin/python -m experiments.robotics.hex_shaping_six_gaps numerics --out "$HEX_SIX_OUT" --device "$HEX_SIX_DEVICE" --model nominal
.venv/bin/python -m experiments.robotics.hex_shaping_six_report --out "$HEX_SIX_OUT" --materials B --methods nominal oracle
.venv/bin/python -m experiments.robotics.hex_shaping_six_gaps verify --out "$HEX_SIX_OUT"
```

For a complete comparison, also plan and execute `identified_A`,
`identified_B`, and `true_A`, then run `numerics` for all five models.
All models use the same protocol and evaluation limit. The verification
report states explicitly whether the data form a complete or partial
comparison. GPU floating-point reductions may change the optimizer path
slightly across reproductions; selected plans are independently replayed.

### Deeper common initialization

A second exploratory run changes only the common starting sequence to
67 / 73 / 75 / 67 / 73 / 75 mm. This follows an earlier, reasonably sized
original-B result. Both the nominal and true-B planners receive this same
initialization and the same 64-evaluation limit. The nominal model, target,
objective, and all other settings remain fixed. This checks sensitivity to
initialization without selecting different starting points for each model.

The output folder is `out/hex_shaping_six_gaps_20260909/deeper_start/`.
Its archived `run.py` changes the initialization and delegates all stages to
the same six-gap driver. The launcher itself is included in its source
snapshot and source checks. Its `reused_inputs.json` records that the two
raw identification probes and two compatibility trajectories were copied
from the first run, with source paths and hashes. Identification is
recomputed from these raw probes.

To reproduce this follow-up with fresh probes, copy the archived launcher
into a fresh folder at the same depth and invoke it without `--out`:

```bash
HEX_SIX_DEEPER=out/hex_shaping_six_gaps_reproduction/deeper_start
mkdir -p "$HEX_SIX_DEEPER"
cp out/hex_shaping_six_gaps_20260909/deeper_start/run.py "$HEX_SIX_DEEPER/run.py"
.venv/bin/python "$HEX_SIX_DEEPER/run.py" prepare --device cuda:0
.venv/bin/python "$HEX_SIX_DEEPER/run.py" plan --model true_B --device cuda:0
.venv/bin/python "$HEX_SIX_DEEPER/run.py" plan --model nominal --device cuda:0
.venv/bin/python "$HEX_SIX_DEEPER/run.py" execute --model true_B --device cuda:0
.venv/bin/python "$HEX_SIX_DEEPER/run.py" execute --model nominal --device cuda:0
.venv/bin/python -m experiments.robotics.hex_shaping_six_report --out "$HEX_SIX_DEEPER" --materials B --methods nominal oracle
.venv/bin/python "$HEX_SIX_DEEPER/run.py" verify
```

The report also saves raw slice outlines under the last squeeze, at
0.30 s after release, and at 1.00 s after release. All slices use the same
physical heights relative to the analytic target.

## First initialization: completed results

Both searches used all 64 objective evaluations and stopped at the budget
limit. The nominal planner selected 67.913 / 81.946 / 81.804 /
76.852 / 80.071 / 82.031 mm. The true-B planner selected
59.155 / 69.860 / 74.963 / 76.663 / 97.984 / 82.241 mm.
These are best-observed controls, not certified optima.

The following errors are independent executions on original material B,
observed 1.00 s after release:

| Numerical setting | Nominal plan (mm) | True-parameter plan (mm) |
| --- | ---: | ---: |
| 48³ grid, 0.10 ms timestep | 1.875065 | 1.635422 |
| 48³ grid, 0.05 ms timestep | 1.827851 | 1.650582 |
| 64³ grid, 0.10 ms timestep | 1.546451 | 1.263434 |

The same selected controls are replayed for the numerical checks; they
are not reoptimized. Compare methods within each row. Grid refinement
also changes the particle sampling and therefore the discretization
contribution to the nearest-neighbor error.

Although the known-law plan has lower position error, its cross-sections
remain rounded. At baseline resolution, its corner-to-face ratios are
1.0370 / 1.0398 / 1.0308 at the three sampled heights; the ideal hexagon
has ratio 1.1547. The corresponding ratios remain approximately
1.030–1.037 with a halved timestep and 1.033–1.034 on the finer grid.
The released surface and raw slice outlines both show this limitation.
The final compressed state is also rounded, so the last release alone
does not account for the missing facets.

The observation at 1.00 s should not be called fully settled. For example,
the baseline known-law replay still has RMS particle speed 56.0 mm/s.
All comparisons use the same specified observation time. The independent
baseline replay differs from its selected search trajectory by
0.00045347 mm at 0.30 s.

`geometry_checks.json` records the geometric diagnostics for every selected
numerical replay and all completed true-model search trials. These are
post-search checks; they do not change the loss or selected controls.

## Deeper initialization: completed results

Both searches again used all 64 evaluations. The selected nominal gaps
are 78.756 / 69.986 / 82.893 / 72.652 / 75.785 / 80.997 mm, and
the selected true-B gaps are 61.009 / 79.240 / 77.455 /
68.551 / 72.803 / 74.422 mm.

On original B at 1.00 s, the independently executed nominal and true-model
plans have errors of 1.818055 and 1.609119 mm, respectively. The latter
still has rounded cross-sections, with corner-to-face ratios
1.0476 / 1.0473 / 1.0323. Its independent replay differs from the selected
search trajectory by 0.00045321 mm at 0.30 s. This second initialization
received baseline replays only; the timestep and grid checks above apply
to the first initialization's controls.

## Decision and scope

Do not replace the paper experiment with this variant. Neither shared
initialization produced a convincing released hexagon for original B,
even when planning with its exact material parameters. The lower average
particle error does not establish good facet or corner recovery. This is
a negative result for these searches and this action/objective setup,
not a proof that six-gap shaping of B is impossible.

The completed coverage is two nominal/true-B searches per initialization,
256 total objective evaluations, six independent baseline executions
(including nominal controls on A), and six numerical replays for the first
initialization. Identification was recomputed for both original materials,
but the identified-model planning searches were not run after the
true-parameter shape-quality check failed. These results must therefore
remain labeled as a partial feasibility study, not an identified-model
comparison. The active manuscript and figure are unchanged.

If pursuing this further, a separate experiment could use a loss that
directly measures the surface shape, keeping the target and nominal law
fixed. This is a hypothesis for future testing. It is not implemented or
validated by the present trials, and cannot be claimed to solve the issue.

## Validation commands

```bash
.venv/bin/pytest -q tests/test_hex_shaping_six_gaps.py tests/test_hex_shaping_control.py tests/test_hex_shaping_pilot.py tests/test_plastic_shaping_study.py
.venv/bin/ruff check experiments/robotics/hex_shaping_six_gaps.py experiments/robotics/hex_shaping_six_report.py tests/test_hex_shaping_six_gaps.py
```
