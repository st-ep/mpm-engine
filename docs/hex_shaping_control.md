# Hexagonal shaping with three gap settings

Tested on 2026-09-08. Six sequential squeezes produce recognizable, rounded
hexagonal prisms for both materials when B's Young's modulus is increased from
80 to 240 kPa. A is unchanged. This is a known-parameter feasibility experiment;
it initially left the paper's shaping study unchanged. The subsequent
[identification and planning study](hex_shaping_study.md) completes the comparison
for revised B and supplies the active manuscript figure.

## Working recipe

Use two flat jaws to squeeze along 0°, 60°, and 120°, and repeat that sequence
once. The closing direction is the normal to the jaw face in the horizontal
plane. Only three final jaw gaps are optimized for each material; the second
pass reuses them. Round the selected gaps to whole millimetres for execution:

| Material | Young's modulus | von Mises yield threshold | Gaps at 0° / 60° / 120° |
|---|---:|---:|---|
| A, unchanged | 80 kPa | 1 kPa | 74 / 84 / 89 mm |
| B, revised stiffness | 240 kPa | 10 kPa | 80 / 81 / 85 mm |

Both materials use Poisson's ratio 0.30 and density 1000 kg/m³. Increasing B's
stiffness at the same yield threshold reduces the elastic strain needed to
reach yielding. The tested B then retains clearer facets after release.
Its original 80 kPa / 10 kPa version remained more rounded in the tested
three- and six-squeeze searches. Changing stiffness is an exploratory choice,
not a claim about an experimentally measured material.

Each pair starts at a 160 mm opening and each jaw moves at approximately
100 mm/s. The closing duration is quantized to the 0.8 ms control period;
velocity is adjusted slightly to reach the prescribed gap. The jaws are
deactivated immediately after closing, with 0.30 s of free evolution before
the next squeeze. The final shape is shown 1.0 s after the last release.
There is no dwell under compression or continuous withdrawal trajectory.
Jaw activation/deactivation and changes of orientation idealize tool handling.
This is open-loop kinematic control in simulation, not force control, robot
joint control, feedback replanning, or a hardware result.

## Common target and simulation

The target is an analytic regular hexagonal prism, 90 mm across flats and
81.6666950594 mm high. Its volume, 0.0005728760037726488 m³, equals the sum of
initial particle volumes at the baseline discretization. It is fixed before
optimization, independent of either material's rollout, and is the same for
both materials. Its center is fixed at x = y = 150 mm, with its bottom at the
floor. There is no outcome-dependent alignment, rotation, or rescaling.

The initial block, nominally 120 × 80 × 60 mm, uses the original study's scene
builder with seed 0 and two particles per cell direction. The baseline has
18,772 particles, a 48³ grid in a 0.30 m domain, and a 0.1 ms timestep.
Contact with the floor and jaws is sticky. The jaws use the previous pilot's
rotated box SDFs, with half-extents 12.5 / 95 / 80 mm, 2 mm SDF sampling, and
a 0.2 mm contact band. The commanded gaps account for that band.

The target is sampled throughout its volume. The objective is half the sum
of the two mean nearest-neighbor distances between the full simulated and
target particle clouds, in millimetres. Distances are unsquared. Optimization
uses the released state at 0.30 s; persistence to 1.0 s is checked separately.
This metric measures a sampled volume discrepancy, not maximum surface error.

## Search and selection

Bounded Nelder-Mead searches three gaps in [40, 115] mm. The initial simplex
subtracts 10 mm from each coordinate in turn. All evaluated trajectories are
saved. The lowest observed objective selects a plan, which is then executed
again. The budgets below are fixed evaluation limits; the reported settings
are the best found, without claiming global optimality or optimizer convergence.

| Output directory | Material change | Passes | Initial gaps (mm) | Objective evaluations |
|---|---|---:|---|---:|
| `three_A` | None | 1 | 75 / 85 / 90 | 44 |
| `three_B` | None | 1 | 65 / 70 / 75 | 44 |
| `six_A` | None | 2 | 80 / 85 / 87 | 32 |
| `six_B` | None | 2 | 65 / 75 / 75 | 32 |
| `B_yield5_screen` | Yield = 5 kPa | 2 | 80 / 85 / 87 | 1 |
| `B_yield3_screen` | Yield = 3 kPa | 2 | 80 / 85 / 87 | 1 |
| `B_E240_screen` | E = 240 kPa | 2 | 80 / 85 / 87 | 1 |
| `six_B_E240` | E = 240 kPa | 2 | 80 / 85 / 87 | 24 |

The one-evaluation cases are property screens at a specified action, not
optimized comparisons. E = 80 kPa / yield = 3 kPa also produced a clearer
hexagon for B. The selected E = 240 kPa / yield = 10 kPa option retains the
original tenfold yield-threshold contrast between materials. These choices
followed inspection of previous outcomes; they are exploratory, not an
independent validation of a prespecified material choice.

The selected unrounded gaps are 73.767063 / 84.335863 / 88.879993 mm for A and
80.095986 / 81.294403 / 84.683578 mm for revised B. Independent execution with
the whole-millimetre controls above confirms that this precision is unnecessary.

## Checks and limits

All values below use the rounded controls, with the same target geometry and
no reoptimization for the numerical checks. Errors are measured 1.0 s after
the last release:

| Material | 48³, 0.1 ms error (mm) | 48³, 0.05 ms error (mm) | 64³, 0.1 ms error (mm) |
|---|---:|---:|---:|
| A | 1.58647393 | 1.57651707 | 1.42666185 |
| B, E = 240 kPa | 1.56074619 | 1.55349076 | 1.19423905 |

The whole-cloud change between 0.30 and 1.0 s after release is 0.06783765 mm
for A and 0.16180875 mm for B. Final RMS particle speeds are 0.00091100 m/s
and 0.01208699 m/s, respectively, so the experiment establishes persistence
of the shape over that interval, not exact static equilibrium. Independent
replays of the unrounded plans differ by 0.00031005 mm for A and 0.00036623 mm
for B. All verification rollouts remain finite with zero inverted particles.

The finer grid changes the particle sampling and initial volume to
0.0005804386289476327 m³. The analytic target dimensions remain fixed, but
its point sampling is refined. Thus these are numerical sensitivity checks,
not a convergence study or an estimate of discretization uncertainty.

For an additional outline diagnostic, convex hulls of particles within ±3 mm
of 25%, 50%, and 75% of the fixed target height are intersected with the analytic
hexagon. Their mean intersection-over-union is 0.95570503 for A and 0.95877691
for B at the baseline settings. These three slice hulls conceal local concavity
and do not measure full 3D accuracy. The resulting bodies have rounded corners
and some variation along their height, as the surface previews show.

All 179 search objectives were recomputed from saved particle data. Seven
focused tests passed, covering rotated contact geometry, target dimensions
and frame, and the outline diagnostic. Ruff passed for the new modules/tests.
The original study's 653 archived scientific artifacts and the active TeX,
PDF, and shaping figure remain unchanged.

Planning here uses the supplied true parameters. Adopting the revised B in the
paper required new identification, held-out validation, and nominal versus
identified planning on the common target; the linked follow-up performs these
steps. The original B identification and error values cannot be transferred
to this changed material. Both materials still belong to the same von Mises
constitutive family; this experiment does not test family selection.

## Reproduction

Run from the repository root using the existing `.venv` and a CUDA device.
Use a fresh directory; launch-source checks intentionally reject resuming a
case with changed source or settings. CUDA reductions can introduce small
floating-point differences, including in the optimization path.

The following reproduces the two selected searches, their independent and
rounded-control checks, and the final plots:

```bash
HEX_CONTROL_OUT=out/hex_shaping_control_reproduction
HEX_CONTROL_DEVICE=cuda:0
.venv/bin/python -m experiments.robotics.hex_shaping_control optimize --out "$HEX_CONTROL_OUT/six_A" --material A --device "$HEX_CONTROL_DEVICE" --cycles 2 --initial-mm 80 85 87 --max-evals 32
.venv/bin/python -m experiments.robotics.hex_shaping_control optimize --out "$HEX_CONTROL_OUT/six_B_E240" --material B --device "$HEX_CONTROL_DEVICE" --cycles 2 --initial-mm 80 85 87 --max-evals 24 --young-kpa 240
.venv/bin/python -m experiments.robotics.hex_shaping_check --plan "$HEX_CONTROL_OUT/six_A" --out "$HEX_CONTROL_OUT/check_A" --device "$HEX_CONTROL_DEVICE"
.venv/bin/python -m experiments.robotics.hex_shaping_check --plan "$HEX_CONTROL_OUT/six_B_E240" --out "$HEX_CONTROL_OUT/check_B_E240" --device "$HEX_CONTROL_DEVICE"
.venv/bin/python -m experiments.robotics.hex_shaping_control_report --out "$HEX_CONTROL_OUT"
```

To reproduce the full exploratory record, also run the following before the
report command. These cases are independent of the selected searches:

```bash
.venv/bin/python -m experiments.robotics.hex_shaping_control optimize --out "$HEX_CONTROL_OUT/three_A" --material A --device "$HEX_CONTROL_DEVICE" --cycles 1 --initial-mm 75 85 90 --max-evals 44
.venv/bin/python -m experiments.robotics.hex_shaping_control optimize --out "$HEX_CONTROL_OUT/three_B" --material B --device "$HEX_CONTROL_DEVICE" --cycles 1 --initial-mm 65 70 75 --max-evals 44
.venv/bin/python -m experiments.robotics.hex_shaping_control optimize --out "$HEX_CONTROL_OUT/six_B" --material B --device "$HEX_CONTROL_DEVICE" --cycles 2 --initial-mm 65 75 75 --max-evals 32
.venv/bin/python -m experiments.robotics.hex_shaping_control optimize --out "$HEX_CONTROL_OUT/B_yield5_screen" --material B --device "$HEX_CONTROL_DEVICE" --cycles 2 --initial-mm 80 85 87 --max-evals 1 --yield-kpa 5
.venv/bin/python -m experiments.robotics.hex_shaping_control optimize --out "$HEX_CONTROL_OUT/B_yield3_screen" --material B --device "$HEX_CONTROL_DEVICE" --cycles 2 --initial-mm 80 85 87 --max-evals 1 --yield-kpa 3
.venv/bin/python -m experiments.robotics.hex_shaping_control optimize --out "$HEX_CONTROL_OUT/B_E240_screen" --material B --device "$HEX_CONTROL_DEVICE" --cycles 2 --initial-mm 80 85 87 --max-evals 1 --young-kpa 240
```

To replay just the reported rounded recipe and numerical checks without
searching again, run `hex_shaping_check` with `--plan` pointing to the archived
`out/hex_shaping_control_20260908/six_A` or `six_B_E240` directory and `--out`
pointing to a fresh `check_A` or `check_B_E240` directory. Then run the report
on their shared parent directory. The exact-plan replay also checks agreement
with the archived selected execution.

```bash
.venv/bin/pytest -q tests/test_hex_shaping_control.py tests/test_hex_shaping_pilot.py
.venv/bin/ruff check experiments/robotics/hex_shaping_control.py experiments/robotics/hex_shaping_check.py experiments/robotics/hex_shaping_control_report.py tests/test_hex_shaping_control.py
```

## Artifacts and visualization

`out/hex_shaping_control_20260908/` contains all searches, per-stage compressed
and released snapshots, selected plans, independent executions, and numerical
checks. Each launch archives its protocol, engine and experiment sources,
source hashes, installed package versions, Git revision, and GPU information.
The final report saves `summary.json`, a `report_source/` bundle, and recursive
artifact checksums. An optional `manuscript_before.json` records the manuscript
state at the start of this experiment; a fresh reproduction does not need it.

`hexagonal_shapes.png` and `.pdf` show the exact analytic target and the two
rounded-control outcomes 1.0 s after release. The target uses a flat-shaded
analytic mesh; the simulated outcomes use the original study's density-surface
renderer, with 2.5 mm voxel spacing and 1.3-voxel smoothing. Camera, scale,
floor frame, and lighting are shared. No particles are moved to improve the
appearance. Surface smoothing rounds visible edges, so
`hexagonal_profiles.png` also shows unsmoothed particle-slice convex hulls at
the fixed target mid-height across the three numerical settings.
