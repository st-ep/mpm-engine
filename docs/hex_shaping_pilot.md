# Hexagonal shaping feasibility pilot

Tested on 2026-09-08. The manuscript and its current shaping figure were retained.
The pilot did not produce a consistently recognizable released hexagonal prism
for both materials with the tested action sequences. This is a geometry and
contact feasibility result, not a comparison of nominal and identified planners.

A [follow-up with optimized gaps](hex_shaping_control.md) subsequently obtained
rounded hexagonal prisms for both materials by repeating three squeezes and
increasing B's stiffness. That changed-material result does not alter the
observations below or replace the paper's identification benchmark.

## What was tested

The starting particles, material laws, density, floor, and base MPM discretization
match the rectangular shaping study: a nominal 120 × 80 × 60 mm block,
E = 80 kPa, nu = 0.30, density 1000 kg/m³, and von Mises yield thresholds
1 kPa (A) and 10 kPa (B). There are 18,772 particles at 48³ resolution,
two particles per cell direction, with seed 0. Both materials start from the
same positions. The previously identified parameters are copied for provenance
but are not used in this feasibility test.

The historical shaping function was left intact. The separate
`experiments/robotics/hex_shaping_pilot.py` uses the engine's existing rotated
SDF colliders to squeeze at 0°, 60°, and 120°. The current axis-aligned `add_box`
interface cannot represent those rotated fingers. Each SDF samples the analytic
box distance at 2 mm spacing and uses sticky contact with a 0.2 mm band.
Jaw centers account for that band, so the prescribed gap is between the two
inner contact boundaries. Fingers have half-extents (12.5, 95, 80) mm in their
local frame. Their long dimension rotates with the closing direction.

The floor is sticky. Each finger moves at approximately 100 mm/s from a 160 mm
opening; the final tick duration is quantized to the 0.8 ms control period and
speed adjusted to reach the prescribed gap. The timestep is 0.1 ms.
Each pair is deactivated after closing, followed by 0.30 s of free evolution
before the next pair is applied. Snapshots include every compressed and released
state. Final outcomes are observed 0.30 s after the last release, without claiming
static equilibrium. No engine or existing study source was changed.

Four exploratory batches tested both materials, for 24 runs total:

| Batch | Gap sequences (mm) | Passes through the three directions |
|---|---|---:|
| Initial | 80/80/80, 90/90/90, 100/100/100 | 1 |
| Stronger compression | 60/60/60, 65/65/65, 70/70/70 | 1 |
| Finishing pass | 70/70/70, 80/80/80, 90/90/90 | 2 |
| Unequal gaps | 45/60/75, 55/70/85, 65/80/95 | 1 |

The initial geometry criterion was to inspect both materials and all trial sizes
for six recognizable faces after release, before comparing planners. Follow-up
batches were chosen after inspecting the preceding shapes. They are exploratory
tests, not a preregistered benchmark or independent repetitions.

## Findings and decision

Material A acquires a rounded hexagonal outline at 90 mm gaps, clearest after
two passes. Material B largely returns to a rounded rectangle under that same
sequence. Stronger equal-gap presses produce greater permanent deformation in B,
but elongate and shear A's outline as later contacts undo earlier faces.
Unequal gaps change the shear and aspect ratio; these tested sequences also fail
to give a clear common hexagonal task for both materials.

For the single-pass 80 mm case, A's x-width between its 1st and 99th particle
percentiles is 79.6010 mm under the first squeeze, 80.7500 mm after that pair is
removed, and 102.8499 mm after the full sequence. This directly shows widening
during later stages. For B in the 90 mm case, that width increases from 92.3668 mm
under the first squeeze to 108.2207 mm after release, demonstrating substantial
recovery. These widths are diagnostics, not target-shape errors.

All 24 rollouts remained finite with zero inverted particles. Four additional
rollouts repeated the six-squeeze, 90 mm sequence at half the timestep and,
separately, on a 64³ grid. The qualitative A/B distinction persists in the
middle-height cross-sections. Whole-cloud differences relative to the base run,
using the original symmetric unsquared nearest-neighbor metric in the physical
floor frame, were:

| Material | Half timestep (mm) | 64³ grid (mm) |
|---|---:|---:|
| A | 0.60288989 | 1.48326623 |
| B | 0.50497990 | 1.40285793 |

The finer grid also changes particle sampling. These are numerical sensitivity
checks, not convergence estimates, planning errors, or uncertainty bounds.
The 64³ floor is shifted by the scene builder's `3*dx`; the cloud comparison
subtracts each simulation's floor height before computing distance.

The result does not establish that a hexagonal target is unreachable. Independent
optimization of all three gaps, different contact, or another action sequence
could work. The tested variants do not yet offer a clear replacement for the
existing figure. No nominal-versus-identified planning comparison was run because
the geometry criterion was not met. No new performance claim was added to the
paper, and no source, PDF, or current figure asset was replaced.

## Reproduction and artifacts

Original outputs are under `out/hex_shaping_20260908/`. Use a fresh directory for
reproduction. From the repository root:

```bash
HEX_OUT=out/hex_shaping_reproduction
HEX_DEVICE=cuda:0
.venv/bin/python -m experiments.robotics.hex_shaping_pilot run --out "$HEX_OUT" --device "$HEX_DEVICE"
.venv/bin/python -m experiments.robotics.hex_shaping_pilot run --out "$HEX_OUT/deeper" --device "$HEX_DEVICE" --gaps-mm 60 65 70
.venv/bin/python -m experiments.robotics.hex_shaping_pilot run --out "$HEX_OUT/repeat" --device "$HEX_DEVICE" --gaps-mm 70 80 90 --cycles 2
.venv/bin/python -m experiments.robotics.hex_shaping_pilot run --out "$HEX_OUT/unequal" --device "$HEX_DEVICE" --gaps-mm 45 55 65 --ramp-mm 15
.venv/bin/python -m experiments.robotics.hex_shaping_pilot numerics --out "$HEX_OUT/numerics" --device "$HEX_DEVICE"
.venv/bin/python -m experiments.robotics.hex_shaping_pilot render --out "$HEX_OUT"
.venv/bin/python -m experiments.robotics.hex_shaping_pilot render --out "$HEX_OUT/deeper"
.venv/bin/python -m experiments.robotics.hex_shaping_pilot render --out "$HEX_OUT/repeat"
.venv/bin/python -m experiments.robotics.hex_shaping_pilot render --out "$HEX_OUT/unequal"
.venv/bin/python -m experiments.robotics.hex_shaping_report --out "$HEX_OUT"
```

The run uses the existing original identification JSON files as provenance inputs;
their values do not influence these true-material rollouts. Keep the manuscript
unchanged during a reproduction if using the report's manuscript hash check.
Each batch saves its protocol, source snapshot, source hashes, and raw trajectories.
Run batches also save package versions and GPU information. Sources evolved to add
follow-up controls and reports; their launch snapshots remain archived separately.
The physics function is unchanged across batches. Resuming with a changed source
is deliberately rejected; use a fresh output directory with the final source.
CUDA accumulation can produce small floating-point differences.

`pilot_preview.png` shows both materials under the same 90 mm gap sequence with
shared camera, scale, appearance, lighting, and surface settings.
The density surface uses the original figure's 2.5 mm voxel spacing and 1.3-voxel
smoothing; it is a visualization of the saved particles. `feasibility.png` in each
batch instead plots unsmoothed particle-slice convex hulls at three height quantiles.
These hulls are diagnostics, not full surface reconstructions. The dashed regular
hexagon is a size guide at the last closing gap, not a supplied planning target.

`numerical_cross_sections.png` compares the middle-height sections across the
three discretizations. `audit.json` stores the numerical checks and all 24 width
diagnostics. `checksums.json` records artifact hashes, and `source_snapshot_final/`
bundles the final experiment, renderer, tests, and engine Python sources.

Four focused contact-geometry tests passed, including the effective gap at all
three orientations and containment of the SDF contact band. Ruff also passed.

```bash
.venv/bin/pytest -q tests/test_hex_shaping_pilot.py
.venv/bin/ruff check experiments/robotics/hex_shaping_pilot.py experiments/robotics/hex_shaping_report.py tests/test_hex_shaping_pilot.py
```
