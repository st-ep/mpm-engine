# Rounded X target informed by the material A pilot

This document records the earlier target-design step. The approved target is
now evaluated for both materials in `docs/x_letter_study.md`, which documents
the current paper figure and its reproduction.

The selected proposal is a symmetric rounded capital X, 64 × 80 mm in plan,
with a 28 mm waist and a uniform extrusion height of 24.946587 mm. Its volume
is exactly 90 mL, matching the original 60 × 60 × 25 mm specimen. The material,
specimen, tool motions and saved simulation results are unchanged in this
design comparison. The active paper is unchanged.

The earlier target demanded long, straight, tapering outer edges. The observed
object has fuller curved sides, blunter ends and a narrower top/bottom notch.
The revised outline represents those features with simple geometric curves.
It preserves a smooth ideal shape and a flat top; it does not reproduce the
surface ridges, small asymmetries or numerical artifacts of the simulation.

## Geometry and source map

- `experiments/robotics/x_letter_soft_target.py` defines three proposals and
  selects `slimmer` through `SELECTED`. No particle positions enter this
  geometry generator.
- `out/x_letter_target_redesign_20260909/target.json` records the selected
  parameters, volume, dimensions, source hash, mesh hash and source rollouts.
- `target.vtp` in that directory is the selected closed triangle mesh.
- `target_outline_m.npy` is its CCW footprint, in metres relative to the
  object center. The mesh is centered at x = y = 80 mm, with its bottom at
  z = 10 mm, as in the existing simulation.
- `selected_source/` records the selected generator and figure source.
- `comparison.json`, `source_snapshot/` and `compare.py` retain the exploratory
  comparison, before selecting the final proposal.
- `revised_target.png` compares the previous target, revised target and the
  unchanged fine-grid material A result at a common scale.

One quadrant is formed by a cubic curve from the waist to the outer arm,
a straight outer segment, a 3 mm circular corner, a short horizontal end,
and a cubic curve to the top notch. Reflection supplies the other quadrants.
The curvature radius at the side waist is 14 mm; at the bottom of the upper
notch it is 6 mm. These are endpoint curvature radii of the cubic curves,
not constant circular radii over their entire lengths. The side radius
matches the cylinder radius; the different upper/lower notch shape is informed
by the deformation remaining after the final side pinch.

The implementation checks reflection symmetry, positive area, exact mesh
volume and closedness. Publication additionally checks connectedness, doubled
curve sampling and refined surface-distance quadrature. The top remains
planar. Thus a remaining 3D error is expected from the uneven released surface.

## What the comparison establishes

These targets were designed **after observing the result**. The numbers below
measure geometric fit to one unchanged simulation, not improved control or a
new identification result. All proposals conserve the same reference volume.

| Target | Mean surface distance (mm) | Top-view IoU |
| --- | ---: | ---: |
| Previous target | 2.052917 | 74.38% |
| Gentle rounded proposal | 1.314140 | 92.06% |
| Fuller rounded proposal | 1.211594 | 95.70% |
| Selected, slightly slimmer proposal | 1.047731 | 93.80% |

The selected proposal gives the closest full 3D match while retaining clear
arm separation and high footprint overlap. The fuller proposal has slightly
higher footprint overlap but a larger 3D mismatch. The selected geometry also
fits the two coarser numerical replays: 1.107089 mm / 90.86% at 64³ and
dt = 50 microseconds, and 1.101715 mm / 92.20% at dt = 25 microseconds.
These replays are not independent tests of generalization to new materials.

The reference fine-grid rollout is
`out/x_letter_A_refine_20260909/rollouts/8d4624c8e588bb9d.npz`.
It uses true material A parameters, 28 mm cylindrical fingers, and the four
previously selected gaps: [20.090210, 30.726318, 34.580444, 20.448608] mm in
y, x, y, x order. The reported state is 1 s after full withdrawal. There is
no alignment, rescaling, particle clipping or modification of these data.
The distance definition and reconstruction settings remain those documented
in `docs/x_letter_A_pilot.md`.

Future planning and comparisons should freeze this revised target before
optimizing any commands and use the same target for every material/model.
No new control search was run as part of this target-design step. Earlier
paper and pilot results continue to refer to their original targets.

## Reproduction

From the repository root, using the recorded environment:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m out.x_letter_target_redesign_20260909.compare
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m out.x_letter_target_redesign_20260909.publish
.venv/bin/ruff check experiments/robotics/x_letter_soft_target.py
```

The comparison verifies the frozen source of the original study and reads its
archived NPZ files. `publish.py` verifies the selected comparison values and
records the final target and figure provenance. Keep the original study's
source manifest and rollouts when reproducing this design comparison.
