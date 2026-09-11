# Cross-execution of the identified A and B shaping plans

Historical position-control comparison. Figure 3 now uses work commands;
see [the current protocol](x_work_study.md). The results below belong to the earlier
position trajectories and remain unchanged.

The comparison asks whether the plan should use the identified model of the
specimen being manipulated. Both previously optimized plans are executed on
both true materials. The diagonal pairs use the matching identified model;
the off-diagonal pairs use the other specimen's identified model. This is a
test of model mismatch between these two specimens, not a comparison against
all generic material assumptions or a demonstration of family selection.

## Frozen inputs and execution

The original study is `out/x_letter_study_20260909/`, documented in
`docs/x_letter_study.md`. The cross-execution archive is
`out/x_letter_cross_20260909/`. Its runner is
`experiments/robotics/x_letter_cross.py`; geometry checks and figure generation
use `experiments/robotics/x_letter_cross_report.py` and the shared
`x_letter_study_report.py`. The shared renderer preserves the latest pressing
scene, held-out force plot, and four-pinch illustration.

No new optimization was performed. The original identified models, selected
commands, target, initial specimen, contact, speeds, observation delay, and
surface metric are fixed. Both model searches had the same initial simplex,
gap bounds, and 32 objective evaluations. The target and common initialization
were informed by earlier true-A feasibility work before those searches. See
the original protocol for that qualification and the supplied-state fitting
assumptions.

| Planning model | First y gap | First x gap | Second y gap | Second x gap |
| --- | ---: | ---: | ---: | ---: |
| Identified A | 18.046539 | 24.657501 | 38.702850 | 26.408508 |
| Identified B | 23.616293 | 18.208131 | 43.478840 | 16.000000 |

Gaps are physical cylinder-surface openings in mm. Each plan retains the full
four-pinch trajectory, including opening, withdrawal, and raised quarter
turns. The other specimen's plan is applied unchanged, without force or shape
feedback. Its complete tool positions, times, and phase records are checked
against the original matched execution. The existing sampled Panda checks
therefore apply to the same commands; no full-arm dynamics or hardware
execution is newly validated.

The target is the same rounded X, 64 by 80 mm in plan, with 90 mL volume.
Initial specimens are 60 by 60 by 25 mm. Both true materials have Hencky
elasticity with E = 80 kPa and nu = 0.30, and von Mises yield thresholds of
1 kPa (A) and 10 kPa (B). Planning uses their separately identified parameters,
not the true values. The 28 mm cylinders and contact implementation are
unchanged.

Six new runs comprise both cross-executions at the original 64³ grid and
50 microsecond timestep, at half the timestep, and on an 80³ grid. The six
matched executions are copied byte-for-byte from the original study, including
their configurations and phase records. Fresh runs use seed 0 and GPU 0.
Two independent processes run concurrently on that GPU. No other job was
stopped, and no timing comparison is claimed.

The metric remains symmetric area-weighted mean closest-triangle surface
distance, in mm, 1 s after full withdrawal. Reference-volume reconstruction
uses 1.25 mm voxels, Gaussian sigma 1.625 mm, and isovalue 0.5. There is no
registration, geometry scaling, particle clipping, or subtraction of surface
reconstruction bias. Additional audits vary the reconstruction voxel size,
refine target quadrature, and measure silhouette overlap and floor penetration.

## Results

The recorded 3D surface errors (mm) are:

| Setting | A, planned for A | A, planned for B | B, planned for A | B, planned for B |
| --- | ---: | ---: | ---: | ---: |
| 64³, 50 microseconds | 1.045335 | 1.192527 | 1.892299 | 1.312977 |
| 64³, 25 microseconds | 1.045445 | 1.211244 | 1.884757 | 1.310257 |
| 80³, 50 microseconds | 0.930919 | 1.184871 | 1.750232 | 1.271931 |

At baseline, matching the planning model reduces error by 12.34% for A and
30.61% for B relative to using the other material's plan. The mean across the
two materials is 1.179156 mm with matching plans and 1.542413 mm with exchanged
plans. The reductions are 13.69% and 30.48% at half the timestep, and 21.43%
and 27.33% on the finer grid. These are numerical sensitivity checks, not a
convergence proof or statistical evidence across materials and trials.

At baseline, A's top-view silhouette IoU is 89.14% with its own plan and
91.74% with B's plan. For B it is 83.62% with its own plan and 78.67% with
A's plan. Thus the matched model improves the 3D surface metric for both
materials, but not every view or metric. The figure retains a common camera,
scale, and target contour without attempting to conceal this distinction.

For context, the original shared nominal plan gives 1.219872 mm for A and
1.418501 mm for B at baseline. Its errors at half the timestep are 1.228266
and 1.408035 mm, and on the finer grid 1.173754 and 1.284778 mm. Matching the
identified model is therefore only 1.00% better than nominal for B on the
finer grid. The larger benefit against the other material's plan must not be
presented as a larger benefit against nominal.

All twelve outcomes pass finite-state, inversion, fixed-volume, command,
source, and input checks. Their surfaces are single connected components and
fit within the common top and oblique render views. The matched-model 3D
ordering holds for 1.0, 1.25, and 1.5 mm reconstruction voxels in all numerical
settings; the smallest margin is 0.128070 mm. Finer target quadrature changes
an error by at most 0.003359 mm.

The maximum pressed height is 34.119962 mm, below the cylinders' 45.5 mm
working height. Raised turns have zero recorded force. Peak sampled force
per finger is 11.271104 N. Grid contact permits up to 1.511977 mm particle
penetration below the floor; at most 2.411024% of particles are below it,
and the maximum volume-weighted mean penetration is 0.012758 mm. These
observations do not establish exact contact or full hardware feasibility.

## Reproduction commands

From the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_letter_cross prepare
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_letter_cross evaluate --material A --device cuda:0
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_letter_cross evaluate --material B --device cuda:0
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_letter_cross verify
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_letter_cross_report audit
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_letter_cross_report report --dest paper/icra2027/figs/identification_plastic_shaping
./paper/icra2027/build.sh
```

For a fresh repetition, use the same new `--out` directory in every command.
Preparation requires a fresh directory. The two evaluation commands may run
concurrently because their outputs are disjoint. Existing outputs can resume
only with matching source, input hashes, commands, and configurations. Skip
preparation and evaluation to verify or render the archived results.

`source_snapshot/` freezes the runner and original physics sources;
`source_sha256.json`, `input_sha256.json`, and `reused_sha256.json` prevent
silent changes to sources, selected plans, target, or matched results.
`summary.json` records all twelve outcomes. `additional_audit.json` records
the geometry and sensitivity checks. `report_source/` and
`figure_provenance.json` map the paper figure to exact sources and inputs.

The original nominal-plan results remain archived and are retained for context.
The separate nominal force prediction in panel (b) is unchanged. Table II and
its unresolved timing scope are unrelated and must remain unchanged.

## Paper adoption

The active `paper/icra2027/paper.pdf` was rebuilt and remains eight pages.
Pages 2 and 6–8 were visually inspected. Panels (a), (b), and (c) are
pixel-identical to the preceding figure. The source changes are confined to
the shaping comparison, its caption, and the stale introduction reference to
two pinches, corrected to four. Table II, Sections III/IV, and the unrelated
tracked changes are unchanged. The original study and both earlier rendering
archives still pass all of their recorded checksums.

Ruff passes for the new runner, audit wrapper, and shared renderer. Compilation
retains the existing bibliography warnings, table overfull box, and float
underfull boxes. The inspected pages have no overlapping or clipped content.
`adoption.json`, `manuscript.diff`, `manuscript_before/`, `manuscript_after/`,
and the rendered pages record the adoption; `checksums.json` freezes the
completed archive.
