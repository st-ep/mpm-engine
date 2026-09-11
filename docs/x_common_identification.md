# Shared cylindrical identification and X shaping

Current status: the common identification and held-out force prediction remain active. The compression-work shaping results below are historical; the active force-profile replacement is documented in `docs/x_force_profile_study.md` and `paper/evidence/x-force-profile-study.md`. Its execution entrypoint is `experiments/robotics/x_force_profile_gains.py` and its figure entrypoint is `x_force_profile_report.py`. The original archived data and numbers below are preserved.

This experiment replaces the disconnected flat-plate calibration and gentle
force-pulse illustration in Figure 3. The active source is
`experiments/robotics/x_common_study.py`; the figure entrypoint is
`experiments/robotics/x_common_report.py`. Earlier plate, force-pulse and work
studies remain separate historical archives.

## Identification

Both initially identical 60 × 60 × 25 mm specimens receive exactly the same
prescribed finger trajectory. Two 28 mm diameter vertical cylinders lower at
72 mm opening, wait 0.20 s, close to 32 mm at 20 mm/s per finger, hold for
0.10 s, reopen at the same speed, and withdraw. Released shapes are observed
1 s after withdrawal. The full A/B pose, velocity and time arrays match exactly.
This is position control during calibration, not force regulation.

The material definitions are unchanged: Hencky elasticity with E = 80 kPa and
nu = 0.30, and von Mises yield thresholds of 1 kPa (A) and 10 kPa (B).
The 32 mm probe was the first tested common position trajectory in this revision.
It activated plasticity in both materials without modifying either material.

The estimator receives noise-free particle positions, velocities and elastic F,
reference volumes and masses. It uses closing frames at 4 ms spacing, takes
every second frame, and assembles 26-frame temporal windows. Elastic moduli
come from the Hencky weak balance. Conservative inward planes at the deepest
finger tangencies and at the floor exclude contact support with a two-cell
margin. Yield is estimated from the elastic deviatoric Hencky-strain cap, using
the existing 99.9th percentile, 5% occupancy and 3× concentration criteria.
Hold and release frames do not enter the fit.

| Parameter | True A | Identified A | True B | Identified B |
|---|---:|---:|---:|---:|
| E (kPa) | 80 | 79.923904 | 80 | 80.015650 |
| nu | 0.30 | 0.300265 | 0.30 | 0.299983 |
| Yield threshold (kPa) | 1 | 0.998837 | 10 | 10.002038 |

The accepted strain-cap fractions are 30.56% and 11.96%; concentrations are
50.54 and 30.72. The independent simulator audit also checks outgoing trial
elastic strains against the true yield caps. It finds continued yield activation
in both materials, including 37.14% of B's particles crossing the cap in at
least one sampled outgoing trial. This audit uses simulator truth and is kept
outside the estimator. A plateau alone is not a general proof of yielding.

No stress, reaction-force, true-parameter or trial-strain arrays enter fitting.
`observation_isolation.json` records a rerun with only the declared observation
channels and geometry metadata. It must reproduce the stored parameter values.
Geometry, density, contact and the Hencky/von Mises family are known. This does
not demonstrate identification of a constitutive family or recovery of elastic
F from camera observations.

## Independent validation and shaping

A held-out cylindrical squeeze closes to 24 mm at 30 mm/s per finger. Its
closing, holding and opening reaction forces are excluded from all fitting.
Force error is the relative L2 norm of both fingers' complete 3D reaction
vectors over those phases, without alignment or force fitting. Matching errors
are 0.113505% for A and 0.020090% for B. Exchanging the fitted models gives
750.8498% and 88.7278%, respectively. The figure plots mean outward force per
finger, whereas the scalar error uses both full reaction vectors.

Identification, validation and shaping use fresh specimens of the same material.
Shaping is not continued from the deformed calibration specimen. Each new fit
receives a fresh, 32-evaluation bounded Nelder–Mead search for four final
openings, alternating y, x, y, x. The common initialization [20, 31, 35, 20] mm,
16–54 mm bounds, −3 mm initial simplex offset and rounded X target are unchanged.
The target and initialization were informed by earlier A feasibility work.
Only the identified model enters its search; true execution outcomes do not.

Each model's selected gap trajectory is replayed at 50 mm/s per finger to
predict four positive compression-work commands. The work controller sums
max(−reaction force · finger velocity, 0) over both fingers at 4 ms intervals.
It stops closing on reaching a command or the common 12 mm opening guard.
The work contains elastic storage, motion and dissipation; it is not a plastic
dissipation measurement. There is no shape feedback or online replanning.
Both command vectors are executed in both true materials, with baseline,
half-timestep and finer-grid checks. Commands remain fixed during those checks.

| Numerical setting | A, plan A | A, plan B | B, plan A | B, plan B |
|---|---:|---:|---:|---:|
| 64³, 50 μs | 1.043612 | 1.503309 | 2.895978 | 1.312329 |
| 64³, 25 μs | 1.036745 | 1.549209 | 2.895896 | 1.303831 |
| 80³, 50 μs | 0.950194 | 1.514705 | 2.821217 | 1.314937 |

These are mean 3D surface errors in millimetres. Matching is better in every
setting, including when reconstructing surfaces at 1, 1.25 and 1.5 mm voxels.
Both matching executions reach every work command. B's plan on A reaches the
12 mm opening guard on pinches 1, 2 and 4 in all settings.

All baseline surfaces are connected. The finer-grid A execution under B's
plan has two reconstructed components, with 99.608% of surface area in the
largest; no components are removed before scoring. Maximum sampled floor
penetration across the executions is 1.574 mm. The maximum work overshoot
is 8.568%, within the first threshold-crossing control tick. These diagnostics
are retained with the results rather than hidden by rendering or clipping.

## Reproduction

The archives are `out/x_common_probe32_pilot_20260910` and
`out/x_common_identification_study_20260910`. They contain source snapshots,
input hashes, package versions, particle observations, all planning evaluations,
commands and execution histories. The study also bundles the target and Panda
model used for rendering and kinematic checks. Source and input hashes are
checked before each study stage; use a separate output directory for changes.

Run from the repository root with its `.venv`. To repeat the probe in a fresh
directory, execute the following for each material (A on cuda:0, B on cuda:1
can run concurrently):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_common_probe record --out out/repeated_common_probe --material A --device cuda:0
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_common_probe fit --out out/repeated_common_probe --material A
```

After both fits pass, prepare a new study:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_common_study prepare --out out/repeated_common_study --pilot out/repeated_common_probe
```

For each material, run `validate`, `plan`, `predict`, and `selfcheck` in that
order, with `--out out/repeated_common_study --material A --device cuda:0`.
The material pipelines can run concurrently on different GPUs. After both
finish, run `evaluate` for each material; it includes all three numerical
settings by default. Finally run:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_common_study verify --out out/repeated_common_study
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_common_report audit --out out/repeated_common_study
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_common_report diagnostics --out out/repeated_common_study
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.robotics.x_common_report report --out out/repeated_common_study --render-out out/repeated_common_study/figure
```

Baseline settings are 64³ grid, 50 μs timestep, 46,080 particles, seed 0,
90 mL volume and 1000 kg/m³ density. Numerical checks use 25 μs or an 80³
grid, with fixed physical geometry and volume. Surface reconstruction uses
1.25 mm voxels, Gaussian sigma 1.625 mm, and isovalue 0.5. The symmetric,
area-weighted 3D distance uses the full target and reconstructed surface,
without alignment, resizing, clipping or subtracting a baseline error.

Repeated GPU reductions need not be bitwise identical. Compare parameter
recovery, forces, stopping decisions and shape errors as well as source hashes.
The archived arrays establish the exact provenance of the published figure.

The adopted figure is
`out/x_compact_layout_figure_20260910/figure/identification_plastic_shaping.pdf`
and is copied to `paper/icra2027/figs/`. This presentation revision uses the
same archived simulations: panel (d) has Target, Matched model and Swapped model
columns, with true materials A and B as rows. The matching model is always in
the middle column, and the other material's identified model is on the right.
Muted blue and warm orange identify actual materials A and B in every specimen
view, including when the planning model is swapped. Labels and force curves use
darker shades of the same hues; the target and gripper remain neutral. The
caption explains that material colors are visualization labels. The earlier
beige matched/swapped layout remains in `out/x_matched_swapped_figure_20260910`.
The earlier equal-row color layout remains in `out/x_material_color_figure_20260910`.
The compact layout reduces the height of panels (a) and (b), uses one heading
band per panel, and places the force legend inside unused plot space. Secondary
protocol details are in the caption. Released A/B use one horizontal margin
crop; the target and all four outcomes use a common crop of the union of their
projected surfaces, with 10 pixels of padding. The cameras, shading and source
rasters are unchanged. No geometry is removed or independently resized.

At the same paper width, the figure is 13.3% shorter. Specimen linear dimensions
increase by 37.2% in (a), 41.2% in (c), and 59.8% in (d). All three images in
(a) retain equal display heights, with labels outside the image band; released
A/B share a physical scale. The action retains its previous relative scale.
`layout_verification.json` records these measurements, exact raster comparisons,
unchanged scientific records, and verification of the 620-file study and 61-file
color-figure archives. The original study figure remains in its frozen archive. 
The active manuscript is
`paper/icra2027/paper.tex`; `./paper/icra2027/build.sh` rebuilds its PDF.
The eight-page build and affected pages were inspected after adoption. The
existing seven missing bibliography entries and the 0.6606 pt overfull equation
warning are unchanged. Table II and its timing scope are unchanged.

The Panda hand and proposed adapters are a kinematic illustration. Sampled IK,
joint range and speed checks do not validate actuator dynamics, full-arm
collision avoidance or hardware execution. Grid-contact penetration is also
audited; a timestep/grid sensitivity check does not establish convergence.
