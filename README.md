# warpmpm

A fast, modular **Warp-MPM** engine for robot manipulation of **deformable and granular
media** — dough first, terrain / rovers next. Built to couple to **MuJoCo now** and
**NVIDIA Isaac Lab later** (Warp ↔ PyTorch is zero-copy on CUDA, so this is the
Isaac-native path), with a **sparse active-block grid** and an **implicit** quasi-static
solver for speed.

## Design

```
robot sim (MuJoCo / Isaac) ──link poses, velocities──▶ warpmpm
                            ◀──per-link 6D wrench, contact obs──
```
The MPM owns the material; the robot sim owns the robot; the coupling exchanges only
compact wrenches (never particles). Core is the warp-mpm fork (12 materials incl.
Newtonian/Bingham dough, mu(I) sand) wrapped behind a stable backend API.

- `core/` — CPU-default `Solver` wrapping the warp-mpm fork; `GridConfig`.
- `materials/` — dough (Newtonian/Bingham) and granular presets.
- `colliders/` — primitive-SDF moving contact (box/capsule/sphere) = robot end-effector.
- `coupling/` — `WarpMPMBackend` (set_robot_kinematics / step / get_link_wrenches), wrench readout.
- `adapters/` — MuJoCo (now), Isaac Lab (later, same contract).
- `render/` — particle + robot rendering.
- `tests/`, `benchmarks/` — conservation, contact-sign, wrench (Newton's 3rd law); ms/step.

## Quickstart

```bash
uv pip install -e ".[dev,mujoco,render]"
python benchmarks/bench_step.py     # baseline ms/step
pytest                              # conservation + sanity
ruff check . && ty check            # lint + types
```

The MPM fork is reached via `src/warpmpm/_vendor.py`; set `WARPMPM_FORK` to override its path.

## Roadmap

0. Package + explicit dense baseline + benchmark + tests  ← here
1. Primitive-SDF moving colliders + `set_robot_kinematics`
2. Per-link wrench readout (Newton's third law) + squeeze cross-validation
3. MuJoCo arm + dough manipulation demo + render
4. Sparse active-block grid (GPUMPM) + implicit Newton-CG (GeoWarp) — the fast version
5. Learned constitutive residual (trainable seam)
6. Terrain / rover navigation on the same core
