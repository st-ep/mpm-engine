"""Four cylindrical pinches stopped by measured positive compression work.

Model-only gap replay predicts work commands; execution observes forces and
finger travel, with a common 4 mm travel guard. No shape feedback is used.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np

from experiments.robotics import x_shaping as core

MIN_GAP = .004


def execute(law, *, budgets=None, gaps=None, grid=64, dt=.00005, device="cuda:0"):
    assert (budgets is None) != (gaps is None)
    cfg = core.CONFIG
    tick = cfg["tick"]
    substeps = round(tick / dt)
    assert abs(substeps * dt - tick) < 1e-12
    values = np.asarray(budgets if budgets is not None else gaps, float)
    assert values.shape == (4,) and np.all(np.isfinite(values)) and np.all(values > 0)
    if gaps is not None:
        assert np.all((values >= 16) & (values <= 60))
    angles = np.deg2rad([90., 0., 90., 0.])
    x, vol = core.specimen(grid)
    with contextlib.redirect_stdout(io.StringIO()):
        solver = core.Solver(core.GridConfig(n_grid=grid, grid_lim=cfg["domain"]),
                             device=device, inversion_policy="raise").load_particles(x, vol)
        solver.set_material(core.vonmises(**law, density=cfg["density"]))
    solver.add_plane(point=(0, 0, cfg["floor"]), normal=(0, 0, 1),
                     surface="separable", friction=cfg["floor_friction"])
    sdf = core.cylinder_sdf(cfg["radius"], cfg["finger_height"])
    pose = core.centers(angles[0], cfg["opening"], cfg["hover_bottom"])
    handles = [solver.add_sdf_collider(sdf, center=cp, band=cfg["band"],
               surface="separable", friction=cfg["tool_friction"]) for cp in pose]
    data = dict(initial=x, vol0=vol, floor=np.array(cfg["floor"]), n_grid=np.array(grid),
                dt=np.array(dt), angles_deg=np.rad2deg(angles))
    if budgets is not None:
        data["requested_work_j"] = values
    else:
        data["planned_gaps_mm"] = values
    fields = {key: [] for key in ["time", "tool_centers", "tool_velocity", "reaction_force",
              "work_increment_j", "phase_id", "pinch_id"]}
    phases, samples = [], []
    elapsed, current_phase, current_pinch = 0., -1, -1

    def start_phase(name):
        nonlocal current_phase
        phases.append(dict(name=name, start_s=elapsed, start_centers=pose.tolist()))
        current_phase = len(phases) - 1

    def advance(nxt):
        nonlocal pose, elapsed
        velocity = (nxt - pose) / tick
        for h, cp, v in zip(handles, pose, velocity, strict=True):
            solver.set_sdf_pose(h, center=cp, velocity=v)
            solver.reset_sdf_force(h)
        solver.step(dt, substeps=substeps)
        force = np.stack([solver.sdf_wrench(h, tick)["force"] for h in handles])
        work = float(np.maximum(-(force * velocity).sum(axis=-1), 0).sum() * tick)
        elapsed += tick
        pose = nxt
        row = dict(time=elapsed, tool_centers=pose.copy(), tool_velocity=velocity,
                   reaction_force=force, work_increment_j=work,
                   phase_id=current_phase, pinch_id=current_pinch)
        for key in fields:
            fields[key].append(row[key])
        phases[-1]["end_centers"] = pose.tolist()
        if len(fields["time"]) % 25 == 0:
            xyz = solver.x()
            samples.append(np.r_[elapsed, xyz.min(axis=0), xyz.max(axis=0)])
            if not np.all(np.isfinite(xyz)) or xyz[:, :2].min() <= .003 or xyz[:, :2].max() >= cfg["domain"] - .003:
                raise RuntimeError("Specimen leaves the interior simulation domain")
        return work

    def move(end, duration, name, angle_start=None, angle_end=None):
        start_phase(name)
        start = pose.copy()
        count = max(1, int(np.ceil(duration / tick)))
        for j in range(count):
            nxt = (start + (end - start) * (j + 1) / count if angle_start is None else
                   core.centers(angle_start + (angle_end - angle_start) * (j + 1) / count,
                                cfg["opening"], cfg["hover_bottom"]))
            advance(nxt)

    actual_gaps, work_totals, reasons = [], [], []
    for i, angle in enumerate(angles):
        if i:
            move(core.centers(angle, cfg["opening"], cfg["hover_bottom"]), 1.,
                 f"{i}:rotate", angles[i - 1], angle)
        move(core.centers(angle, cfg["opening"], cfg["finger_bottom"]),
             (cfg["hover_bottom"] - cfg["finger_bottom"]) / cfg["vertical_speed"], f"{i}:approach")
        start_phase(f"{i}:close")
        current_pinch = i
        gap, total = cfg["opening"], 0.
        end_gap = MIN_GAP if budgets is not None else values[i] / 1000
        while gap > end_gap + 1e-12:
            gap = max(end_gap, gap - 2 * cfg["finger_speed"] * tick)
            total += advance(core.centers(angle, gap, cfg["finger_bottom"]))
            if budgets is not None and total >= values[i]:
                break
        reasons.append("gap_prediction" if budgets is None else
                       ("work" if total >= values[i] else "travel_guard"))
        work_totals.append(total)
        actual_gaps.append(gap * 1000)
        data[f"stage_{i}_pressed"] = solver.x().copy()
        data[f"stage_{i}_centers"] = pose.copy()
        data[f"stage_{i}_pressed_time_s"] = np.array(elapsed)
        current_pinch = -1
        move(core.centers(angle, cfg["opening"], cfg["finger_bottom"]),
             (cfg["opening"] - gap) / (2 * cfg["finger_speed"]), f"{i}:open")
        move(core.centers(angle, cfg["opening"], cfg["hover_bottom"]),
             (cfg["hover_bottom"] - cfg["finger_bottom"]) / cfg["vertical_speed"], f"{i}:withdraw")
        data[f"stage_{i}_released"] = solver.x().copy()
    data["x_after_withdrawal"] = solver.x().copy()
    move(pose.copy(), cfg["release_s"], "final:wait")
    data.update(x_after_1s=solver.x().copy(), v_after_1s=solver.v().copy(),
                inverted_count=np.array(solver.inverted_count()), gaps_mm=np.array(actual_gaps),
                actual_work_j=np.array(work_totals), extent_samples=np.array(samples))
    for key, values in fields.items():
        data[key] = np.array(values)
    for i, phase in enumerate(phases):
        phase["duration_s"] = (phases[i + 1]["start_s"] if i + 1 < len(phases) else elapsed) - phase["start_s"]
    assert int(data["inverted_count"]) == 0 and all(np.all(np.isfinite(v)) for v in data.values())
    return data, phases, reasons
