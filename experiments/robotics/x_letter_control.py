"""Law-parameterized rounded-X control: four alternating y/x pinches.

Derived from the archived A feasibility executor. The raised rotation takes
one second and alternates 90/0 degrees for the Panda wrist. Only the two
cylinder contacts enter MPM; arm dynamics are not simulated.
"""

import contextlib
import io
import numpy as np
from experiments.robotics import x_shaping as core


def execute(law, gaps, grid=64, dt=0.00005, device="cuda:0"):
    cfg = core.CONFIG
    tick = cfg["tick"]
    substeps = round(tick / dt)
    assert abs(substeps * dt - tick) < 1e-12
    gaps = np.asarray(gaps, float) / 1000
    assert 1 <= len(gaps) <= 8 and np.all((gaps >= 0.016) & (gaps <= 0.060))
    angles = np.deg2rad(np.tile([90.0, 0.0], (len(gaps) + 1) // 2)[: len(gaps)])
    x, vol = core.specimen(grid)
    with contextlib.redirect_stdout(io.StringIO()):
        solver = core.Solver(
            core.GridConfig(n_grid=grid, grid_lim=cfg["domain"]),
            device=device,
            inversion_policy="raise",
        ).load_particles(x, vol)
        solver.set_material(core.vonmises(**law, density=cfg["density"]))
    solver.add_plane(
        point=(0, 0, cfg["floor"]),
        normal=(0, 0, 1),
        surface="separable",
        friction=cfg["floor_friction"],
    )
    sdf = core.cylinder_sdf(cfg["radius"], cfg["finger_height"])
    pose = core.centers(angles[0], cfg["opening"], cfg["hover_bottom"])
    handles = [
        solver.add_sdf_collider(
            sdf, center=c, band=cfg["band"], surface="separable", friction=cfg["tool_friction"]
        )
        for c in pose
    ]
    data = dict(
        initial=x,
        vol0=vol,
        floor=np.array(cfg["floor"]),
        n_grid=np.array(grid),
        dt=np.array(dt),
        gaps_mm=gaps * 1000,
        angles_deg=np.rad2deg(angles),
    )
    elapsed = 0.0
    frames = []
    ids = []
    forces = []
    times = []
    phases = []

    def move(end, duration, name, angle_start=None, angle_end=None):
        nonlocal pose, elapsed
        start = pose.copy()
        count = max(1, int(np.ceil(duration / tick)))
        phases.append(
            dict(
                name=name,
                start_s=elapsed,
                duration_s=count * tick,
                start_centers=start.tolist(),
                end_centers=end.tolist(),
            )
        )
        for j in range(count):
            nxt = (
                start + (end - start) * (j + 1) / count
                if angle_start is None
                else core.centers(
                    angle_start + (angle_end - angle_start) * (j + 1) / count,
                    cfg["opening"],
                    cfg["hover_bottom"],
                )
            )
            velocity = (nxt - pose) / tick
            for handle, c, v in zip(handles, pose, velocity, strict=True):
                solver.set_sdf_pose(handle, center=c, velocity=v)
                solver.reset_sdf_force(handle)
            solver.step(dt, substeps=substeps)
            forces.append(np.stack([solver.sdf_wrench(h, tick)["force"] for h in handles]))
            frames.append(nxt.copy())
            times.append(elapsed + tick)
            ids.append(len(phases) - 1)
            pose = nxt
            elapsed += tick

    for i, (gap, angle) in enumerate(zip(gaps, angles, strict=True)):
        if i:
            move(
                core.centers(angle, cfg["opening"], cfg["hover_bottom"]),
                abs(angle - angles[i - 1]) / (np.pi / 2),
                f"{i}:rotate",
                angles[i - 1],
                angle,
            )
        move(
            core.centers(angle, cfg["opening"], cfg["finger_bottom"]),
            (cfg["hover_bottom"] - cfg["finger_bottom"]) / cfg["vertical_speed"],
            f"{i}:approach",
        )
        move(
            core.centers(angle, gap, cfg["finger_bottom"]),
            (cfg["opening"] - gap) / (2 * cfg["finger_speed"]),
            f"{i}:close",
        )
        data[f"stage_{i}_pressed"] = solver.x().copy()
        data[f"stage_{i}_centers"] = pose.copy()
        move(
            core.centers(angle, cfg["opening"], cfg["finger_bottom"]),
            (cfg["opening"] - gap) / (2 * cfg["finger_speed"]),
            f"{i}:open",
        )
        move(
            core.centers(angle, cfg["opening"], cfg["hover_bottom"]),
            (cfg["hover_bottom"] - cfg["finger_bottom"]) / cfg["vertical_speed"],
            f"{i}:withdraw",
        )
        data[f"stage_{i}_released"] = solver.x().copy()
    data["x_after_withdrawal"] = solver.x().copy()
    move(pose.copy(), cfg["release_s"], "final:wait")
    data.update(
        x_after_1s=solver.x().copy(),
        v_after_1s=solver.v().copy(),
        inverted_count=np.array(solver.inverted_count()),
        time=np.array(times),
        tool_centers=np.array(frames),
        reaction_force=np.array(forces),
        phase_id=np.array(ids),
    )
    assert int(data["inverted_count"]) == 0 and all(np.all(np.isfinite(v)) for v in data.values())
    return data, phases
