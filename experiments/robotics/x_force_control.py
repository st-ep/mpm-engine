"""Four force-command pinches using the qualified single-pinch PI controller.

The initial specimen and contacts are unchanged. Only the common force-pulse
amplitudes are planning variables; gaps emerge from measured force feedback.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np

from experiments.robotics import x_shaping as core
from experiments.robotics.x_pinch_force_pilot import CONTROL, pulse


def execute(law, peaks, grid=64, dt=0.00005, device="cuda:0"):
    cfg, c = core.CONFIG, CONTROL
    peaks = np.asarray(peaks, float)
    assert 1 <= len(peaks) <= 4 and np.all((peaks >= 0) & (peaks <= 14))
    angles = np.deg2rad(np.tile([90., 0.], 2)[:len(peaks)])
    tick = cfg["tick"]
    substeps = round(tick / dt)
    assert abs(substeps * dt - tick) < 1e-12
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
    data = dict(initial=x, vol0=vol, floor=np.array(cfg["floor"]),
                n_grid=np.array(grid), dt=np.array(dt), peaks_n=peaks,
                angles_deg=np.rad2deg(angles))
    fields = {key: [] for key in ["time", "tool_centers", "tool_velocity", "reaction_force",
              "force_per_finger", "target_force", "filtered_force", "force_error",
              "integral_velocity", "command_velocity", "speed_limited", "acceleration_limited",
              "travel_limited", "phase_id", "pulse_id"]}
    phases = []
    elapsed, filtered = 0., 0.
    alpha = 1 - np.exp(-tick / c["filter_s"])
    current_phase, current_pulse = -1, -1
    samples = []

    def start_phase(name):
        nonlocal current_phase
        phases.append(dict(name=name, start_s=elapsed, start_centers=pose.tolist()))
        current_phase = len(phases) - 1

    def advance(nxt, target=0., error=0., integral=0., command=0., speed=False,
                acceleration=False, travel=False):
        nonlocal pose, elapsed, filtered
        velocity = (nxt - pose) / tick
        for h, cp, v in zip(handles, pose, velocity, strict=True):
            solver.set_sdf_pose(h, center=cp, velocity=v)
            solver.reset_sdf_force(h)
        solver.step(dt, substeps=substeps)
        force = np.stack([solver.sdf_wrench(h, tick)["force"] for h in handles])
        normal = nxt[1] - nxt[0]
        normal /= np.linalg.norm(normal)
        compressive = (force @ normal) * np.array([-1., 1.])
        filtered += alpha * (compressive.mean() - filtered)
        elapsed += tick
        pose = nxt
        row = dict(time=elapsed, tool_centers=pose.copy(), tool_velocity=velocity,
                   reaction_force=force, force_per_finger=compressive,
                   target_force=target, filtered_force=filtered, force_error=error,
                   integral_velocity=integral, command_velocity=command,
                   speed_limited=speed, acceleration_limited=acceleration,
                   travel_limited=travel, phase_id=current_phase, pulse_id=current_pulse)
        for key in fields:
            fields[key].append(row[key])
        phases[-1]["end_centers"] = pose.tolist()
        if len(fields["time"]) % 25 == 0:
            xyz = solver.x()
            samples.append(np.r_[elapsed, xyz.min(axis=0), xyz.max(axis=0)])
            if not np.all(np.isfinite(xyz)) or xyz[:, :2].min() <= .003 or xyz[:, :2].max() >= cfg["domain"] - .003:
                raise RuntimeError("Specimen leaves the interior simulation domain")

    def move(end, duration, name, angle_start=None, angle_end=None):
        start_phase(name)
        start = pose.copy()
        count = max(1, int(np.ceil(duration / tick)))
        for j in range(count):
            nxt = (start + (end - start) * (j + 1) / count if angle_start is None else
                   core.centers(angle_start + (angle_end - angle_start) * (j + 1) / count,
                                cfg["opening"], cfg["hover_bottom"]))
            advance(nxt)

    min_gaps = []
    for i, (peak, angle) in enumerate(zip(peaks, angles, strict=True)):
        if i:
            move(core.centers(angle, cfg["opening"], cfg["hover_bottom"]), 1.,
                 f"{i}:rotate", angles[i - 1], angle)
        move(core.centers(angle, cfg["opening"], cfg["finger_bottom"]),
             (cfg["hover_bottom"] - cfg["finger_bottom"]) / cfg["vertical_speed"], f"{i}:lower")
        move(core.centers(angle, c["start_gap_m"], cfg["finger_bottom"]),
             (cfg["opening"] - c["start_gap_m"]) / (2 * c["touch_speed_m_s"]), f"{i}:engage")
        move(pose.copy(), c["pre_pulse_wait_s"], f"{i}:pre_pulse_wait")
        data[f"stage_{i}_before_pulse"] = solver.x().copy()
        start_phase(f"{i}:force_pulse")
        current_pulse = i
        duration = c["rise_s"] + c["hold_s"] + c["fall_s"]
        integral, last_velocity = 0., 0.
        minimum_gap = c["max_gap_m"]
        for j in range(round(duration / tick)):
            target = pulse((j + 0.5) * tick, peak)
            error = target - filtered
            trial_integral = integral + c["ki_m_s2_n"] * error * tick
            raw = c["kp_m_s_n"] * error + trial_integral
            v = np.clip(raw, -c["max_speed_m_s"], c["max_speed_m_s"])
            speed_limited = abs(raw - v) > 1e-12
            va = np.clip(v, last_velocity - c["max_acceleration_m_s2"] * tick,
                         last_velocity + c["max_acceleration_m_s2"] * tick)
            acceleration_limited = abs(va - v) > 1e-12
            gap = np.linalg.norm(pose[1] - pose[0]) - 2 * cfg["radius"]
            next_gap = np.clip(gap - 2 * va * tick, c["min_gap_m"], c["max_gap_m"])
            actual_v = (gap - next_gap) / (2 * tick)
            travel_limited = abs(actual_v - va) > 1e-12
            if (not (speed_limited or acceleration_limited or travel_limited)
                    or error * (raw - actual_v) <= 0):
                integral = trial_integral
            advance(core.centers(angle, next_gap, cfg["finger_bottom"]),
                    target, error, integral, raw, speed_limited, acceleration_limited, travel_limited)
            last_velocity = actual_v
            if next_gap < minimum_gap:
                minimum_gap = next_gap
                data[f"stage_{i}_pressed"] = solver.x().copy()
                data[f"stage_{i}_centers"] = pose.copy()
                data[f"stage_{i}_pressed_time_s"] = np.array(elapsed)
        min_gaps.append(minimum_gap * 1000)
        data[f"stage_{i}_end_pulse"] = solver.x().copy()
        current_pulse = -1
        gap = np.linalg.norm(pose[1] - pose[0]) - 2 * cfg["radius"]
        move(core.centers(angle, cfg["opening"], cfg["finger_bottom"]),
             abs(cfg["opening"] - gap) / (2 * cfg["finger_speed"]), f"{i}:open")
        move(core.centers(angle, cfg["opening"], cfg["hover_bottom"]),
             (cfg["hover_bottom"] - cfg["finger_bottom"]) / cfg["vertical_speed"], f"{i}:withdraw")
        data[f"stage_{i}_released"] = solver.x().copy()
    data["x_after_withdrawal"] = solver.x().copy()
    move(pose.copy(), cfg["release_s"], "final:wait")
    data.update(x_after_1s=solver.x().copy(), v_after_1s=solver.v().copy(),
                inverted_count=np.array(solver.inverted_count()),
                min_gaps_mm=np.array(min_gaps), extent_samples=np.array(samples))
    for key, values in fields.items():
        data[key] = np.array(values)
    for i, phase in enumerate(phases):
        phase["duration_s"] = (phases[i + 1]["start_s"] if i + 1 < len(phases) else elapsed) - phase["start_s"]
    assert int(data["inverted_count"]) == 0
    assert all(np.all(np.isfinite(v)) for v in data.values())
    return data, phases


def diagnostics(data, phases):
    result = []
    for i, peak in enumerate(data["peaks_n"]):
        selected = data["pulse_id"] == i
        phase = next(p for p in phases if p["name"] == f"{i}:force_pulse")
        t = data["time"] - phase["start_s"]
        plateau = selected & (t > .6) & (t <= 1.6)
        f = data["force_per_finger"].mean(axis=1)
        result.append(dict(
            peak_command_n=float(peak), min_gap_mm=float(data["min_gaps_mm"][i]),
            mean_plateau_force_n=float(f[plateau].mean()),
            plateau_rmse_n=float(np.sqrt(np.mean((f[plateau] - peak) ** 2))),
            whole_pulse_mae_n=float(np.mean(abs(f[selected] - data["target_force"][selected]))),
            travel_limited_fraction=float(data["travel_limited"][selected].mean()),
            speed_limited_fraction=float(data["speed_limited"][selected].mean()),
        ))
    return dict(pulses=result,
        max_recorded_height_mm=float((data["extent_samples"][:, 6].max() - data["floor"]) * 1000),
        max_recorded_floor_penetration_mm=float(max(0, data["floor"] - data["extent_samples"][:, 3].min()) * 1000),
        rms_speed_mm_s=float(np.sqrt(np.mean(np.sum(data["v_after_1s"] ** 2, axis=1))) * 1000))
