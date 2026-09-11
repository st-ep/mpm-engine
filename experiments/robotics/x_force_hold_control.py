"""Force-established dwell actions for cylindrical shaping; separate from all frozen studies.

Each action specifies force amplitude and dwell after force attainment. Motion follows measured
per-finger force with common PI gains and speed/acceleration/travel limits.
No work target, final-opening target, or shape feedback is used.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np

from experiments.robotics import x_shaping as core
from experiments.robotics.x_pinch_force_pilot import CONTROL as LEGACY_CONTROL

CONTROL = dict(LEGACY_CONTROL, min_gap_m=.012, rise_s=.2, fall_s=.12,
               contact_force_n=.05, contact_samples=3, contact_timeout_s=4.,
               attain_tolerance=.10, attain_samples=3, attain_timeout_s=3.)
CONTROL.pop("hold_s")


def ramp(t, peak):
    return peak * .5 * (1 - np.cos(np.pi * np.clip(t / CONTROL["rise_s"], 0, 1)))


def execute(law, peaks, durations, grid=64, dt=0.00005, device="cuda:0"):
    cfg, c = core.CONFIG, CONTROL
    peaks = np.asarray(peaks, float)
    durations = np.asarray(durations, float)
    assert 1 <= len(peaks) <= 4 and np.all((peaks >= 0) & (peaks <= 14))
    assert durations.shape == peaks.shape and np.all((durations >= .08) & (durations <= 1.5))
    durations = np.round(durations / core.CONFIG["tick"]) * core.CONFIG["tick"]
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
                n_grid=np.array(grid), dt=np.array(dt), peaks_n=peaks, durations_s=durations,
                angles_deg=np.rad2deg(angles))
    fields = {key: [] for key in ["time", "tool_centers", "tool_velocity", "reaction_force",
              "force_per_finger", "target_force", "filtered_force", "force_error",
              "integral_velocity", "command_velocity", "speed_limited", "acceleration_limited",
              "travel_limited", "phase_id", "pulse_id", "control_mode"]}
    phases = []
    elapsed, filtered = 0., 0.
    alpha = 1 - np.exp(-tick / c["filter_s"])
    current_phase, current_pulse, mode = -1, -1, -1
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
                   travel_limited=travel, phase_id=current_phase, pulse_id=current_pulse, control_mode=mode)
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

    min_gaps, stop_reasons = [], []
    for i, (peak, angle) in enumerate(zip(peaks, angles, strict=True)):
        if i:
            move(core.centers(angle, cfg["opening"], cfg["hover_bottom"]), 1.,
                 f"{i}:rotate", angles[i - 1], angle)
        move(core.centers(angle, cfg["opening"], cfg["finger_bottom"]),
             (cfg["hover_bottom"] - cfg["finger_bottom"]) / cfg["vertical_speed"], f"{i}:lower")
        # Establish contact gently before starting the force-pulse clock.
        # The common force threshold does not prescribe a stopping gap.
        start_phase(f"{i}:engage")
        contact_count = 0
        for _ in range(round(c["contact_timeout_s"] / tick)):
            gap = np.linalg.norm(pose[1] - pose[0]) - 2 * cfg["radius"]
            next_gap = max(c["min_gap_m"], gap - 2 * c["touch_speed_m_s"] * tick)
            advance(core.centers(angle, next_gap, cfg["finger_bottom"]))
            contact_count = contact_count + 1 if filtered >= c["contact_force_n"] else 0
            if contact_count >= c["contact_samples"]:
                break
            if next_gap <= c["min_gap_m"] + 1e-12:
                raise RuntimeError("Contact search reached the minimum opening")
        else:
            raise RuntimeError("Contact search timed out")
        data[f"stage_{i}_contact_force_n"] = np.array(filtered)
        data[f"stage_{i}_contact_gap_mm"] = np.array(next_gap * 1000)
        data[f"stage_{i}_before_pulse"] = solver.x().copy()
        start_phase(f"{i}:force_action")
        current_pulse = i
        duration = durations[i]
        integral, last_velocity = 0., 0.
        minimum_gap = c["max_gap_m"]
        stop_reason = "attainment_timeout"
        attained, hold_count, fall_count, attain_count = False, 0, 0, 0
        mode = 0  # 0: ramp/attain; 1: established-force dwell; 2: unload
        for j in range(round((c["attain_timeout_s"] + duration + c["fall_s"]) / tick) + 2):
            if mode == 0:
                target = ramp((j + .5) * tick, peak)
            elif mode == 1:
                target = peak
            else:
                target = peak * .5 * (1 + np.cos(np.pi * min(1, (fall_count + .5) * tick / c["fall_s"])))
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
            if next_gap <= c["min_gap_m"] + 1e-12:
                stop_reason = "travel_guard"
                break
            if mode == 0:
                in_band = abs(filtered - peak) <= c["attain_tolerance"] * peak
                attain_count = attain_count + 1 if in_band and (j + 1) * tick >= c["rise_s"] else 0
                if attain_count >= c["attain_samples"]:
                    attained = True
                    mode = 1
                    data[f"stage_{i}_attained_time_s"] = np.array(elapsed)
                elif (j + 1) * tick >= c["attain_timeout_s"] - 1e-9:
                    break
            elif mode == 1:
                hold_count += 1
                if hold_count >= round(duration / tick):
                    mode = 2
            else:
                fall_count += 1
                if fall_count >= round(c["fall_s"] / tick):
                    stop_reason = "dwell_complete"
                    break
        data[f"stage_{i}_attained"] = np.array(attained)
        data[f"stage_{i}_actual_dwell_s"] = np.array(hold_count * tick)
        mode = -1
        min_gaps.append(minimum_gap * 1000)
        stop_reasons.append(stop_reason)
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
    return data, phases, stop_reasons


def diagnostics(data, phases):
    result = []
    for i, peak in enumerate(data["peaks_n"]):
        selected = data["pulse_id"] == i
        plateau = selected & (data["control_mode"] == 1)
        f = data["force_per_finger"].mean(axis=1)
        rmse = float(np.sqrt(np.mean((f[plateau] - peak) ** 2))) if plateau.any() else None
        result.append(dict(
            peak_command_n=float(peak), dwell_command_s=float(data["durations_s"][i]),
            actual_dwell_s=float(data[f"stage_{i}_actual_dwell_s"]),
            force_attained=bool(data[f"stage_{i}_attained"]),
            min_gap_mm=float(data["min_gaps_mm"][i]),
            mean_plateau_force_n=float(f[plateau].mean()) if plateau.any() else None,
            plateau_rmse_n=rmse, relative_hold_rmse=rmse / peak if rmse is not None else None,
            whole_pulse_mae_n=float(np.mean(abs(f[selected] - data["target_force"][selected]))),
            travel_limited_fraction=float(data["travel_limited"][selected].mean()),
            speed_limited_fraction=float(data["speed_limited"][selected].mean()),
        ))
    return dict(pulses=result,
        max_recorded_height_mm=float((data["extent_samples"][:, 6].max() - data["floor"]) * 1000),
        max_recorded_floor_penetration_mm=float(max(0, data["floor"] - data["extent_samples"][:, 3].min()) * 1000),
        rms_speed_mm_s=float(np.sqrt(np.mean(np.sum(data["v_after_1s"] ** 2, axis=1))) * 1000))
