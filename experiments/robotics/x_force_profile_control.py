"""Force-trajectory tracking with nominal closing-velocity feedforward.

Reference profiles are generated in identified models. Execution receives
only force/time and velocity/time arrays, with no final-gap command or work
budget. One common feedback law and travel/speed limits apply to all cases.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np

from experiments.robotics import x_shaping as core

CONTROL = dict(kp_m_s_n=.050, ki_m_s2_n=.040, filter_s=.012,
    max_speed_m_s=.050, max_acceleration_m_s2=.50,
    min_gap_m=.012, max_gap_m=.072, gain_reference_n=1.25,
    nominal_speed_m_s=.020)


def nominal_profiles(gaps_mm):
    assert len(gaps_mm)==4 and np.all((np.asarray(gaps_mm)>=16)&(np.asarray(gaps_mm)<=54))
    profiles=[]
    tick=core.CONFIG["tick"]
    ramp_ticks=round(CONTROL["nominal_speed_m_s"] / CONTROL["max_acceleration_m_s2"] / tick)
    for gap in gaps_mm:
        distance=(core.CONFIG["opening"]-gap/1000)/2
        count=int(np.ceil(distance/(CONTROL["nominal_speed_m_s"]*tick)))+ramp_ticks
        j=np.arange(count)+.5
        weights=np.minimum(np.minimum(j/ramp_ticks,(count-j)/ramp_ticks),1.)
        velocity=weights*distance/(tick*weights.sum())
        assert velocity.max() <= CONTROL["nominal_speed_m_s"]+1e-12
        profiles.append(dict(velocity_ff=velocity, feedback_reference=np.zeros(count),
                             force_reference=np.zeros(count), peak_n=1.))
    return profiles


def execute(law, profiles, grid=64, dt=0.00005, device="cuda:0", reference=False):
    cfg, c = core.CONFIG, CONTROL
    assert 1 <= len(profiles) <= 4
    peaks = np.array([p["peak_n"] for p in profiles])
    durations = np.array([len(p["velocity_ff"]) * cfg["tick"] for p in profiles])
    for p in profiles:
        assert len(p["velocity_ff"]) == len(p["feedback_reference"]) == len(p["force_reference"])
        assert np.max(np.abs(p["velocity_ff"])) <= c["nominal_speed_m_s"] + 1e-12
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
                angles_deg=np.rad2deg(angles), reference_generation=np.array(reference))
    fields = {key: [] for key in ["time", "tool_centers", "tool_velocity", "reaction_force",
              "force_per_finger", "target_force", "filtered_force", "force_error",
              "integral_velocity", "command_velocity", "speed_limited", "acceleration_limited",
              "travel_limited", "phase_id", "pulse_id", "feedforward_velocity", "reference_force_after", "filtered_force_before"]}
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
                acceleration=False, travel=False, ff=0., reference_force=0.):
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
        filtered_before = filtered
        filtered += alpha * (compressive.mean() - filtered)
        elapsed += tick
        pose = nxt
        row = dict(time=elapsed, tool_centers=pose.copy(), tool_velocity=velocity,
                   reaction_force=force, force_per_finger=compressive,
                   target_force=target, filtered_force=filtered, force_error=error,
                   integral_velocity=integral, command_velocity=command,
                   speed_limited=speed, acceleration_limited=acceleration,
                   travel_limited=travel, phase_id=current_phase, pulse_id=current_pulse, feedforward_velocity=ff,
                   reference_force_after=reference_force, filtered_force_before=filtered_before)
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
        data[f"stage_{i}_before_pulse"] = solver.x().copy()
        start_phase(f"{i}:force_profile")
        current_pulse = i
        integral, last_velocity = 0., 0.
        gain_scale = c["gain_reference_n"] / max(c["gain_reference_n"], peak)
        data[f"stage_{i}_gain_scale"] = np.array(gain_scale)
        minimum_gap = c["max_gap_m"]
        stop_reason = "profile_complete"
        profile = profiles[i]
        for j, ff in enumerate(profile["velocity_ff"]):
            target = filtered if reference else profile["feedback_reference"][j]
            error = target - filtered
            trial_integral = integral + c["ki_m_s2_n"] * gain_scale * error * tick
            raw = ff + c["kp_m_s_n"] * gain_scale * error + trial_integral
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
            advance(core.centers(angle, next_gap, cfg["finger_bottom"]), target, error,
                    integral, raw, speed_limited, acceleration_limited, travel_limited,
                    ff=ff, reference_force=profile["force_reference"][j])
            last_velocity = actual_v
            if next_gap < minimum_gap:
                minimum_gap = next_gap
                data[f"stage_{i}_pressed"] = solver.x().copy()
                data[f"stage_{i}_centers"] = pose.copy()
                data[f"stage_{i}_pressed_time_s"] = np.array(elapsed)
            if next_gap <= c["min_gap_m"] + 1e-12:
                stop_reason = "travel_guard"
                break
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


def reference_profiles(data):
    assert data["reference_generation"]
    profiles=[]
    for i in range(4):
        selected=data["pulse_id"]==i
        force=data["force_per_finger"][selected].mean(axis=1)
        profiles.append(dict(velocity_ff=data["feedforward_velocity"][selected].copy(),
            feedback_reference=data["filtered_force_before"][selected].copy(),
            force_reference=force.copy(),peak_n=float(force.max())))
    return profiles


def diagnostics(data, phases):
    result=[]
    for i,peak in enumerate(data["peaks_n"]):
        selected=data["pulse_id"]==i
        actual=data["force_per_finger"][selected].mean(axis=1)
        desired=data["reference_force_after"][selected]
        denom=np.linalg.norm(desired)
        result.append(dict(peak_command_n=float(peak),duration_command_s=float(data["durations_s"][i]),
            min_gap_mm=float(data["min_gaps_mm"][i]),
            force_relative_l2=float(np.linalg.norm(actual-desired)/denom) if denom>1e-12 else None,
            force_rmse_n=float(np.sqrt(np.mean((actual-desired)**2))),
            speed_limited_fraction=float(data["speed_limited"][selected].mean()),
            travel_limited_fraction=float(data["travel_limited"][selected].mean())))
    return dict(pulses=result,
        max_recorded_height_mm=float((data["extent_samples"][:,6].max()-data["floor"])*1000),
        max_recorded_floor_penetration_mm=float(max(0,data["floor"]-data["extent_samples"][:,3].min())*1000),
        rms_speed_mm_s=float(np.sqrt(np.mean(np.sum(data["v_after_1s"]**2,axis=1)))*1000))
