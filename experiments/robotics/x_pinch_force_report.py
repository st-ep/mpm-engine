"""Diagnostics and common-scale views of the single force-controlled pinch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from experiments.robotics.x_pinch_force_pilot import OUT, CONTROL, check, digest
from experiments.robotics.plastic_shaping_figure import surface
from experiments.robotics.plastic_shaping_study import save_json

COLORS = dict(A="#2375aa", B="#c96932")


def render(mesh, centers=None, top=True):
    p = pv.Plotter(off_screen=True, window_size=(700, 620))
    p.set_background("white")
    p.add_mesh(mesh, color="#c9aa7d", smooth_shading=True, ambient=.4,
               diffuse=.7, specular=.1)
    p.add_mesh(pv.Plane(center=(.08, .08, .0099), i_size=.13, j_size=.13),
               color="#f1f2f2", lighting=False)
    if centers is not None:
        for cp in centers:
            p.add_mesh(pv.Cylinder(center=cp, direction=(0, 0, 1), radius=.014,
                       height=.045, resolution=80), color="#5b6872", opacity=.72,
                       smooth_shading=True)
    focal = np.array([.08, .08, .025])
    p.camera_position = [focal + ([0, 0, .3] if top else [.12, -.16, .18]), focal, (0, 1, 0)]
    p.enable_parallel_projection()
    p.camera.parallel_scale = .061
    p.enable_anti_aliasing("ssaa")
    img = p.screenshot(return_img=True)
    p.close()
    return img


def audit(folder):
    check(folder)
    records = []
    for path in sorted(folder.glob("*.npz")):
        d = np.load(path)
        record = json.loads(path.with_suffix(".json").read_text())
        assert digest(path) == record["data_sha256"]
        widths = []
        for h in [.001, .00125, .0015]:
            mesh = surface(d["x_after_1s"], d["vol0"], h=h)
            section = mesh.slice(normal=(1, 0, 0), origin=(.08, .08, .025))
            widths.append(float(np.ptp(section.points[:, 1]) * 1000))
        connected = mesh.connectivity()
        component_count = len(np.unique(connected.cell_data["RegionId"]))
        all_states = np.concatenate([d["initial"], d["pulse_frames"].reshape(-1, 3), d["x_after_1s"]])
        penetration = np.maximum(.01 - all_states[:, 2], 0)
        phase_max_force = {}
        phases = json.loads(path.with_suffix(".phases.json").read_text())
        for i, phase in enumerate(phases):
            selected = d["phase_id"] == i
            phase_max_force[phase["name"]] = float(np.abs(d["reaction_force"][selected]).max())
        active = d["phase_id"] == 3
        velocity = d["tool_velocity"][active, 0, 1]
        gaps = d["tool_centers"][active, 1, 1] - d["tool_centers"][active, 0, 1] - .028
        assert np.max(np.abs(velocity)) <= CONTROL["max_speed_m_s"] + 1e-10
        assert gaps.min() >= CONTROL["min_gap_m"] - 1e-10
        assert gaps.max() <= CONTROL["max_gap_m"] + 1e-10
        assert phase_max_force["wait"] == 0
        assert component_count == 1
        assert all_states[:, :2].min() > 0.01 and all_states[:, :2].max() < 0.15
        records.append(dict(**record, file=path.name, surface_waist_mm=widths[1],
            reconstruction_voxels_mm=[1., 1.25, 1.5], reconstruction_waist_mm=widths,
            surface_components=component_count,
            max_recorded_height_mm=float((all_states[:, 2].max() - .01) * 1000),
            max_recorded_floor_penetration_mm=float(penetration.max() * 1000),
            state_min_m=all_states.min(axis=0).tolist(), state_max_m=all_states.max(axis=0).tolist(),
            phase_max_abs_force_n=phase_max_force))
    pairs = []
    for setting in sorted({r["setting"] for r in records}):
        for peak in sorted({r["peak_force_n"] for r in records if r["setting"] == setting}):
            pair = [next((r for r in records if r["setting"] == setting and
                    r["peak_force_n"] == peak and r["material"] == name), None) for name in "AB"]
            if None in pair:
                continue
            a, b = pair
            da, db = [np.load(folder / r["file"]) for r in pair]
            np.testing.assert_array_equal(da["initial"], db["initial"])
            ma = da["phase_id"] == 3
            mb = db["phase_id"] == 3
            np.testing.assert_array_equal(da["target_force"][ma], db["target_force"][mb])
            fa, fb = [d["force_per_finger"][m].mean(axis=1) for d, m in [(da, ma), (db, mb)]]
            pairs.append(dict(setting=setting, peak_force_n=peak,
                waist_B_minus_A_mm=b["surface_waist_mm"] - a["surface_waist_mm"],
                raw_waist_B_minus_A_mm=b["raw_waist_mm"] - a["raw_waist_mm"],
                raw_waist_98_B_minus_A_mm=b["raw_waist_98_mm"] - a["raw_waist_98_mm"],
                whole_pulse_force_AB_rmse_n=float(np.sqrt(np.mean((fa - fb) ** 2))),
                tracking_relative_rmse_max=max(r["force_plateau_rmse_n"] for r in pair) / peak if peak else None,
                any_travel_limit=any(r["travel_limited_fraction"] > 0 for r in pair),
                any_speed_limit=any(r["speed_limited_fraction"] > 0 for r in pair)))
    result = dict(results=records, pairs=pairs,
        waist="Full y extent of reconstructed surface intersected with x=80 mm plane; no registration",
        scope="Fixed-duration force-controller pilot, no target fitting or identification-to-control result")
    save_json(folder / "audit.json", result)
    print(json.dumps(pairs, indent=2))
    return result


def report(folder, selected_peak):
    check(folder)
    result = json.loads((folder / "audit.json").read_text())
    rows = result["results"]
    baseline = [r for r in rows if r["setting"] == "baseline"]
    peaks = sorted({r["peak_force_n"] for r in baseline})
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    renders = folder / "renders"
    renders.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, len(peaks), figsize=(2.45 * len(peaks), 5.2), squeeze=False)
    fig.subplots_adjust(left=.045, right=.995, top=.86, bottom=.09, wspace=.01, hspace=.13)
    for row, name in enumerate("AB"):
        for col, peak in enumerate(peaks):
            record = next(r for r in baseline if r["material"] == name and r["peak_force_n"] == peak)
            d = np.load(folder / record["file"])
            mesh = surface(d["x_after_1s"], d["vol0"], h=.00125)
            pic = render(mesh)
            plt.imsave(renders / f"{name}_{peak:g}N.png", pic)
            ax = axes[row, col]
            ax.imshow(pic)
            ax.axis("off")
            if row == 0:
                ax.set_title(f"{peak:g} N / finger", fontsize=11)
            if col == 0:
                ax.text(-.12, .5, name, transform=ax.transAxes, color=COLORS[name],
                        fontsize=16, weight="bold", va="center")
            ax.text(.5, -.01, f"RMS displacement {record['rms_displacement_mm']:.2f} mm", transform=ax.transAxes,
                    ha="center", fontsize=10)
            if record["travel_limited_fraction"] > 0:
                ax.text(.5, -.105, "Travel limit reached", transform=ax.transAxes,
                        ha="center", fontsize=9, color="#a64734")
    fig.suptitle("One force-controlled pinch: released shapes", fontsize=16, y=.98)
    fig.text(.5, .015, "Same initial specimen, pulse duration, controller and view. Observed 1 s after withdrawal. Listed forces are requested peaks.",
             ha="center", fontsize=10)
    fig.savefig(folder / "sweep_shapes.png", dpi=180)
    fig.savefig(folder / "sweep_shapes.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for name in "AB":
        for setting, style in [("baseline", "-o"), ("half_dt", "--x"), ("grid80", ":s")]:
            rr = sorted([r for r in rows if r["material"] == name and r["setting"] == setting], key=lambda r: r["peak_force_n"])
            if not rr:
                continue
            xx = [r["peak_force_n"] for r in rr]
            for ax, key in zip(axes, ["rms_displacement_mm", "force_plateau_mean_n"], strict=True):
                ax.plot(xx, [r[key] for r in rr], style, color=COLORS[name], label=f"{name}, {setting}")
    axes[0].set(ylabel="RMS particle displacement after release (mm)", xlabel="Requested peak force per finger (N)")
    axes[1].set(ylabel="Mean measured plateau force (N)", xlabel="Requested peak force per finger (N)")
    axes[1].plot([0, max(peaks)], [0, max(peaks)], color=".65", linestyle="--", label="Perfect tracking")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=8)
    if any(r["travel_limited_fraction"] > 0 for r in rows):
        fig.suptitle("At 2 N, A reaches the travel limit and does not track the force command.",
                     fontsize=10, color="#a64734")
    fig.savefig(folder / "sweep_metrics.png", dpi=180)
    fig.savefig(folder / "sweep_metrics.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.7))
    fig.subplots_adjust(left=.11, right=.995, top=.88, bottom=.09, wspace=.015, hspace=.17)
    for row, name in enumerate("AB"):
        d = np.load(folder / f"baseline_{name}_{selected_peak:g}N.npz")
        r = next(r for r in baseline if r["material"] == name and r["peak_force_n"] == selected_peak)
        for col, key in enumerate(["initial", "most_compressed", "x_after_1s"]):
            mesh = surface(d[key], d["vol0"], h=.00125)
            pic = render(mesh, d["compressed_centers"] if col == 1 else None)
            ax = axes[row, col]
            ax.imshow(pic)
            ax.axis("off")
            if row == 0:
                ax.set_title(["Initial specimen", "Smallest finger separation", "After release"][col], fontsize=12)
            if col == 0:
                ax.text(-.05, .5, f"{name}\nYield {1 if name == 'A' else 10} kPa", transform=ax.transAxes,
                        color=COLORS[name], va="center", ha="right", fontsize=11, weight="bold")
            if col == 2:
                ax.text(.5, -.02, f"RMS displacement: {r['rms_displacement_mm']:.2f} mm", transform=ax.transAxes,
                        ha="center", fontsize=11)
    fig.suptitle(f"Same {selected_peak:g} N force command, different deformation", fontsize=16, y=.98)
    fig.text(.5, .017, "One centered pinch; same geometry and controller. Released states shown 1 s after full withdrawal, at a common scale.", ha="center", fontsize=10)
    fig.savefig(folder / "selected_comparison.png", dpi=200)
    fig.savefig(folder / "selected_comparison.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8, 5.8), sharex=True, layout="constrained")
    for name in "AB":
        d = np.load(folder / f"baseline_{name}_{selected_peak:g}N.npz")
        t = d["time"] - d["pulse_start_s"]
        active = d["phase_id"] == 3
        f = d["force_per_finger"].mean(axis=1)
        axes[0].plot(t[active], f[active], color=COLORS[name], alpha=.55, label=f"{name}, measured")
        axes[0].plot(t[active], d["filtered_force"][active], color=COLORS[name], linewidth=1.5)
        gap = (d["tool_centers"][:, 1, 1] - d["tool_centers"][:, 0, 1] - .028) * 1000
        axes[1].plot(t[active], gap[active], color=COLORS[name], label=name)
        if name == "A":
            axes[0].plot(t[active], d["target_force"][active], "k--", label="Requested")
    axes[0].set(ylabel="Force per finger (N)")
    axes[1].set(ylabel="Finger surface gap (mm)", xlabel="Time since force pulse began (s)")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.15)
        ax.legend(fontsize=9)
    fig.savefig(folder / "selected_tracking.png", dpi=180)
    fig.savefig(folder / "selected_tracking.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.3), sharex=True, sharey=True, layout="constrained")
    for ax, name in zip(axes, "AB", strict=True):
        d = np.load(folder / f"baseline_{name}_{selected_peak:g}N.npz")
        xy = (d["x_after_1s"][:, :2] - .08) * 1000
        ax.scatter(xy[:, 0], xy[:, 1], s=.5, alpha=.25, color=COLORS[name], rasterized=True)
        ax.set(xlim=(-50, 50), ylim=(-40, 40), aspect="equal", xlabel="x from specimen center (mm)",
               title=f"Material {name}: all raw particles")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("y from specimen center (mm)")
    fig.savefig(folder / "selected_raw_particles.png", dpi=200)
    fig.savefig(folder / "selected_raw_particles.pdf")
    plt.close(fig)
    shutil.copy2(Path(__file__), folder / "report_source.py")
    save_json(folder / "figure_provenance.json", dict(
        selected_peak_n=selected_peak, report_sha256=digest(Path(__file__)),
        inputs={r["file"]: r["data_sha256"] for r in rows},
        reconstruction=dict(voxel_m=.00125, sigma_voxels=1.3, isovalue=.5),
        camera=dict(parallel_scale_m=.061, focal_m=[.08, .08, .025], offset_m=[0, 0, .3]),
        controller=CONTROL, scope="Pilot only; original paper untouched",
    ))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["audit", "report"])
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--selected-peak", type=float, default=2.)
    args = p.parse_args()
    if args.stage == "audit":
        audit(args.out)
    else:
        report(args.out, args.selected_peak)
