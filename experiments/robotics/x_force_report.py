"""Audit and render the force-command X study without changing old figures."""

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

from experiments.robotics.x_force_study import OUT, check, score, digest
from experiments.robotics.plastic_shaping_study import save_json
from experiments.robotics.plastic_shaping_figure import surface, label_panel, COLUMNS, ROWS, COLORS
from experiments.robotics.hex_shaping_figure import panel_b
from experiments.robotics.hex_shaping_surface import mesh_distance_mm
from experiments.robotics.x_shaping_report import render as render_action
from experiments.robotics.x_shaping_franka import audit as robot_audit
from experiments.robotics.x_pinch_force_pilot import CONTROL


def render(mesh, *, mask=False, top=False, scale=.061, color="#cfb58c"):
    p = pv.Plotter(off_screen=True, window_size=(650, 700))
    p.set_background("white")
    p.add_mesh(mesh, color="black" if mask else color, lighting=not mask,
               smooth_shading=True, split_sharp_edges=True, ambient=.4, diffuse=.7, specular=.1)
    if not mask:
        p.add_mesh(pv.Plane(center=(.08, .08, .0099), i_size=.13, j_size=.13),
                   color="#f1f2f2", lighting=False)
    focal = np.array([.08, .08, .027])
    p.camera_position = [focal + ([0, 0, .3] if top else [0, -.13, .24]), focal, (0, 1, 0)]
    p.enable_parallel_projection()
    p.camera.parallel_scale = scale
    if not mask:
        p.enable_anti_aliasing("ssaa")
    image = p.screenshot(return_img=True)
    p.close()
    return image


def audit(folder):
    check(folder)
    for name in "AB":
        selected_path = folder / f"plans/{name}/selected.json"
        scheduled = folder / f"execution_plan_{name}.json"
        if scheduled.exists():
            execution = json.loads(scheduled.read_text())
            assert digest(selected_path) == execution["selected_sha256"]
            assert digest(folder / f"execution_plan_{name}_source.py") == execution["source_sha256"]
        refinement = folder / "refinement" / name
        if refinement.exists():
            from experiments.robotics.x_force_refine import state
            state(folder, name)
            complete = json.loads((refinement / "complete.json").read_text())
            assert complete["selected"] == json.loads(selected_path.read_text())
    summary = json.loads((folder / "summary.json").read_text())
    target = pv.read(folder / "target.vtp")
    assert target.n_open_edges == 0
    np.testing.assert_allclose(target.volume, .00009, rtol=2e-6)
    target_mask = render(target, mask=True, top=True)[..., :3].mean(-1) < 128
    rows = []
    for r in summary["results"]:
        data = np.load(folder / r["file"])
        mesh = surface(data["x_after_1s"], data["vol0"], h=.00125)
        top = render(mesh, mask=True, top=True)[..., :3].mean(-1) < 128
        oblique = render(mesh, mask=True)[..., :3].mean(-1) < 128
        for m in [top, oblique]:
            assert not m[[0, -1], :].any() and not m[:, [0, -1]].any(), (r["file"], "crop")
        connected = mesh.connectivity(extraction_mode="all").compute_cell_sizes(length=False, volume=False)
        areas = np.bincount(connected.cell_data["RegionId"], weights=connected.cell_data["Area"])
        penetration = np.maximum(.01 - data["x_after_1s"][:, 2], 0)
        phases = json.loads((folder / r["file"]).with_suffix(".phases.json").read_text())
        turns = [i for i, p in enumerate(phases) if p["name"].endswith("rotate")]
        assert np.max(np.abs(data["reaction_force"][np.isin(data["phase_id"], turns)])) == 0
        assert np.max(np.abs(data["reaction_force"][data["phase_id"] == len(phases) - 1])) == 0
        normal = data["tool_centers"][:, 1] - data["tool_centers"][:, 0]
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        np.testing.assert_allclose(data["force_per_finger"],
                                   np.einsum("tfc,tc->tf", data["reaction_force"], normal) * [-1., 1.], atol=1e-10)
        active = data["pulse_id"] >= 0
        gaps = np.linalg.norm(np.diff(data["tool_centers"][active], axis=1)[:, 0], axis=-1) - .028
        speeds = np.linalg.norm(data["tool_velocity"][active], axis=-1)
        assert gaps.min() >= CONTROL["min_gap_m"] - 1e-10
        assert gaps.max() <= CONTROL["max_gap_m"] + 1e-10
        assert speeds.max() <= CONTROL["max_speed_m_s"] + 1e-10
        key = f"{r['setting']}_{'B' if r['material'] == 'A' else 'A'}_plan_{r['planned_for']}.npz"
        other = np.load(folder / key)
        np.testing.assert_array_equal(data["initial"], other["initial"])
        for i in range(4):
            np.testing.assert_array_equal(data["target_force"][data["pulse_id"] == i],
                                          other["target_force"][other["pulse_id"] == i])
        power = np.maximum(-(data["reaction_force"] * data["tool_velocity"]).sum(axis=-1), 0).sum(axis=-1)
        work = {p["name"]: float(power[data["phase_id"] == i].sum() * .004)
                for i, p in enumerate(phases)}
        rows.append(dict(**r,
            footprint_iou=float(np.count_nonzero(top & target_mask) / np.count_nonzero(top | target_mask)),
            reconstruction_voxels_mm=[1., 1.25, 1.5],
            reconstruction_errors_mm=[score(data, target, h) for h in [.001, .00125, .0015]],
            finer_quadrature_error_mm=mesh_distance_mm(mesh, target.subdivide(1, subfilter="linear")),
            surface_components=len(areas), largest_surface_area_fraction=float(areas.max() / areas.sum()),
            max_floor_penetration_mm=float(penetration.max() * 1000),
            fraction_particles_below_floor=float(np.mean(penetration > 0)),
            mean_floor_penetration_mm=float(np.average(penetration, weights=data["vol0"]) * 1000),
            positive_work_by_phase_j=work,
            force_pulse_fraction_of_positive_work=sum(v for k, v in work.items() if k.endswith("force_pulse")) / sum(work.values()),
            max_force_phase_finger_speed_mm_s=float(speeds.max() * 1000),
            minimum_physical_gap_mm=float(gaps.min() * 1000),
            force_peak_n=float(np.linalg.norm(data["reaction_force"], axis=-1).max())))
    save_json(folder / "additional_audit.json", dict(results=rows,
        scope="Same force commands and feedback law; actual trajectories and forces can differ; numerical sensitivity, not convergence"))
    robot_audit(folder, check_speeds=True)
    shutil.copy2(Path(__file__), folder / "audit_source.py")


def panel_a(ax, folder, images, xml):
    label_panel(ax, "(a) Same force command, different shapes")
    data = {n: np.load(folder / f"inputs/same_force_{n}.npz") for n in "AB"}
    np.testing.assert_array_equal(data["A"]["initial"], data["B"]["initial"])
    action = dict(data["A"])
    action.update(stage_0_pressed=data["A"]["most_compressed"],
                  stage_0_centers=data["A"]["compressed_centers"], angles_deg=np.array([90.]))
    picture = render_action(action, stage=0, color="#cfb58c", path=images / "common_force_action.png", robot_xml=xml)
    ia = ax.inset_axes([.005, .16, .40, .63])
    ia.imshow(picture)
    ia.axis("off")
    ax.text(.205, .825, "Pinch", ha="center", fontsize=7.4)
    ax.text(.705, .825, "After release (1 s)", ha="center", fontsize=7.4)
    initial_mask = render(surface(data["A"]["initial"], data["A"]["vol0"], h=.00125),
                          mask=True, top=True, scale=.047)[..., :3].mean(-1) < 128
    for name, cx in [("A", .555), ("B", .855)]:
        picture = render(surface(data[name]["x_after_1s"], data[name]["vol0"], h=.00125), top=True, scale=.047)
        plt.imsave(images / f"common_force_released_{name}.png", picture)
        ia = ax.inset_axes([cx - .145, .19, .29, .48])
        ia.imshow(picture)
        ia.contour(initial_mask, [.5], colors="#425868", linewidths=.6, linestyles=[(0, (3, 2))])
        ia.axis("off")
        ax.text(cx, .64, name, ha="center", fontsize=8, color=COLORS[name], weight="bold", zorder=20)
    ax.text(.5, .115, "1.25 N per finger; 2 s pulse", ha="center", fontsize=7.2, zorder=20)
    ax.text(.5, .025, "Dashed: the common initial outline.",
            ha="center", fontsize=6.8, color="#46515a")


def panel_c(ax, folder, images, xml):
    label_panel(ax, "(c) Shaping with force feedback")
    data = np.load(folder / "baseline_A_plan_A.npz")
    for stage, x, direction in [(2, .26, "y"), (3, .75, "x")]:
        ax.text(x, .83, f"{stage + 1}. Pinch along {direction}", ha="center", fontsize=7.4)
        picture = render_action(data, stage=stage, color="#cfb58c", path=images / f"action_{stage}.png", robot_xml=xml)
        ia = ax.inset_axes([x - .23, .22, .46, .57])
        ia.imshow(picture)
        ia.axis("off")
        ax.text(x, .18, f"Peak {data['peaks_n'][stage]:.2f} N / finger", ha="center", fontsize=7.1)
    sequence = " → ".join(f"{axis}: {peak:.2f}" for axis, peak in zip("yxyx", data["peaks_n"], strict=True))
    ax.text(.5, .085, sequence + " N", ha="center", fontsize=6.8)
    ax.text(.5, .025, "Open, lift, and turn between pinches.", ha="center", fontsize=6.8, color="#46515a")


def diagnostics_report(folder):
    """Keep force tracking, travel limits, and all search trials reviewable."""
    check(folder)
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
    for ax, name in zip(axes, "AB", strict=True):
        rows = json.loads((folder / f"plans/{name}/evaluations.json").read_text())
        refinement = folder / "refinement" / name
        if (refinement / "complete.json").exists():
            import hashlib
            protocol = json.loads((refinement / "protocol.json").read_text())
            for peaks in protocol["coarse_peaks_n"]:
                key = hashlib.sha256(np.array(peaks).tobytes()).hexdigest()[:16]
                rows.append(json.loads((refinement / f"{key}.json").read_text()))
            rows.extend(json.loads((refinement / "evaluations.json").read_text()))
            ax.axvline(32.5, color="#777777", ls=":", lw=1)
        errors = np.array([r.get("surface_mm", np.nan) for r in rows])
        feasible = np.array([r["feasible"] for r in rows])
        index = np.arange(1, len(rows) + 1)
        ax.scatter(index[feasible], errors[feasible], color=COLORS[name], label="Feasible")
        ax.scatter(index[~feasible], errors[~feasible], color="#b54444", marker="x", label="Travel or height guard")
        ax.plot(index, np.minimum.accumulate(np.where(feasible, errors, np.inf)), color=COLORS[name], lw=1)
        ax.set_ylabel(f"Model {name}\nSurface error (mm)")
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    axes[-1].set_xlabel("Objective evaluation (including any cached repeat)")
    fig.tight_layout()
    fig.savefig(folder / "optimization_history.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(4, 4, figsize=(13, 10), sharex=True)
    rows = []
    for row, (material, planned) in enumerate((a, b) for a in "AB" for b in "AB"):
        path = folder / f"baseline_{material}_plan_{planned}.npz"
        data = np.load(path)
        phases = json.loads(path.with_suffix(".phases.json").read_text())
        for i, ax in enumerate(axes[row]):
            phase = next(p for p in phases if p["name"] == f"{i}:force_pulse")
            active = data["pulse_id"] == i
            t = data["time"][active] - phase["start_s"]
            commanded = data["target_force"][active]
            measured = data["force_per_finger"][active].mean(axis=-1)
            gaps = (np.linalg.norm(np.diff(data["tool_centers"][active], axis=1)[:, 0], axis=-1) - .028) * 1000
            ax.plot(t, commanded, color="black", ls="--", lw=1, label="Command")
            ax.plot(t, measured, color=COLORS[material], lw=1, label="Measured mean")
            ax.fill_between(t, 0, 1, where=data["travel_limited"][active],
                            transform=ax.get_xaxis_transform(), color="#c45252", alpha=.15)
            ax.set_title(f"Pinch {i + 1}; min gap {gaps.min():.1f} mm", fontsize=9)
            ax.grid(alpha=.2)
            if i == 0:
                ax.set_ylabel(f"True {material}, plan {planned}\nForce per finger (N)")
            if row == 3:
                ax.set_xlabel("Pulse time (s)")
            rows.append(dict(material=material, planned_for=planned, pinch=i + 1,
                min_gap_mm=float(gaps.min()),
                whole_pulse_rmse_n=float(np.sqrt(np.mean((measured - commanded) ** 2))),
                travel_limited_fraction=float(data["travel_limited"][active].mean())))
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Same controller in all cases. Red shading marks the common travel guard.", fontsize=11)
    fig.tight_layout()
    fig.savefig(folder / "baseline_force_tracking.png", dpi=180)
    plt.close(fig)
    save_json(folder / "tracking_review.json", rows)


def report(folder, dest=None):
    check(folder)
    audit_data = json.loads((folder / "additional_audit.json").read_text())
    robot = json.loads((folder / "franka/audit.json").read_text())
    assert len(robot["results"]) == 4 and robot["speed_limits_checked"]
    images = folder / "renders"
    images.mkdir(exist_ok=True)
    target = pv.read(folder / "target.vtp")
    mask = render(target, mask=True)[..., :3].mean(-1) < 128
    pictures = {"target": render(target, color="#b5b6b0")}
    records = {}
    for name in "AB":
        for planned in "AB":
            key = f"{name}_plan_{planned}"
            path = folder / f"baseline_{key}.npz"
            data = np.load(path)
            pictures[key] = render(surface(data["x_after_1s"], data["vol0"], h=.00125))
            records[key] = json.loads(path.with_suffix(".json").read_text())
            plt.imsave(images / f"{key}.png", pictures[key])
    headings = ["Target", "Planned for A", "Planned for B"]
    guarded = {key: any(p["travel_limited_fraction"] > 0 for p in r["diagnostics"]["pulses"])
               for key, r in records.items()}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 6.0))
    fig.subplots_adjust(left=.03, right=.995, bottom=.065, top=.94, wspace=.025, hspace=.13)
    for row, name in enumerate("AB"):
        for col, planned in enumerate([None, "A", "B"]):
            ax = axes[row, col]
            key = "target" if planned is None else f"{name}_plan_{planned}"
            ax.imshow(pictures[key])
            ax.axis("off")
            if row == 0:
                ax.set_title(headings[col], fontsize=12)
            if col == 0:
                ax.text(0, .5, name, transform=ax.transAxes, fontsize=13, weight="bold", color=COLORS[name])
            else:
                ax.contour(mask, [.5], colors="#253139", linewidths=.8, linestyles=[(0, (3, 2))])
                marker = "†" if guarded[key] else ""
                ax.text(.5, -.035, f"{records[key]['surface_mm']:.3f} mm{marker}", transform=ax.transAxes, ha="center", fontsize=11)
    footer = "Four pinches with force feedback; outcomes 1 s after withdrawal."
    if any(guarded.values()):
        footer += "  † Travel guard active."
    fig.text(.5, .018, footer, ha="center", fontsize=9)
    for suffix in ["png", "pdf"]:
        fig.savefig(folder / f"comparison.{suffix}", dpi=220)
    plt.close(fig)

    fig = plt.figure(figsize=(7.2, 4.15))
    grid = fig.add_gridspec(2, 2, left=.02, right=.985, bottom=.025, top=.99, wspace=.12, hspace=.12)
    xml = folder / "franka/panda_model_snapshot/panda.xml"
    panels = []
    ax = fig.add_subplot(grid[0, 0])
    panel_a(ax, folder, images, xml)
    panels.append(ax)
    ax = fig.add_subplot(grid[0, 1])
    panel_b(ax, folder / "inputs")
    panels.append(ax)
    ax = fig.add_subplot(grid[1, 0])
    panel_c(ax, folder, images, xml)
    panels.append(ax)
    ax = fig.add_subplot(grid[1, 1])
    label_panel(ax, "(d) Target and executed shapes")
    for x, label in zip(COLUMNS, headings, strict=True):
        ax.text(x, .83, label, ha="center", va="center", fontsize=7.)
    for row, name in enumerate("AB"):
        ax.text(.005, ROWS[row] + .11, name, color=COLORS[name], fontsize=8, weight="bold", va="center")
        for col, planned in enumerate([None, "A", "B"]):
            key = "target" if planned is None else f"{name}_plan_{planned}"
            ia = ax.inset_axes([COLUMNS[col] - .145, ROWS[row], .29, .29])
            ia.imshow(pictures[key])
            ia.axis("off")
            if planned is not None:
                ia.contour(mask, [.5], colors="#253139", linewidths=.55, linestyles=[(0, (2.4, 1.8))])
                marker = "†" if guarded[key] else ""
                ax.text(COLUMNS[col], ROWS[row] - .022, f"{records[key]['surface_mm']:.2f} mm{marker}",
                        ha="center", va="top", fontsize=7.2, zorder=20)
    footer = ("3D surface error; † travel guard active." if any(guarded.values())
              else "3D surface error 1 s after full withdrawal.")
    ax.text(.5, .025, footer, ha="center", fontsize=6.8, color="#46515a")
    panels.append(ax)
    stem = folder / "identification_plastic_shaping"
    for suffix in [".png", ".pdf"]:
        fig.savefig(stem.with_suffix(suffix), dpi=300, bbox_inches="tight", pad_inches=.02)
    for name, ax in zip("abcd", panels, strict=True):
        bounds = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(folder / f"panel_{name}.png", dpi=300, bbox_inches=bounds.expanded(1.01, 1.01))
    plt.close(fig)
    sources = folder / "report_source"
    sources.mkdir(exist_ok=True)
    hashes = {}
    for name in ["x_force_report.py", "x_shaping_report.py", "x_shaping_franka.py", "hex_shaping_figure.py", "plastic_shaping_figure.py"]:
        path = Path(__file__).with_name(name)
        shutil.copy2(path, sources / name)
        hashes[name] = digest(path)
    save_json(folder / "figure_provenance.json", dict(
        sources=hashes, records=records, study_directory=str(folder), target_sha256=digest(folder / "target.vtp"),
        data={p.name: digest(p) for p in folder.glob("baseline_*.npz")},
        panel_a="Frozen single-pinch 1.25 N-command pilot; A shown at its minimum gap, both released at 1 s after withdrawal",
        panel_b="Unchanged held-out press force arrays, copied without particle states",
        panel_c="Minimum-gap states during third/fourth pinches of A-matched execution; peak commands listed",
        panel_d="Common target and camera/scale for all four true-material executions",
        audit_cases=len(audit_data["results"]), scope="Simulation, proposed Panda adapters and sampled kinematics; no hardware or actuator force-tracking validation"))
    if dest:
        for suffix in [".png", ".pdf"]:
            shutil.copy2(stem.with_suffix(suffix), dest.with_suffix(suffix))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["audit", "diagnostics", "report"])
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--dest", type=Path)
    a = p.parse_args()
    if a.stage == "audit":
        audit(a.out)
    elif a.stage == "diagnostics":
        diagnostics_report(a.out)
    else:
        report(a.out, a.dest)
