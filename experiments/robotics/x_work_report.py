"""Audits and paper figure for force-observed compression-work commands."""

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

from experiments.robotics.x_work_study import OUT, check, score, digest, MIN_GAP
from experiments.robotics.plastic_shaping_study import save_json
from experiments.robotics.plastic_shaping_figure import surface, label_panel, COLORS, COLUMNS
from experiments.robotics.hex_shaping_surface import mesh_distance_mm
from experiments.robotics.hex_shaping_figure import panel_b
from experiments.robotics.x_force_report import render as shared_render
from experiments.robotics.x_work_probe_panel import panel_a
from experiments.robotics.x_shaping_report import render as render_action
from experiments.robotics.x_shaping_franka import audit as robot_audit

VIEW_SCALE = .055


def render(mesh, **kwargs):
    return shared_render(mesh, scale=VIEW_SCALE, **kwargs)


def audit(folder):
    check(folder)
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
        for mask in [top, oblique]:
            assert not mask[[0, -1], :].any() and not mask[:, [0, -1]].any(), (r["file"], "crop")
        connected = mesh.connectivity(extraction_mode="all").compute_cell_sizes(length=False, volume=False)
        areas = np.bincount(connected.cell_data["RegionId"], weights=connected.cell_data["Area"])
        penetration = np.maximum(.01 - data["x_after_1s"][:, 2], 0)
        phases = json.loads((folder / r["file"]).with_suffix(".phases.json").read_text())
        turns = [i for i, p in enumerate(phases) if p["name"].endswith("rotate")]
        assert np.max(np.abs(data["reaction_force"][np.isin(data["phase_id"], turns)])) == 0
        assert np.max(np.abs(data["reaction_force"][data["phase_id"] == len(phases) - 1])) == 0
        active = data["pinch_id"] >= 0
        speeds = np.linalg.norm(data["tool_velocity"][active], axis=-1)
        assert speeds.max() <= .050000001
        gaps = np.linalg.norm(np.diff(data["tool_centers"][active], axis=1)[:, 0], axis=-1) - .028
        assert gaps.min() >= MIN_GAP - 1e-10 and gaps.max() <= .072 + 1e-10
        other_name = "B" if r["material"] == "A" else "A"
        other = np.load(folder / f"{r['setting']}_{other_name}_plan_{r['planned_for']}.npz")
        np.testing.assert_array_equal(data["initial"], other["initial"])
        np.testing.assert_array_equal(data["requested_work_j"], other["requested_work_j"])
        phase_work = {p["name"]: float(data["work_increment_j"][data["phase_id"] == i].sum())
                      for i, p in enumerate(phases)}
        overshoots = []
        for i, stop in enumerate(r["stops"]):
            if stop == "work":
                actual, commanded = data["actual_work_j"][i], data["requested_work_j"][i]
                overshoots.append(float((actual - commanded) / commanded))
        rows.append(dict(**r,
            footprint_iou=float(np.count_nonzero(top & target_mask) / np.count_nonzero(top | target_mask)),
            reconstruction_voxels_mm=[1., 1.25, 1.5],
            reconstruction_errors_mm=[score(data, target, h) for h in [.001, .00125, .0015]],
            finer_quadrature_error_mm=mesh_distance_mm(mesh, target.subdivide(1, subfilter="linear")),
            surface_components=len(areas), largest_surface_area_fraction=float(areas.max() / areas.sum()),
            max_final_floor_penetration_mm=float(penetration.max() * 1000),
            fraction_particles_below_floor=float(np.mean(penetration > 0)),
            mean_floor_penetration_mm=float(np.average(penetration, weights=data["vol0"]) * 1000),
            positive_work_by_phase_j=phase_work,
            closing_fraction_of_positive_work=float(data["work_increment_j"][active].sum() / data["work_increment_j"].sum()),
            relative_work_overshoots=overshoots,
            max_closing_finger_speed_mm_s=float(speeds.max() * 1000),
            peak_force_per_finger_n=float(np.linalg.norm(data["reaction_force"], axis=-1).max())))
    save_json(folder / "additional_audit.json", dict(results=rows,
        scope="Identical work commands and stopping rule; realized forces, gaps, and durations can differ. Numerical sensitivity, not convergence."))
    robot_audit(folder, check_speeds=True)
    shutil.copy2(__file__, folder / "audit_source.py")


def diagnostics(folder):
    check(folder)
    fig, axes = plt.subplots(4, 4, figsize=(13, 10), sharex=True)
    for row, (material, planned) in enumerate((a, b) for a in "AB" for b in "AB"):
        path = folder / f"baseline_{material}_plan_{planned}.npz"
        data = np.load(path)
        record = json.loads(path.with_suffix(".json").read_text())
        for i, ax in enumerate(axes[row]):
            active = data["pinch_id"] == i
            gap = (np.linalg.norm(np.diff(data["tool_centers"][active], axis=1)[:, 0], axis=-1) - .028) * 1000
            work = np.cumsum(data["work_increment_j"][active]) * 1000
            ax.plot(gap, work, color=COLORS[material], lw=1.5)
            ax.axhline(data["requested_work_j"][i] * 1000, color="black", ls="--", lw=1)
            ax.set_xlim(72, MIN_GAP * 1000)
            ax.set_title(f"Pinch {i + 1}: {record['stops'][i].replace('_', ' ')}", fontsize=9)
            ax.grid(alpha=.2)
            if i == 0:
                ax.set_ylabel(f"True {material}, plan {planned}\nClosing work (mJ)")
            if row == 3:
                ax.set_xlabel("Finger opening (mm)")
    fig.suptitle("Dashed: commanded work of both fingers. Solid: measured positive closing work.", fontsize=11)
    fig.tight_layout()
    fig.savefig(folder / "work_stopping_diagnostics.png", dpi=180)
    plt.close(fig)


def panel_c(ax, folder, images, xml, *, color="#cfb58c"):
    label_panel(ax, "(c) Planned work per pinch")
    data = np.load(folder / "baseline_A_plan_A.npz")
    for stage, x, direction in [(2, .26, "y"), (3, .75, "x")]:
        ax.text(x, .83, f"{stage + 1}. Pinch along {direction}", ha="center", fontsize=7.4)
        picture = render_action(data, stage=stage, color=color, path=images / f"action_{stage}.png", robot_xml=xml)
        ia = ax.inset_axes([x - .23, .22, .46, .57])
        ia.imshow(picture)
        ia.axis("off")
        ax.text(x, .18, f"Command {data['requested_work_j'][stage] * 1000:.1f} mJ", ha="center", fontsize=7.1)
    sequence = " → ".join(f"{axis}: {work * 1000:.1f}" for axis, work in zip("yxyx", data["requested_work_j"], strict=True))
    ax.text(.5, .085, sequence + " mJ", ha="center", fontsize=6.8)
    ax.text(.5, .025, "Work of both fingers; open and lift between pinches.",
            ha="center", fontsize=6.6, color="#46515a")


def report(folder, dest=None, entrypoint=None, render_out=None):
    check(folder)
    audited = json.loads((folder / "additional_audit.json").read_text())
    robot = json.loads((folder / "franka/audit.json").read_text())
    assert len(audited["results"]) == 12 and len(robot["results"]) == 4 and robot["speed_limits_checked"]
    output = folder if render_out is None else render_out
    output.mkdir(parents=True, exist_ok=True)
    images = output / "renders"
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
    guarded = {key: "travel_guard" in record["stops"] for key, record in records.items()}
    headings = ["Target", "Planned for A", "Planned for B"]
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
    footer = "Four pinches with work feedback; outcomes 1 s after withdrawal."
    if any(guarded.values()):
        footer += "  † Minimum opening reached."
    fig.text(.5, .018, footer, ha="center", fontsize=9)
    for suffix in ["png", "pdf"]:
        fig.savefig(output / f"comparison.{suffix}", dpi=220)
    plt.close(fig)

    fig = plt.figure(figsize=(7.2, 4.15))
    grid = fig.add_gridspec(2, 2, left=.02, right=.985, bottom=.025, top=.99, wspace=.12, hspace=.12)
    xml = folder / "franka/panda_model_snapshot/panda.xml"
    panels = []
    for name, position, painter in [("a", (0, 0), panel_a), ("b", (0, 1), panel_b), ("c", (1, 0), panel_c)]:
        ax = fig.add_subplot(grid[position])
        if name == "b":
            painter(ax, folder / "inputs")
        else:
            painter(ax, folder, images, xml)
        panels.append(ax)
    ax = fig.add_subplot(grid[1, 1])
    label_panel(ax, "(d) Target and executed shapes")
    for x, label in zip(COLUMNS, headings, strict=True):
        ax.text(x, .83, label, ha="center", va="center", fontsize=7.)
    result_rows = (.515, .145)
    for row, name in enumerate("AB"):
        ax.text(.005, result_rows[row] + .15, name, color=COLORS[name], fontsize=8, weight="bold", va="center")
        for col, planned in enumerate([None, "A", "B"]):
            key = "target" if planned is None else f"{name}_plan_{planned}"
            ia = ax.inset_axes([COLUMNS[col] - .145, result_rows[row], .29, .30])
            ia.imshow(pictures[key])
            ia.axis("off")
            if planned is not None:
                ia.contour(mask, [.5], colors="#253139", linewidths=.55, linestyles=[(0, (2.4, 1.8))])
                marker = "†" if guarded[key] else ""
                ax.text(COLUMNS[col], result_rows[row] - .012, f"{records[key]['surface_mm']:.2f} mm{marker}",
                        ha="center", va="top", fontsize=7.2, zorder=20)
    footer = "3D surface error; † minimum opening reached." if any(guarded.values()) else "3D surface error 1 s after full withdrawal."
    ax.text(.5, .025, footer, ha="center", fontsize=6.8, color="#46515a")
    panels.append(ax)
    stem = output / "identification_plastic_shaping"
    for suffix in [".png", ".pdf"]:
        fig.savefig(stem.with_suffix(suffix), dpi=300, bbox_inches="tight", pad_inches=.02)
    for name, ax in zip("abcd", panels, strict=True):
        bounds = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(output / f"panel_{name}.png", dpi=300, bbox_inches=bounds.expanded(1.01, 1.01))
    plt.close(fig)
    sources = output / "report_source"
    sources.mkdir(exist_ok=True)
    hashes = {}
    for name in ["x_work_report.py", "x_work_probe_panel.py", "x_force_report.py", "x_shaping_report.py", "x_shaping_franka.py", "hex_shaping_figure.py", "plastic_shaping_figure.py", "hex_shaping_surface.py"]:
        path = Path(__file__).with_name(name)
        shutil.copy2(path, sources / name)
        hashes[name] = digest(path)
    if entrypoint is not None:
        shutil.copy2(entrypoint, sources / entrypoint.name)
        hashes[entrypoint.name] = digest(entrypoint)
    save_json(output / "figure_provenance.json", dict(sources=hashes, records=records,
        protocol_sha256=digest(folder / "protocol.json"), minimum_opening_m=MIN_GAP,
        study_directory=str(folder), target_sha256=digest(folder / "target.vtp"),
        data={p.name: digest(p) for p in folder.glob("baseline_*.npz")},
        input_manifest_sha256=digest(folder / "input_sha256.json"),
        panel_a_calibration_sha256=digest(images / "probe_view_calibration.json"),
        panel_a="Separate frozen single-pinch 1.25 N-command pilot; A shown at minimum gap, both specimens released for 1 s",
        panel_b="Unchanged held-out press force arrays",
        panel_c="End of third/fourth work-controlled closes in true A; commands are positive closing work of both fingers",
        panel_d="Common target, camera, and scale for all true-material outcomes; travel-guard cases explicitly marked",
        scope="Simulation with measured-work stopping; proposed Panda adapters and sampled kinematics, no hardware or actuator-dynamics validation"))
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
        diagnostics(a.out)
    else:
        report(a.out, a.dest)
