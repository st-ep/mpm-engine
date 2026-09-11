"""Shared rounded-X renderer and the original nominal-comparison audit.

The active paper uses x_letter_cross_report.py to supply the A/B cross-executions.
Without cross_folder this module reproduces the historical nominal comparison.
"""

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

from experiments.robotics.x_letter_study import OUT, check, digest, score
from experiments.robotics.plastic_shaping_figure import (
    surface,
    snapshot as press_snapshot,
    label_panel,
    COLUMNS,
    ROWS,
    COLORS,
)
from experiments.robotics.hex_shaping_figure import panel_b
from experiments.robotics.x_shaping_report import render as render_action
from experiments.robotics.hex_shaping_surface import mesh_distance_mm
from experiments.robotics.plastic_shaping_study import save_json


def recovery_view(data):
    """Released state and initial silhouette, using one fixed frontal camera."""
    floor = float(data["floor"])
    p = pv.Plotter(off_screen=True, window_size=(800, 500))
    p.set_background("white")
    p.add_mesh(
        surface(data["released"], data["vol0"]), color="#cfb58c",
        smooth_shading=True, ambient=0.4, diffuse=0.7, specular=0.1,
    )
    p.add_mesh(
        pv.Plane(center=(0.15, 0.15, floor), i_size=0.15, j_size=0.11),
        color="#eef0f0", lighting=False,
    )
    focal = np.array([0.15, 0.15, floor + 0.035])
    p.camera_position = [focal + np.array([0, -0.35, 0.035]), focal, (0, 0, 1)]
    p.enable_parallel_projection()
    p.camera.parallel_scale = 0.047
    p.enable_anti_aliasing("ssaa")
    picture = p.screenshot(return_img=True)
    p.clear()
    p.add_mesh(
        surface(data["initial"], data["vol0"]), color="black",
        lighting=False, reset_camera=False,
    )
    mask = p.screenshot(return_img=True)[..., :3].mean(-1) < 128
    p.close()
    assert not mask[[0, -1], :].any() and not mask[:, [0, -1]].any()
    return picture, mask


def panel_a(ax, folder, images):
    """A prominent press scene followed by a common-scale recovery comparison."""
    label_panel(ax, "(a) Same press, different recovery")
    data = {name: np.load(folder / f"probe_{name}.npz") for name in "AB"}
    np.testing.assert_array_equal(data["A"]["initial"], data["B"]["initial"])
    initial_height = float(data["A"]["initial"][:, 2].max() - data["A"]["floor"])
    picture = press_snapshot(
        data["A"]["pressed"], floor=float(data["A"]["floor"]),
        volume0=data["A"]["vol0"], path=images / "press_action.png", mode="press",
        plate_bottom=float(data["A"]["floor"]) + 0.06 - float(data["A"]["depth"]),
        plate_thickness=4 * float(data["A"]["grid_lim"]) / int(data["A"]["n_grid"]),
        press_stroke=float(data["A"]["depth"]),
    )
    ia = ax.inset_axes([0.005, 0.16, 0.40, 0.63])
    ia.imshow(picture)
    ia.axis("off")
    ax.text(0.205, 0.825, "Press", ha="center", fontsize=7.4)
    ax.text(0.705, 0.825, "After release (0.12 s)", ha="center", fontsize=7.4)
    ax.text(0.205, 0.125, "14 mm stroke", ha="center", fontsize=7.4, zorder=20)
    records = {}
    for name, cx in [("A", 0.555), ("B", 0.855)]:
        item = data[name]
        assert float(item["release_time"]) == 0.12
        picture, mask = recovery_view(item)
        plt.imsave(images / f"recovery_{name}.png", picture)
        np.save(images / f"recovery_{name}_initial_mask.npy", mask)
        ia = ax.inset_axes([cx - 0.145, 0.19, 0.29, 0.48])
        ia.imshow(picture)
        ia.contour(mask, [0.5], colors="#425868", linewidths=0.75,
                   linestyles=[(0, (3, 2))])
        ia.axis("off")
        ax.text(cx, 0.60, name, ha="center", fontsize=8,
                color=COLORS[name], fontweight="bold", zorder=20)
        height = float(item["released"][:, 2].max() - item["floor"])
        ax.text(cx, 0.125, f"{height * 1000:.1f} mm", ha="center",
                fontsize=7.4, zorder=20)
        records[name] = dict(
            data_sha256=digest(folder / f"probe_{name}.npz"),
            initial_height_mm=initial_height * 1000, released_height_mm=height * 1000,
        )
    ax.text(0.5, 0.025, f"Dashed: initial shape ({initial_height * 1000:.1f} mm high).",
            ha="center", fontsize=6.8, color="#46515a")
    return dict(
        records=records, height="Maximum particle z minus the fixed support height",
        press_scene="Recorded material A at the end of the 14 mm press; schematic mount; blue ghost is the initial plate pose",
        observation="0.12 s after instantaneous tool removal; not a settled state",
        camera=dict(window_size=[800, 500], focal_height_m=0.035,
                    offset_m=[0, -0.35, 0.035], parallel_scale_m=0.047),
        surface=dict(voxel_m=0.0025, gaussian_sigma_voxels=1.3, isovalue=0.5),
        geometry="Unmodified recorded states, same camera and scale; initial surface silhouette",
    )


def render(mesh, *, mask=False, top=False, color="#4a91b4"):
    p = pv.Plotter(off_screen=True, window_size=(650, 700))
    p.set_background("white")
    p.add_mesh(
        mesh,
        color="black" if mask else color,
        lighting=not mask,
        smooth_shading=True,
        split_sharp_edges=True,
        ambient=0.4,
        diffuse=0.7,
        specular=0.1,
    )
    if not mask:
        p.add_mesh(
            pv.Plane(center=(0.08, 0.08, 0.0099), i_size=0.102, j_size=0.102),
            color="#f1f2f2",
            lighting=False,
        )
    focal = np.array([0.08, 0.08, 0.027])
    p.camera_position = [
        focal + np.array([0, 0, 0.3] if top else [0, -0.13, 0.24]),
        focal,
        (0, 1, 0),
    ]
    p.enable_parallel_projection()
    p.camera.parallel_scale = 0.049
    if not mask:
        p.enable_anti_aliasing("ssaa")
    img = p.screenshot(return_img=True)
    p.close()
    return img


def audit(folder):
    check(folder)
    laws = json.loads((folder / "inputs/models.json").read_text())
    protocol = json.loads((folder / "protocol.json").read_text())
    for model in protocol["models"]:
        for path in (folder / "plans" / model).glob("*.npz"):
            config = json.loads(path.with_suffix(".config.json").read_text())
            assert config["law"] == laws[model]
            assert config["n_grid"] == protocol["n_grid"]
            assert config["dt"] == protocol["dt"]
            gaps = np.array(config["gaps_mm"])
            bounds = np.array(protocol["bounds_mm"])
            assert np.all((gaps >= bounds[:, 0]) & (gaps <= bounds[:, 1]))
    for setting in ["baseline", "half_dt", "grid80"]:
        reference = np.load(folder / f"{setting}_A_nominal.npz")
        for name in "AB":
            nominal = np.load(folder / f"{setting}_{name}_nominal.npz")
            np.testing.assert_array_equal(nominal["gaps_mm"], reference["gaps_mm"])
            np.testing.assert_array_equal(nominal["time"], reference["time"])
            np.testing.assert_array_equal(nominal["tool_centers"], reference["tool_centers"])
            for method in ["nominal", "identified"]:
                data = np.load(folder / f"{setting}_{name}_{method}.npz")
                np.testing.assert_array_equal(data["initial"], reference["initial"])
                np.testing.assert_array_equal(data["vol0"], reference["vol0"])
    target = pv.read(folder / "target.vtp")
    assert target.n_open_edges == 0
    np.testing.assert_allclose(target.volume, 0.00009, rtol=2e-6)
    target_mask = render(target, mask=True, top=True)[..., :3].mean(-1) < 128
    rows = []
    for path in sorted(folder.glob("*.npz")):
        record_path = path.with_suffix(".json")
        if not record_path.exists():
            continue
        data = np.load(path)
        record = json.loads(record_path.read_text())
        mesh = surface(data["x_after_1s"], data["vol0"], h=0.00125)
        mask = render(mesh, mask=True, top=True)[..., :3].mean(-1) < 128
        oblique_mask = render(mesh, mask=True)[..., :3].mean(-1) < 128
        for silhouette in [mask, oblique_mask]:
            assert not silhouette[[0, -1], :].any(), "Outcome cropped vertically"
            assert not silhouette[:, [0, -1]].any(), "Outcome cropped horizontally"
        connected = mesh.connectivity(extraction_mode="all").compute_cell_sizes(
            length=False, volume=False
        )
        areas = np.bincount(connected.cell_data["RegionId"], weights=connected.cell_data["Area"])
        penetration = np.maximum(float(data["floor"]) - data["x_after_1s"][:, 2], 0.0)
        phases = json.loads(path.with_suffix(".phases.json").read_text())
        turns = [i for i, p in enumerate(phases) if p["name"].endswith("rotate")]
        force = data["reaction_force"][np.isin(data["phase_id"], turns)]
        assert np.max(np.abs(force)) == 0.0
        assert areas.max() / areas.sum() > 0.995
        assert int(data["inverted_count"]) == 0
        np.testing.assert_allclose(score(data, target), record["surface_mm"], rtol=1e-10)
        pressed_height = max(
            float(data[f"stage_{i}_pressed"][:, 2].max() - data["floor"]) for i in range(4)
        )
        assert pressed_height < 0.0455
        rows.append(
            dict(
                **record,
                file=path.name,
                sha256=digest(path),
                footprint_iou=float(
                    np.count_nonzero(mask & target_mask) / np.count_nonzero(mask | target_mask)
                ),
                reconstruction_voxels_mm=[1.0, 1.25, 1.5],
                reconstruction_errors_mm=[score(data, target, h) for h in [0.001, 0.00125, 0.0015]],
                finer_quadrature_error_mm=mesh_distance_mm(
                    mesh, target.subdivide(1, subfilter="linear")
                ),
                connected_surface_area_fraction=float(areas.max() / areas.sum()),
                surface_components=int(len(areas)),
                maximum_floor_penetration_mm=float(penetration.max() * 1000),
                fraction_particles_below_floor=float(np.mean(penetration > 0)),
                volume_weighted_floor_penetration_mm=float(
                    np.average(penetration, weights=data["vol0"]) * 1000
                ),
                max_pressed_height_mm=pressed_height * 1000,
                peak_rotation_force_n=float(np.max(np.abs(force))),
                peak_force_per_finger_n=float(
                    np.linalg.norm(data["reaction_force"], axis=-1).max()
                ),
                reconstructed_volume_ml=float(mesh.volume * 1e6),
            )
        )
    assert len(rows) == 12
    save_json(
        folder / "additional_audit.json",
        dict(
            results=rows,
            scope="Fixed commands; numerical sensitivity, not a convergence proof; no registration or particle clipping",
        ),
    )
    print(json.dumps(rows, indent=2), flush=True)


def report(folder, dest=None, render_out=None, cross_folder=None):
    check(folder)
    assert (folder / "summary.json").exists()
    assert (folder / "additional_audit.json").exists()
    robot = json.loads((folder / "franka/audit.json").read_text())
    assert len(robot["results"]) == 4 and robot["speed_limits_checked"]
    output = folder if render_out is None else render_out
    output.mkdir(parents=True, exist_ok=True)
    images = output / "renders"
    images.mkdir(exist_ok=True)
    target = pv.read(folder / "target.vtp")
    target_mask = render(target, mask=True)[..., :3].mean(-1) < 128
    pictures = {"target": render(target, color="#b5b6b0")}
    methods = ["nominal", "identified"]
    headings = ["Target", "Nominal", "Identified"]
    colors = ["#bf7858", "#4a91b4"]
    outcome_folder = folder
    if cross_folder is not None:
        from experiments.robotics.x_letter_cross import check as check_cross

        check_cross(cross_folder)
        assert (cross_folder / "summary.json").exists()
        assert (cross_folder / "additional_audit.json").exists()
        assert digest(cross_folder / "target.vtp") == digest(folder / "target.vtp")
        methods = ["plan_A", "plan_B"]
        headings = ["Target", "Planned for A", "Planned for B"]
        colors = ["#4a91b4", "#bf7858"]
        outcome_folder = cross_folder
    records = {}
    for name in "AB":
        records[name] = {}
        for method, color in zip(methods, colors, strict=True):
            path = outcome_folder / f"baseline_{name}_{method}.npz"
            data = np.load(path)
            records[name][method] = json.loads(path.with_suffix(".json").read_text())
            pictures[f"{name}_{method}"] = render(
                surface(data["x_after_1s"], data["vol0"], h=0.00125), color=color
            )
    for key, picture in pictures.items():
        plt.imsave(images / f"{key}.png", picture)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 6.0))
    fig.subplots_adjust(left=0.025, right=0.995, bottom=0.065, top=0.94, wspace=0.025, hspace=0.13)
    for row, name in enumerate("AB"):
        for col, method in enumerate(["target", *methods]):
            ax = axes[row, col]
            ax.imshow(pictures["target" if method == "target" else f"{name}_{method}"])
            ax.axis("off")
            if row == 0:
                ax.set_title(headings[col], fontsize=12)
            if col == 0:
                ax.text(
                    0,
                    0.5,
                    name,
                    transform=ax.transAxes,
                    fontsize=13,
                    fontweight="bold",
                    color=COLORS[name],
                )
            else:
                ax.contour(
                    target_mask, [0.5], colors="#253139", linewidths=0.8, linestyles=[(0, (3, 2))]
                )
                ax.text(
                    0.5,
                    -0.035,
                    f"{records[name][method]['surface_mm']:.3f} mm",
                    ha="center",
                    transform=ax.transAxes,
                    fontsize=11,
                )
    fig.text(
        0.5,
        0.018,
        "Four position-controlled pinches; true-material outcomes 1 s after withdrawal.",
        ha="center",
        fontsize=10,
    )
    for suffix in ["png", "pdf"]:
        fig.savefig(output / f"comparison.{suffix}", dpi=220)
    plt.close(fig)

    fig = plt.figure(figsize=(7.2, 4.15))
    grid = fig.add_gridspec(
        2, 2, left=0.02, right=0.985, bottom=0.025, top=0.99, wspace=0.12, hspace=0.12
    )
    press_ax = fig.add_subplot(grid[0, 0])
    recovery = panel_a(press_ax, folder / "inputs", images)
    panel_b(fig.add_subplot(grid[0, 1]), folder / "inputs")
    ax = fig.add_subplot(grid[1, 0])
    label_panel(ax, "(c) Four position-controlled pinches")
    data = np.load(folder / "baseline_A_identified.npz")
    xml = folder / "franka/panda_model_snapshot/panda.xml"
    for stage, x, direction in [(2, 0.26, "y"), (3, 0.75, "x")]:
        ax.text(x, 0.83, f"{stage + 1}. Pinch along {direction}", ha="center", fontsize=7.4)
        picture = render_action(
            data, stage=stage, path=images / f"action_{stage}.png", robot_xml=xml
        )
        ia = ax.inset_axes([x - 0.23, 0.22, 0.46, 0.57])
        ia.imshow(picture)
        ia.axis("off")
        ax.text(x, 0.18, f"Opening {data['gaps_mm'][stage]:.1f} mm", ha="center", fontsize=7.1)
    gap_line = " → ".join(
        f"{axis}: {gap:.1f}" for axis, gap in zip("yxyx", data["gaps_mm"], strict=True)
    )
    ax.text(0.5, 0.085, gap_line + " mm", ha="center", fontsize=6.8)
    ax.text(
        0.5,
        0.025,
        "Open, lift, and turn between pinches.",
        ha="center",
        fontsize=6.8,
        color="#46515a",
    )
    ax = fig.add_subplot(grid[1, 1])
    label_panel(ax, "(d) Target and executed shapes")
    for x, label in zip(COLUMNS, headings, strict=True):
        ax.text(x, 0.83, label, ha="center", va="center", fontsize=7.0 if cross_folder else 7.4)
    for row, name in enumerate("AB"):
        ax.text(
            0.005,
            ROWS[row] + 0.11,
            name,
            color=COLORS[name],
            fontsize=8,
            fontweight="bold",
            va="center",
        )
        for col, method in enumerate(["target", *methods]):
            key = "target" if method == "target" else f"{name}_{method}"
            ia = ax.inset_axes([COLUMNS[col] - 0.145, ROWS[row], 0.29, 0.29])
            ia.imshow(pictures[key])
            ia.axis("off")
            if method != "target":
                ia.contour(
                    target_mask,
                    [0.5],
                    colors="#253139",
                    linewidths=0.55,
                    linestyles=[(0, (2.4, 1.8))],
                )
                ax.text(
                    COLUMNS[col],
                    ROWS[row] - 0.022,
                    f"{records[name][method]['surface_mm']:.2f} mm",
                    ha="center",
                    va="top",
                    fontsize=7.2,
                    zorder=20,
                )
    ax.text(
        0.5,
        0.025,
        "3D surface error 1 s after full withdrawal.",
        ha="center",
        fontsize=6.8,
        color="#46515a",
    )
    stem = output / "identification_plastic_shaping"
    for suffix in [".png", ".pdf"]:
        fig.savefig(stem.with_suffix(suffix), dpi=300, bbox_inches="tight", pad_inches=0.02)
    panel_bounds = press_ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(output / "panel_a.png", dpi=300, bbox_inches=panel_bounds.expanded(1.01, 1.01))
    panel_bounds = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(output / "panel_d.png", dpi=300, bbox_inches=panel_bounds.expanded(1.01, 1.01))
    plt.close(fig)
    snapshot = output / "report_source"
    snapshot.mkdir(exist_ok=True)
    source_hashes = {}
    for name in [
        "x_letter_study_report.py",
        "x_shaping_report.py",
        "x_shaping_franka.py",
        "plastic_shaping_figure.py",
        "hex_shaping_figure.py",
    ]:
        path = Path(__file__).with_name(name)
        shutil.copy2(path, snapshot / name)
        source_hashes[name] = digest(path)
    save_json(
        output / "figure_provenance.json",
        dict(
            records=records,
            study_directory=str(folder),
            outcome_directory=str(outcome_folder),
            outcome_columns=headings,
            outcome_colors=dict(zip(methods, colors, strict=True)),
            panel_a=recovery,
            target_sha256=digest(folder / "target.vtp"),
            renderer_sources=source_hashes,
            data={p.name: digest(p) for p in outcome_folder.glob("baseline_*.npz")},
            panel_c="Recorded third and fourth pinches of identified-A plan executed in true A; all four commanded gaps listed",
            scope="Simulation with a kinematic Panda hand illustration and proposed adapters; no hardware claim",
            camera="Common fixed view and scale for targets and all outcomes; no registration or rescaling",
        ),
    )
    if dest:
        for suffix in [".png", ".pdf"]:
            shutil.copy2(stem.with_suffix(suffix), dest.with_suffix(suffix))
    print(str(stem), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["audit", "report"])
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--dest", type=Path)
    p.add_argument("--render-out", type=Path, help="Separate rendering archive; keeps study outputs intact")
    p.add_argument("--cross-out", type=Path, help="Verified cross-material executions for panel (d)")
    a = p.parse_args()
    if a.stage == "audit":
        audit(a.out.resolve())
    else:
        report(a.out.resolve(), a.dest, a.render_out.resolve() if a.render_out else None,
               a.cross_out.resolve() if a.cross_out else None)
