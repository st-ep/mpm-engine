"""Views of recorded cylindrical-finger X experiments, with common world scale."""
from __future__ import annotations

import argparse
import json
import hashlib
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from experiments.robotics.x_shaping import CONFIG, OUT, target_mesh, score
from experiments.robotics.plastic_shaping_figure import surface


def render(data=None, *, stage=None, top=False, color="#4a91b4", path=None, mask=False, robot_xml=None):
    mesh = target_mesh() if data is None else surface(data["x_after_1s"] if stage is None else data[f"stage_{stage}_pressed"], data["vol0"], h=CONFIG["voxel"])
    p = pv.Plotter(off_screen=True, window_size=(700, 650)); p.set_background("white")
    p.add_mesh(mesh, color="black" if mask else ("#b5b6b0" if data is None else color), smooth_shading=True, split_sharp_edges=True,
               ambient=.4, diffuse=.7, specular=.1, lighting=not mask)
    floor = CONFIG["floor"]; c = CONFIG["domain"]/2
    if not mask:
        p.add_mesh(pv.Plane(center=(c, c, floor-.0001), i_size=.105, j_size=.105), color="#f1f2f2", lighting=False)
    if stage is not None:
        centers = data[f"stage_{stage}_centers"]
        for center in centers:
            p.add_mesh(pv.Cylinder(center=center, direction=(0, 0, 1), radius=CONFIG["radius"],
                                  height=CONFIG["finger_height"], resolution=80),
                       color="#697680", opacity=.5, smooth_shading=True)
        # Actual Panda hand mesh, posed by IK. The cylindrical contacts are MPM tools.
        from experiments.robotics.x_shaping_franka import Panda
        angle = float(data["angles_deg"][stage])*np.pi/180
        for hand_mesh, hand_color in Panda(robot_xml).meshes(centers, angle):
            p.add_mesh(hand_mesh, color=hand_color, smooth_shading=True, split_sharp_edges=True,
                       ambient=.6, diffuse=.5, specular=.1)
    focal = np.array([c, c, floor+.017])
    p.camera_position = [focal+np.array([0., -.13, .24]) if not top else focal+[0, 0, .3], focal, (0, 1, 0)]
    if stage is not None:
        focal = np.array([c, c, floor+.025])
        p.camera_position = [focal+np.array([.20, -.20, .055]), focal, (0, 0, 1)]
    p.enable_parallel_projection(); p.camera.parallel_scale = .048 if stage is None else .058
    p.enable_anti_aliasing("ssaa")
    img = p.screenshot(str(path) if path else None, return_img=True); p.close()
    return img


def study_report(folder, execution=None):
    from experiments.robotics.x_shaping_study import check_sources
    from experiments.robotics.plastic_shaping_figure import panel_a, label_panel, COLUMNS, ROWS, COLORS
    from experiments.robotics.hex_shaping_figure import panel_b
    from experiments.robotics.plastic_shaping_study import save_json
    check_sources(folder)
    data_folder = folder if execution is None else execution
    if execution is not None:
        protocol = json.loads((execution/"execution_protocol.json").read_text())
        assert Path(protocol["planner_source"]).resolve() == folder.resolve()
    images = folder/"renders"; images.mkdir(exist_ok=True)
    palette = dict(nominal="#bf7858", identified="#4a91b4", oracle="#586b68")
    mask = render(mask=True)[..., :3].mean(axis=-1) < 128
    pictures = {"target": render()}
    records = {}
    methods = ["nominal", "identified"]
    if all((data_folder/f"baseline_{name}_oracle.npz").exists() for name in "AB"):
        methods.append("oracle")
    for name in "AB":
        records[name] = {}
        for method in methods:
            path = data_folder/f"baseline_{name}_{method}.npz"
            pictures[f"{name}_{method}"] = render(np.load(path), color=palette[method], path=images/f"{name}_{method}.png")
            records[name][method] = json.loads(path.with_suffix(".json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, len(methods)+1, figsize=(2.5*(len(methods)+1), 5.4))
    fig.subplots_adjust(left=.025, right=.99, bottom=.07, top=.92, wspace=.02, hspace=.12)
    for row, name in enumerate("AB"):
        for col, method in enumerate(["target", *methods]):
            ax = axes[row, col]; key = "target" if method == "target" else f"{name}_{method}"
            ax.imshow(pictures[key]); ax.axis("off")
            if row == 0:
                ax.set_title(dict(target="Target", nominal="Nominal", identified="Identified", oracle="True parameters")[method], fontsize=11)
            if col == 0:
                ax.text(0, .5, name, transform=ax.transAxes, fontsize=12, fontweight="bold")
            else:
                ax.contour(mask, [.5], colors="black", linewidths=.7, linestyles=[(0, (2.4, 1.8))])
                ax.text(.5, .0, f"{records[name][method]['surface_mm']:.3f} mm", ha="center", transform=ax.transAxes, fontsize=10)
    fig.text(.5, .02, "Two position-controlled pinches. Outcomes 1 s after withdrawal; common target, camera, and scale.", ha="center", fontsize=9)
    fig.savefig(folder/"comparison.png", dpi=200); fig.savefig(folder/"comparison.pdf"); plt.close(fig)
    fig = plt.figure(figsize=(7.2, 4.15))
    grid = fig.add_gridspec(2, 2, left=.02, right=.985, bottom=.025, top=.99, wspace=.12, hspace=.12)
    panel_a(fig.add_subplot(grid[0, 0]), folder/"inputs", images)
    panel_b(fig.add_subplot(grid[0, 1]), folder/"inputs")
    ax = fig.add_subplot(grid[1, 0]); label_panel(ax, "(c) Position-controlled pinches")
    data = np.load(data_folder/"baseline_A_identified.npz")
    for stage, x in enumerate([.26, .75]):
        ax.text(x, .83, f"{stage+1}. Close at {stage*90}°", ha="center", fontsize=7.4)
        xml = data_folder/"franka/panda_model_snapshot/panda.xml"
        assert xml.exists(), "Run the Franka audit to freeze robot assets before the paper render"
        img = render(data, stage=stage, path=images/f"action_{stage}.png", robot_xml=xml)
        ia = ax.inset_axes([x-.23, .19, .46, .60]); ia.imshow(img); ia.axis("off")
        ax.text(x, .15, f"Final opening {data['gaps_mm'][stage]:.1f} mm", ha="center", fontsize=7.1)
    ax.text(.5, .045, "Open, lift, and rotate between pinches.", ha="center", fontsize=6.8, color="#46515a")
    ax = fig.add_subplot(grid[1, 1]); label_panel(ax, "(d) Target and executed shapes")
    for x, label in zip(COLUMNS, ["Target", "Nominal", "Identified"], strict=True):
        ax.text(x, .83, label, ha="center", va="center", fontsize=7.4)
    for row, name in enumerate("AB"):
        ax.text(.005, ROWS[row]+.11, name, color=COLORS[name], fontsize=8, fontweight="bold", va="center")
        for col, method in enumerate(["target", "nominal", "identified"]):
            key = "target" if method == "target" else f"{name}_{method}"
            ia = ax.inset_axes([COLUMNS[col]-.145, ROWS[row], .29, .29]); ia.imshow(pictures[key]); ia.axis("off")
            if method != "target":
                ia.contour(mask, [.5], colors="black", linewidths=.55, linestyles=[(0, (2.4, 1.8))])
                ax.text(COLUMNS[col], ROWS[row]-.022, f"{records[name][method]['surface_mm']:.2f} mm", ha="center", va="top", fontsize=7.2)
    ax.text(.5, .025, "Surface error 1 s after full withdrawal.", ha="center", fontsize=6.8, color="#46515a")
    stem = folder/"identification_plastic_shaping_x"
    for suffix, dpi in [(".png", 260), (".pdf", 300)]:
        fig.savefig(stem.with_suffix(suffix), dpi=dpi, bbox_inches="tight", pad_inches=.02)
    plt.close(fig)
    source = folder/"report_source"; source.mkdir(exist_ok=True)
    for name in ["x_shaping_report.py", "x_shaping_franka.py"]:
        shutil.copy2(Path(__file__).with_name(name), source/name)
    save_json(folder/"figure_provenance.json", dict(records=records, execution_directory=str(data_folder.resolve()),
              renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              data={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in data_folder.glob("baseline_*.npz")},
              scope="Raw simulated outcomes; Panda hand is a kinematic visualization with proposed adapters; no hardware execution claim",
              camera="Common fixed view and scale for all outcomes; no alignment or rescaling"))
    return stem


def overview(folder):
    files = sorted(folder.glob("*.npz"))
    configs = [json.loads(path.with_suffix(".config.json").read_text())["config"] for path in files]
    for config in configs[1:]:
        for key in ["domain", "floor", "target_width", "target_notch_radius", "voxel"]:
            assert config[key] == configs[0][key]
    if configs:
        CONFIG.update(configs[0])
    fig, axes = plt.subplots(2, len(files)+1, figsize=(3*(len(files)+1), 6.))
    for col, path in enumerate([None, *files]):
        data = None if path is None else np.load(path)
        for row in range(2):
            axes[row, col].imshow(render(data, top=row == 1)); axes[row, col].axis("off")
        title = "Fixed X target" if path is None else path.stem.replace("true_", "")
        if path is not None:
            title += f"\n{score(data):.3f} mm"
        axes[0, col].set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(folder/"overview.png", dpi=150); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--dest", type=Path, help="Explicit manuscript figure stem; rebuild PDF immediately after use")
    p.add_argument("--execution", type=Path, help="Explicit directory of verified retimed executions")
    args = p.parse_args()
    if (args.out/"inputs/models.json").exists():
        stem = study_report(args.out, args.execution)
        if args.dest:
            assert (args.out/"summary.json").exists(), "Verify the completed study before adopting its figure"
            data_folder = args.out if args.execution is None else args.execution
            assert (data_folder/"summary.json").exists()
            assert (data_folder/"additional_audit.json").exists(), "Complete the additional geometry audit before adoption"
            robot_audit = json.loads((data_folder/"franka/audit.json").read_text())
            assert len(robot_audit["results"]) == 6
            assert robot_audit["speed_limits_checked"], "Check the reported execution's Panda speeds before adoption"
            for suffix in [".png", ".pdf"]:
                shutil.copy2(stem.with_suffix(suffix), args.dest.with_suffix(suffix))
    else:
        assert args.dest is None
        overview(args.out)


if __name__ == "__main__":
    main()
