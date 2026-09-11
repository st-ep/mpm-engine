"""Active force-pulse shaping figure, retaining the compact four-panel layout.

Identification/held-out validation are copied from the frozen common probe;
shaping images and scores come only from the force-pulse true executions.
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
from experiments.robotics import x_force_pulse_plan as planning
from experiments.robotics import x_work_report as shared
from experiments.robotics.x_common_report import (
    panel_a, panel_b, panel_frame, image_at, common_image_crop,
    FIGURE_SIZE, PANEL_WIDTH, TOP_HEIGHT, BOTTOM_HEIGHT, SURFACE_COLORS)
from experiments.robotics.plastic_shaping_figure import COLORS, surface
from experiments.robotics.plastic_shaping_study import save_json

study = planning.study


def panel_c(ax, folder, images, xml):
    panel_frame(ax, "(c) Force-controlled shaping")
    data = np.load(folder / "baseline_A_plan_A.npz")
    width = (PANEL_WIDTH - .08) / 2
    height = width * 650 / 700
    top = BOTTOM_HEIGHT - .37
    for stage, left, direction in [(2, 0, "y"), (3, width + .08, "x")]:
        cx = left + width / 2
        ax.text(cx, BOTTOM_HEIGHT - .30, f"{stage + 1}. Pinch along {direction}",
                ha="center", va="center", fontsize=7.4)
        picture = shared.render_action(data, stage=stage, color=SURFACE_COLORS["A"],
                                       path=images / f"action_{stage}.png", robot_xml=xml)
        image_at(ax, picture, [left, top - height, width, height])
        ax.text(cx, .31, f"{data['peaks_n'][stage]:.2f} N · {data['durations_s'][stage]:.2f} s",
                ha="center", va="center", fontsize=7.2, zorder=20)


def report(folder, output, dest=None):
    planning.stage(folder)
    audit = json.loads((folder / "additional_audit.json").read_text())
    robot = json.loads((folder / "franka/audit.json").read_text())
    assert len(audit["results"]) == 12 and len(robot["results"]) == 4
    assert robot["speed_limits_checked"]
    assert json.loads((folder / "identification_validation.json").read_text())["exact_same_probe_trajectory"]
    output.mkdir(parents=True, exist_ok=True)
    images = output / "renders"
    images.mkdir(exist_ok=True)
    target = pv.read(folder / "target.vtp")
    masks = {"target": shared.render(target, mask=True)[..., :3].mean(-1) < 128}
    pictures = {"target": shared.render(target, color="#b5b6b0")}
    records = {}
    for n in "AB":
        for planned in "AB":
            key = f"{n}_plan_{planned}"
            path = folder / f"baseline_{key}.npz"
            data = np.load(path)
            mesh = surface(data["x_after_1s"], data["vol0"], h=.00125)
            pictures[key] = shared.render(mesh, color=SURFACE_COLORS[n])
            masks[key] = shared.render(mesh, mask=True)[..., :3].mean(-1) < 128
            records[key] = json.loads(path.with_suffix(".json").read_text())
            assert records[key]["material"] == n and records[key]["planned_for"] == planned
            assert study.digest(path) == records[key]["data_sha256"]
            audited = next(r for r in audit["results"] if r["file"] == path.name)
            assert audited["data_sha256"] == records[key]["data_sha256"]
    for key, picture in pictures.items():
        plt.imsave(images / f"{key}.png", picture)
    pictures, mask, outcome_crop = common_image_crop(pictures, masks)
    save_json(images / "outcome_crop.json", outcome_crop)
    for key, picture in pictures.items():
        plt.imsave(images / f"{key}_display.png", picture)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "pdf.fonttype": 42})
    fig = plt.figure(figsize=FIGURE_SIZE)
    xml = folder / "franka/panda_model_snapshot/panda.xml"
    panels = []
    for left, bottom, height in [(x, y, h) for y, h in [(2.345, TOP_HEIGHT), (.035, BOTTOM_HEIGHT)]
                                  for x in [.035, 3.72]]:
        ax = fig.add_axes([left / FIGURE_SIZE[0], bottom / FIGURE_SIZE[1],
                           PANEL_WIDTH / FIGURE_SIZE[0], height / FIGURE_SIZE[1]])
        ax.set(xlim=(0, PANEL_WIDTH), ylim=(0, height))
        panels.append(ax)
    panel_a(panels[0], folder, images, xml)
    panel_b(panels[1], folder)
    panel_c(panels[2], folder, images, xml)
    ax = panels[3]
    panel_frame(ax, "(d) Target and executed shapes")
    column_labels = ["Target", "Matched model", "Swapped model"]
    row_models = {name: [None, name, "B" if name == "A" else "A"] for name in "AB"}
    columns = [.54, 1.68, 2.84]
    for x, label in zip(columns, column_labels, strict=True):
        ax.text(x, BOTTOM_HEIGHT - .30, label, ha="center", va="center", fontsize=7.)
    guarded = {key: "travel_guard" in record["stops"] for key, record in records.items()}
    image_height = .72
    image_width = image_height * mask.shape[1] / mask.shape[0]
    for row, name in enumerate("AB"):
        bottom = [BOTTOM_HEIGHT - .37 - image_height, .12][row]
        ax.text(.025, bottom + image_height / 2, name, color=COLORS[name], fontsize=8,
                weight="bold", va="center")
        for col, planned in enumerate(row_models[name]):
            key = "target" if planned is None else f"{name}_plan_{planned}"
            ia = image_at(ax, pictures[key], [columns[col] - image_width / 2, bottom,
                                             image_width, image_height])
            if planned is not None:
                ia.contour(mask, [.5], colors="#253139", linewidths=.55, linestyles=[(0, (2.4, 1.8))])
                marker = "†" if guarded[key] else ""
                ax.text(columns[col], bottom - .025, f"{records[key]['surface_mm']:.2f} mm{marker}",
                        ha="center", va="top", fontsize=7.2, zorder=20)
    layout = dict(figure_size_inches=FIGURE_SIZE, panel_width_inches=PANEL_WIDTH,
        top_panel_height_inches=TOP_HEIGHT, bottom_panel_height_inches=BOTTOM_HEIGHT,
        panel_d_image_size_inches=[image_width, image_height], panel_d_crop=outcome_crop,
        panel_c_image_size_inches=[(PANEL_WIDTH - .08) / 2, (PANEL_WIDTH - .08) / 2 * 650 / 700])
    save_json(output / "layout.json", layout)
    stem = output / "identification_plastic_shaping"
    for suffix in [".png", ".pdf"]:
        fig.savefig(stem.with_suffix(suffix), dpi=300, bbox_inches="tight", pad_inches=.02)
    for name, ax in zip("abcd", panels, strict=True):
        bounds = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(output / f"panel_{name}.png", dpi=300, bbox_inches=bounds.expanded(1.01, 1.01))
    plt.close(fig)
    source = output / "report_source"
    source.mkdir(exist_ok=True)
    for name in ["x_force_pulse_report.py", "x_common_report.py", "x_work_report.py", "x_work_probe_panel.py", "x_force_report.py",
                 "x_shaping_report.py", "x_shaping_franka.py", "plastic_shaping_figure.py", "hex_shaping_surface.py"]:
        shutil.copy2(Path(__file__).with_name(name), source / name)
    save_json(output / "figure_provenance.json", dict(study_directory=str(folder.resolve()),
        protocol_sha256=study.digest(folder / "protocol.json"),
        planning_protocol_sha256=study.digest(folder / "planning_protocol.json"),
        selection_protocol_sha256=study.digest(folder / "selection_protocol.json"),
        selected_plan_sha256={n: study.digest(folder / f"plans/{n}/selected.json") for n in "AB"},
        audit_sha256=study.digest(folder / "additional_audit.json"),
        inputs_sha256=study.digest(folder / "input_sha256.json"),
        records=records, panel_a="Actual common identification squeeze and released calibration specimens",
        panel_b="New deeper faster cylindrical validation, excluded from fitting",
        panel_c="Matched-A force-pulse execution, final two pinches; labels are commanded peak mean force per finger and total pulse duration",
        panel_d="Rows are true materials; middle column uses the matching identified model, right column the other material's identified model",
        panel_d_columns=column_labels, panel_d_row_models=row_models,
        layout=layout,
        material_surface_colors=SURFACE_COLORS, material_label_curve_colors=COLORS,
        material_color_encoding="Visualization labels for the actual material, not physical appearance or planning model; probe action and shaping actions use A",
        sources={p.name: study.digest(p) for p in source.iterdir()},
        scope="Simulation with supplied x, v and elastic F, and ideal per-finger force sensing during shaping; stock Franka Hand force-pulse operation is not established"))
    if dest is not None:
        for suffix in [".png", ".pdf"]:
            shutil.copy2(stem.with_suffix(suffix), dest.with_suffix(suffix))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=study.OUT)
    parser.add_argument("--render-out", type=Path, required=True)
    parser.add_argument("--dest", type=Path)
    args = parser.parse_args()
    report(args.out, args.render_out, args.dest)
