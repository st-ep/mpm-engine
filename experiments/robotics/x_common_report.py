"""Figure and physical audits for the shared cylindrical identification study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pyvista as pv

# Configure common-study provenance and the 12 mm guard before shared imports.
from experiments.robotics import x_common_study as study
from experiments.robotics import x_work_report as shared
from experiments.robotics.x_work_probe_panel import probe_view, WINDOW, ANCHOR_HEIGHT, MOTION_COLOR
from experiments.robotics.plastic_shaping_figure import COLORS, surface
from experiments.robotics.plastic_shaping_study import save_json

shared.check = study.check
shared.MIN_GAP = .012

# Light surface tints retain shading; darker labels/curves use the same hues.
# Color identifies the actual material, independently of the planning model.
SURFACE_COLORS = {"A": "#7aacc7", "B": "#d39b76"}

# Panel coordinates are inches, so titles and image bands have the same
# spacing even though identification has less vertical space than shaping.
FIGURE_SIZE = (7.2, 3.60)
PANEL_WIDTH = 3.435
TOP_HEIGHT = 1.24
BOTTOM_HEIGHT = 2.12


def panel_frame(ax, title):
    height = ax.get_ylim()[1]
    ax.axis("off")
    ax.text(0, height - .01, title, va="top", fontsize=8.5,
            weight="bold", color="#202a32")
    ax.plot([0, PANEL_WIDTH], [height - .205] * 2, color="#d8dddf", lw=.55)


def inset_at(ax, bounds):
    x, y, width, height = bounds
    return ax.inset_axes([x / PANEL_WIDTH, y / ax.get_ylim()[1],
                          width / PANEL_WIDTH, height / ax.get_ylim()[1]])


def image_at(ax, picture, bounds):
    ia = inset_at(ax, bounds)
    ia.imshow(picture)
    ia.axis("off")
    return ia


def common_image_crop(pictures, masks, padding=10):
    """Crop only margins, using one union of all complete projected surfaces."""
    union = np.logical_or.reduce(list(masks.values()))
    ys, xs = np.nonzero(union)
    left, right = int(xs.min()) - padding, int(xs.max()) + padding + 1
    top, bottom = int(ys.min()) - padding, int(ys.max()) + padding + 1
    assert 0 <= left < right <= union.shape[1]
    assert 0 <= top < bottom <= union.shape[0]
    cropped = {}
    for key, picture in pictures.items():
        assert masks[key][top:bottom, left:right].sum() == masks[key].sum(), key
        cropped[key] = picture[top:bottom, left:right]
    return cropped, masks["target"][top:bottom, left:right], dict(
        columns=[left, right], rows=[top, bottom], padding_px=padding,
        original_window_px=[union.shape[1], union.shape[0]],
        display_window_px=[right - left, bottom - top],
        scope="One union crop for the target and all four outcomes; all surface pixels retained")


def check_render_inputs(folder):
    """Verify frozen experiment inputs, with drawing helpers versioned separately."""
    assert json.loads((folder / "protocol.json").read_text()) == json.loads(json.dumps(study.PROTOCOL))
    drawing_helpers = {"experiments/robotics/x_work_probe_panel.py",
                       "experiments/robotics/x_work_report.py"}
    for rel, sha in json.loads((folder / "source_sha256.json").read_text()).items():
        assert study.digest(folder / "source_snapshot" / rel) == sha, rel
        if rel not in drawing_helpers:
            assert study.digest(study.ROOT / rel) == sha, rel
    for manifest in ["input_sha256.json", "robot_input_sha256.json"]:
        for rel, sha in json.loads((folder / manifest).read_text()).items():
            assert study.digest(folder / rel) == sha, rel


def panel_a(ax, folder, images, xml):
    panel_frame(ax, "(a) Shared identification squeeze")
    data = {name: np.load(folder / f"inputs/probe_{name}.npz") for name in "AB"}
    np.testing.assert_array_equal(data["A"]["initial"], data["B"]["initial"])
    rendered = {}
    for name, action in [("pinch", True), ("A", False), ("B", False)]:
        material = "A" if action else name
        rendered[name] = probe_view(data[material], action=action, robot_xml=xml,
                                    color=SURFACE_COLORS[material])
    # All three pictures have the same displayed height. Released A/B have
    # identical cameras and a single crop; the action retains its wider view.
    released_bounds = [rendered[name][1]["specimen_bounds_px"] for name in "AB"]
    lower = min(bounds[0][1] for bounds in released_bounds)
    upper = max(bounds[1][1] for bounds in released_bounds)
    crop_height = int(np.ceil(upper - lower)) + 12
    top_released = int(np.floor(WINDOW[1] - upper)) - 6
    left_released = int(np.floor(min(b[0][0] for b in released_bounds))) - 6
    right_released = int(np.ceil(max(b[1][0] for b in released_bounds))) + 6
    height = .85
    widths = np.array([WINDOW[0], right_released - left_released,
                       right_released - left_released]) * height / crop_height
    gap = .05
    left = (PANEL_WIDTH - widths.sum() - 2 * gap) / 2
    assert left >= 0
    views = {}
    for name, width in zip(["pinch", "A", "B"], widths, strict=True):
        action = name == "pinch"
        picture, info = rendered[name]
        top = (int(np.ceil(WINDOW[1] - info["specimen_bounds_px"][0][1])) + 6 - crop_height
               if action else top_released)
        crop_left, crop_right = (0, WINDOW[0]) if action else (left_released, right_released)
        assert 0 <= top and top + crop_height <= WINDOW[1]
        assert top < WINDOW[1] - info["specimen_bounds_px"][1][1]
        assert top + crop_height > WINDOW[1] - info["specimen_bounds_px"][0][1]
        assert crop_left < info["specimen_bounds_px"][0][0]
        assert crop_right > info["specimen_bounds_px"][1][0]
        picture = picture[top:top + crop_height, crop_left:crop_right]
        info["display_crop_rows"] = [top, top + crop_height]
        info["display_crop_columns"] = [crop_left, crop_right]
        info["display_window_px"] = [crop_right - crop_left, crop_height]
        info["display_size_inches"] = [float(width), height]
        info["screen_inches_per_mm"] = height / crop_height * info["pixels_per_mm"]
        plt.imsave(images / f"probe_{name}.png", picture)
        ia = image_at(ax, picture, [left, .015, width, height])
        if action:
            for arrow in info["motion_arrows"]:
                endpoints = np.array([arrow["tail_px"], arrow["tip_px"]])
                endpoints[:, 0] -= crop_left
                endpoints[:, 1] = WINDOW[1] - endpoints[:, 1] - top
                assert np.all(endpoints > 0) and np.all(endpoints < [crop_right - crop_left, crop_height])
                ia.add_patch(FancyArrowPatch(endpoints[0], endpoints[1], arrowstyle="->",
                    mutation_scale=8, linewidth=1.3, color=MOTION_COLOR,
                    shrinkA=0, shrinkB=0, zorder=20))
        else:
            # Initial 60 mm square, projected with the released specimen's
            # calibrated camera and translated by exactly the same crop.
            center = np.array([WINDOW[0] / 2, WINDOW[1] * (1 - ANCHOR_HEIGHT)])
            square = center + np.array([[-30, -30], [30, -30], [30, 30], [-30, 30], [-30, -30]]) * info["pixels_per_mm"]
            square -= [crop_left, top]
            assert np.all(square >= 0) and np.all(square < [crop_right - crop_left, crop_height])
            ia.plot(square[:, 0], square[:, 1], color="#425868", lw=.6, dashes=(3, 2))
        ax.text(left + width / 2, TOP_HEIGHT - .30,
                "Squeeze" if action else f"Released {name}", ha="center", va="center",
                fontsize=7.4, weight="bold" if action else "normal",
                color=MOTION_COLOR if action else COLORS[name])
        views[name] = info
        left += width + gap
    save_json(images / "probe_view_calibration.json", dict(views=views,
        image_height_inches=height, image_bottom_inches=.015,
        scope="Equal visible image heights with labels outside the image band. Released A/B share a physical scale and crop; the action specimen is 0.733 times their screen scale to show more of the gripper. Action has a separate vertical crop and geometric foreshortening. Every specimen is complete."))


def panel_b(panel, folder):
    panel_frame(panel, "(b) Held-out force prediction")
    ax = inset_at(panel, [.34, .265, PANEL_WIDTH - .38, .705])
    for n in "AB":
        true = np.load(folder / f"validation/true_{n}.npz")
        predicted = np.load(folder / f"validation/identified_{n}.npz")
        active = np.isin(true["phase_id"], [2, 3, 4])
        t = true["time"][active] - true["probe_start_s"]
        force = (true["reaction_force"][active, :, 1] * [-1, 1]).mean(axis=1)
        fitted = (predicted["reaction_force"][active, :, 1] * [-1, 1]).mean(axis=1)
        ax.plot(t, force, color=COLORS[n], lw=1.15)
        ax.plot(t, fitted, lw=0, color=COLORS[n], marker="o", markersize=2.4,
                markerfacecolor="white", markeredgewidth=.65, markevery=16)
        peak = np.argmax(force)
        ax.annotate(n, (t[peak], force[peak]), xytext=(5, 2), textcoords="offset points",
                    color=COLORS[n], fontsize=7.4, weight="bold")
    ax.set(xlabel="Time (s)", ylabel="Force (N)", xlim=(0, 1.7))
    ax.set_xticks([0, .5, 1, 1.5])
    ax.set_yticks([0, 5, 10])
    ax.set_ylim(-.5, ax.get_ylim()[1] * 1.14)
    ax.tick_params(labelsize=7.4, pad=2, length=3)
    ax.xaxis.label.set_size(7.4)
    ax.yaxis.label.set_size(7.4)
    ax.xaxis.labelpad = ax.yaxis.labelpad = 2
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(.7)
    ax.grid(axis="y", alpha=.15)
    ax.legend(handles=[Line2D([], [], color="#333333", lw=1.2, label="True"),
              Line2D([], [], color="#333333", marker="o", markerfacecolor="white",
                     lw=0, markersize=2.5, label="Identified")],
              loc="upper right", fontsize=7.4, frameon=False,
              handlelength=1.5, labelspacing=.35, borderaxespad=.2)


def panel_c(ax, folder, images, xml):
    panel_frame(ax, "(c) Work-controlled shaping")
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
        ax.text(cx, .31, f"Work: {data['requested_work_j'][stage] * 1000:.1f} mJ",
                ha="center", va="center", fontsize=7.2, zorder=20)


def report(folder, output, dest=None):
    check_render_inputs(folder)
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
    for name in ["x_common_report.py", "x_work_report.py", "x_work_probe_panel.py", "x_force_report.py",
                 "x_shaping_report.py", "x_shaping_franka.py", "plastic_shaping_figure.py", "hex_shaping_surface.py"]:
        shutil.copy2(Path(__file__).with_name(name), source / name)
    save_json(output / "figure_provenance.json", dict(study_directory=str(folder.resolve()),
        protocol_sha256=study.digest(folder / "protocol.json"),
        inputs_sha256=study.digest(folder / "input_sha256.json"),
        records=records, panel_a="Actual common identification squeeze and released calibration specimens",
        panel_b="New deeper faster cylindrical validation, excluded from fitting",
        panel_c="New matched-A work execution, final two pinches",
        panel_d="Rows are true materials; middle column uses the matching identified model, right column the other material's identified model",
        panel_d_columns=column_labels, panel_d_row_models=row_models,
        layout=layout,
        material_surface_colors=SURFACE_COLORS, material_label_curve_colors=COLORS,
        material_color_encoding="Visualization labels for the actual material, not physical appearance or planning model; probe action and shaping actions use A",
        sources={p.name: study.digest(p) for p in source.iterdir()},
        scope="Simulation with supplied x, v and elastic F; no hardware-state estimation or hardware execution"))
    if dest is not None:
        for suffix in [".png", ".pdf"]:
            shutil.copy2(stem.with_suffix(suffix), dest.with_suffix(suffix))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["audit", "diagnostics", "report"])
    parser.add_argument("--out", type=Path, default=study.OUT)
    parser.add_argument("--render-out", type=Path, default=study.OUT / "figure")
    parser.add_argument("--dest", type=Path)
    args = parser.parse_args()
    if args.stage == "audit":
        shared.audit(args.out)
    elif args.stage == "diagnostics":
        shared.diagnostics(args.out)
    else:
        report(args.out, args.render_out, args.dest)
