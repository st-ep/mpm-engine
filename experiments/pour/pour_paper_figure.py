"""Compose the private manuscript's pouring figure from verified data and renders.

No dynamics, parameter fitting, photo reassignment, or volume reading occurs here.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.container import ErrorbarContainer
from matplotlib.legend_handler import HandlerErrorbar
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out/pour_figure_final_20260911"
DATA = ROOT / "out/pour_hardware_receiver_remap_review_20260911"
FIGS = ROOT / "paper/icra2027/figs"
SIZE = (7.2, 4.0)
INK, BLUE, ORANGE = "#202a32", "#2375aa", "#c96932"


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def compose(install=False):
    summary_path = DATA / "summary.csv"
    revision = json.loads((DATA / "share_plot_revision.json").read_text())
    assert digest(summary_path) == revision["unchanged_summary_sha256"]
    assert digest(DATA / "receiver_readings.csv") == revision["unchanged_receiver_readings_sha256"]
    provenance = json.loads((DATA / "provenance.json").read_text())
    checked = {str(summary_path.relative_to(ROOT)): digest(summary_path)}
    for key in ("handoff", "dataset_order", "simulation_provenance"):
        item = provenance[key]
        assert digest(ROOT / item["path"]) == item["sha256"]
        checked[item["path"]] = item["sha256"]
    for item in provenance["predictions"]:
        for key in ("result", "review"):
            path, sha = item[f"{key}_path"], item[f"{key}_sha256"]
            assert digest(ROOT / path) == sha
            checked[path] = sha
    rows = list(csv.DictReader(summary_path.open()))
    target = np.array([float(r["target_ml"]) for r in rows])
    measured = np.array([[float(r[f"seed{k}_ml"]) for k in range(1, 6)] for r in rows])
    mean, sd = measured.mean(1), measured.std(1, ddof=1)
    np.testing.assert_array_equal(target, [60, 80, 100, 120, 140, 160])
    np.testing.assert_allclose(mean, [float(r["mean_ml"]) for r in rows], atol=1e-12)
    np.testing.assert_allclose(sd, [float(r["sample_sd_ml"]) for r in rows], atol=1e-12)
    predicted = np.array([float(r["mpm_receiver_ml"]) for r in rows])
    np.testing.assert_allclose(predicted, [p["predicted_ml"] for p in provenance["predictions"][:6]])
    scene = json.loads((OUT / "render_assets/scene.json").read_text())
    assert "surface_reconstruction" in scene, "The final figure requires the actual liquid snapshot"
    assert scene["surface_reconstruction"]["snapshot_sha256"] == digest(OUT / "replay/snapshot.npz")
    render = json.loads((OUT / "render_provenance.json").read_text())
    assert render["scene_sha256"] == digest(OUT / "render_assets/scene.json")
    assert render["snapshot_sha256"] == digest(OUT / "replay/snapshot.npz")
    assert render["rendered_image_sha256"] == digest(OUT / "mpm_raw.png")
    if install:
        assert json.loads((OUT / "replay/verification.json").read_text())["render_snapshot_accepted"]
    scale = scene["render_scale"]
    crop = tuple(scene["upright_crop_xyxy"])
    raw = Image.open(OUT / "mpm_raw.png")
    assert raw.size == (848*scale, 480*scale)
    simulation = raw.transpose(Image.Transpose.ROTATE_90).crop(tuple(c*scale for c in crop))
    simulation.save(OUT / "mpm_crop.png")
    hardware = Image.open(OUT / "hardware_crop.png")
    group_path = ROOT / "pouring_real_data/pouring_figs/all_6_levels.jpg"
    group_crop = (30, 1770, 5440, 2960)
    group = Image.open(group_path).crop(group_crop)
    group.save(OUT / "six_cups_crop.png")
    group_mapping_path = group_path.with_name("group_photo_mapping.csv")
    group_record_path = group_path.with_name("group_photo_provenance.json")
    group_record = json.loads(group_record_path.read_text())
    assert digest(group_path) == group_record["group_photo_sha256"]
    assert digest(group_mapping_path) == group_record["mapping_csv_sha256"]
    group_labels = list(csv.DictReader(group_mapping_path.open()))
    np.testing.assert_array_equal([float(r["target_ml"]) for r in group_labels], target)
    np.testing.assert_array_equal([float(r["receiver_reading_ml"]) for r in group_labels], measured[:, 1])
    for row in group_labels:
        assert int(row["seed"]) == 2
        assert int(row["display_receiver_ml"]) == round(float(row["receiver_reading_ml"]))
        assert digest(group_path.parent / row["measurement_image"]) == row["measurement_image_sha256"]

    # Typography follows experiments/robotics/x_common_report.py, without
    # importing its simulation/reporting dependencies or changing its files.
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
        "ytick.color": INK, "axes.linewidth": .65})
    fig = plt.figure(figsize=SIZE, facecolor="white")

    def axes(bounds):
        x, y, w, h = bounds
        return fig.add_axes([x/SIZE[0], y/SIZE[1], w/SIZE[0], h/SIZE[1]])

    def title(x, y, width, text):
        fig.text(x/SIZE[0], y/SIZE[1], text, fontsize=8.5, weight="bold", va="top")
        fig.add_artist(Line2D([x/SIZE[0], (x+width)/SIZE[0]],
            [(y-.195)/SIZE[1]]*2, transform=fig.transFigure, color="#d8dddf", lw=.55))

    title(0, 3.99, 3.435, "(a) 60° identification pour")
    title(3.765, 3.99, 3.435, "(b) Target volume results")
    image_h = 1.755
    image_w = image_h*hardware.width/hardware.height
    for x, picture, label in [(0, hardware, "Hardware"), (3.435-image_w, simulation, "MPM")]:
        ax = axes((x, 1.90, image_w, image_h))
        ax.imshow(picture, interpolation="lanczos")
        ax.set_axis_off()
        fig.text((x+image_w/2)/SIZE[0], 3.72/SIZE[1], label,
            ha="center", va="center", fontsize=7.4, color=BLUE if label == "MPM" else ORANGE)

    ax = axes((4.12, 2.16, 2.94, 1.54))
    volume_ax = ax
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e4e7e9", lw=.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#68737b")
    ax.plot([50, 170], [50, 170], color="#9aa1a8", ls="--", lw=.85, zorder=1)
    ax.plot(target, predicted, color=BLUE, lw=1., marker="s", markersize=3.3,
        markerfacecolor="white", markeredgewidth=.85, zorder=3)
    hardware_bars = ax.errorbar(target, mean, yerr=sd, color=ORANGE, lw=.9, marker="o",
        markersize=2.1, capsize=3.2, capthick=1.05, elinewidth=1.05, zorder=4,
        label="Hardware mean ± SD")
    ax.set(xlim=(50, 170), ylim=(50, 170), xticks=target, yticks=target)
    ax.tick_params(labelsize=7.4, width=.6, length=2.8, pad=2)
    ax.set_xlabel("Target volume (mL)", fontsize=7.4, labelpad=4)
    ax.set_ylabel("Receiver volume (mL)", fontsize=7.4, labelpad=3)
    handles = [
        Line2D([], [], color="#9aa1a8", ls="--", lw=.85, label="Target"),
        Line2D([], [], color=BLUE, lw=1., marker="s", markersize=3.3,
            markerfacecolor="white", markeredgewidth=.85, label="MPM prediction"),
        hardware_bars,
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True, fancybox=False,
        facecolor="white", edgecolor="none", framealpha=1., fontsize=7.4,
        handlelength=1.7, handletextpad=.6, borderpad=.15, labelspacing=.45,
        handler_map={ErrorbarContainer: HandlerErrorbar(yerr_size=.65)})

    title(0, 1.805, 7.2, "(c) Poured volumes · repeat 2")
    fig.text(7.18/SIZE[0], 1.795/SIZE[1], "Target → measured (mL)",
             ha="right", va="top", fontsize=7.4, color=INK)
    panorama_h = 7.2*group.height/group.width
    ax = axes((0, 0, 7.2, panorama_h))
    ax.imshow(group, interpolation="lanczos")
    ax.set_axis_off()
    for row in group_labels:
        pieces = [TextArea(str(int(float(row["target_ml"]))), textprops=dict(color=INK, fontsize=8, weight="bold")),
                  TextArea(" → ", textprops=dict(color="#63717c", fontsize=8)),
                  TextArea(row["display_receiver_ml"], textprops=dict(color="#9d431c", fontsize=8, weight="bold"))]
        label = HPacker(children=pieces, align="center", pad=0, sep=0)
        ax.add_artist(AnnotationBbox(label,
            (float(row["label_center_original_x_px"])-group_crop[0],
             float(row["label_center_original_y_px"])-group_crop[1]),
            xycoords="data", frameon=True, box_alignment=(.5, .5),
            bboxprops=dict(boxstyle="round,pad=0.28,rounding_size=0.12", fc="white", ec="none", alpha=.94)))
    fig.canvas.draw()
    identity_vector = np.diff(volume_ax.transData.transform([[60, 60], [160, 160]]), axis=0)[0]
    identity_angle = float(np.degrees(np.arctan2(identity_vector[1], identity_vector[0])))
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"pouring.{suffix}", dpi=450, facecolor="white", pad_inches=0,
                    metadata={"Creator": "MPM pouring figure"} if suffix == "pdf" else None)
    plt.close(fig)
    for path in [OUT / "mpm_raw.png", OUT / "hardware_raw.png", OUT / "replay/snapshot.npz",
                 OUT / "render_assets/scene.json", OUT / "render_provenance.json",
                 OUT / "camera_verification.json", OUT / "replay/verification.json",
                 group_path, group_mapping_path, group_record_path, DATA / "receiver_readings.csv"]:
        checked[str(path.relative_to(ROOT))] = digest(path)
    manifest = dict(figure_size_inches=SIZE, primary_readings="receiver cups",
        observations_in_statistics=int(measured.size), repetitions_per_target=5,
        plotted_individual_observations=0, plotted_hardware_summaries=len(target),
        error_bars="Sample standard deviation (ddof=1), not total measurement uncertainty",
        individual_target_coordinates_ml=np.repeat(target[:, None], 5, axis=1).tolist(),
        individual_x_display_note="Individual points omitted at user's request; all five readings per target enter the unchanged mean and SD",
        axis_limits_ml={"target": [50, 170], "receiver": [50, 170]},
        equal_physical_axis_scaling=False, plot_size_inches=[2.94, 1.54],
        identity_line_display_angle_degrees=identity_angle,
        error_bar_style=dict(capsize_pt=3.2, capthick_pt=1.05, linewidth_pt=1.05,
            mean_marker_size_pt=2.1, legend="Actual error-bar container with a vertical capped SD symbol"),
        origin="Shared cropped numeric ranges and target line y=x; wide layout intentionally has unequal physical axis scales",
        camera_time_difference_s=scene["hardware_frame"]["time_difference_s"],
        calibration_pair="Same original 60-degree recording used for viscosity and effective contact",
        cup_photo_crop_xyxy=group_crop, cup_photo_role="Seed 2 identified by user; existing measurements, not an additional repeat",
        cup_photo_labels=group_labels, cup_photo_measurement_display="Nearest 1 mL; does not imply 1 mL accuracy",
        mapping_record=provenance["dataset_order"],
        known_limitations="See receiver review README and docs/pour_20260908_numerical_replay.md; failed checks retained",
        style_reference="experiments/robotics/x_common_report.py",
        title_font_pt=8.5, supporting_font_pt=7.4, input_sha256=checked,
        generator_sha256=digest(Path(__file__)), installed=install)
    (OUT / "figure_provenance.json").write_text(json.dumps(manifest, indent=2)+"\n")
    if install:
        for suffix in ("pdf", "png"):
            shutil.copyfile(OUT / f"pouring.{suffix}", FIGS / f"pouring.{suffix}")
        shutil.copyfile(OUT / "figure_provenance.json", FIGS / "pouring.provenance.json")
    print(f"Saved {OUT / 'pouring.pdf'}; {measured.size} unchanged readings; installed={install}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--install", action="store_true")
    compose(p.parse_args().install)
