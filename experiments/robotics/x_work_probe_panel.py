"""Aligned, physically calibrated views of the frozen single-pinch pilot."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pyvista as pv

from experiments.robotics.plastic_shaping_figure import COLORS, label_panel, surface
from experiments.robotics.plastic_shaping_study import save_json
from experiments.robotics.x_shaping import CONFIG
from experiments.robotics.x_shaping_franka import Panda

WINDOW = (600, 650)
PARALLEL_SCALE = .055
ACTION_SCALE = .075
ANCHOR_HEIGHT = .40
MOTION_COLOR = "#41434a"


def probe_view(data, *, action=False, robot_xml=None, color="#cfb58c"):
    points = data["most_compressed"] if action else data["x_after_1s"]
    mesh = surface(points, data["vol0"], h=.00125)
    p = pv.Plotter(off_screen=True, window_size=WINDOW)
    p.set_background("white")
    p.add_mesh(mesh, color=color, smooth_shading=True, split_sharp_edges=True,
               ambient=.4, diffuse=.7, specular=.1)
    if action:
        centers = data["compressed_centers"]
        for center in centers:
            p.add_mesh(pv.Cylinder(center=center, direction=(0, 0, 1),
                                  radius=CONFIG["radius"], height=CONFIG["finger_height"],
                                  resolution=80), color="#697680", opacity=.5, smooth_shading=True)
        for hand_mesh, hand_color in Panda(robot_xml).meshes(centers, np.pi / 2):
            p.add_mesh(hand_mesh, color=hand_color, smooth_shading=True, split_sharp_edges=True,
                       ambient=.6, diffuse=.5, specular=.1)

    # The action view has its own zoom to show more of the hand. Released A/B
    # retain identical magnification for the material comparison.
    view_scale = ACTION_SCALE if action else PARALLEL_SCALE
    anchor = np.array([.08, .08, .01 + .025 / 2])
    offset = np.array([.20, -.20, .055] if action else [0., 0., .3])
    viewing = -offset / np.linalg.norm(offset)
    up = np.array([0., 0., 1.] if action else [0., 1., 0.])
    right = np.cross(viewing, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, viewing)
    focal = anchor + up * ((.5 - ANCHOR_HEIGHT) * 2 * view_scale)
    p.camera_position = [focal + offset, focal, up]
    p.enable_parallel_projection()
    p.camera.parallel_scale = view_scale
    p.enable_anti_aliasing("ssaa")

    transform = p.camera.GetCompositeProjectionTransformMatrix(WINDOW[0] / WINDOW[1], -1, 1)
    matrix = np.array([[transform.GetElement(i, j) for j in range(4)] for i in range(4)])

    def pixels(x):
        projected = np.column_stack([x, np.ones(len(x))]) @ matrix.T
        ndc = projected[:, :2] / projected[:, 3, None]
        return (ndc + 1) * np.array(WINDOW) / 2

    projected = pixels(mesh.points)
    assert np.all(projected > 1) and np.all(projected < np.array(WINDOW) - 1), "Specimen cropped"
    calibration = pixels(np.stack([anchor, anchor + right * .01]))
    pixels_per_mm = float(np.linalg.norm(np.diff(calibration, axis=0)) / 10)
    np.testing.assert_allclose(pixels_per_mm, WINDOW[1] / (2 * view_scale) / 1000, rtol=1e-10)
    np.testing.assert_allclose(calibration[0], [WINDOW[0] / 2, WINDOW[1] * ANCHOR_HEIGHT], atol=1e-8)
    arrows = []
    if action:
        # Project the actual horizontal closing axis. Arrow length is a visual
        # cue, not a force magnitude or a measured displacement.
        centers = data["compressed_centers"]
        closing_axis = centers[1] - centers[0]
        closing_axis /= np.linalg.norm(closing_axis)
        projected_centers = pixels(centers)
        axis = projected_centers[1] - projected_centers[0]
        axis /= np.linalg.norm(axis)
        # Attach cues outside the fingers, clear of the specimen and hand.
        # Their heights differ to use the exposed shaft region in each view.
        for center, sign, height in zip(centers, [1, -1], [.047, .044], strict=True):
            cue_anchor = center - sign * closing_axis * (CONFIG["radius"] + .010)
            cue_anchor[2] = height
            tip = pixels(cue_anchor[None])[0]
            tail = tip - sign * axis * 115
            arrows.append(dict(tail_px=tail.tolist(), tip_px=tip.tolist(),
                               anchor_world_m=cue_anchor.tolist(), outward_clearance_m=.010))
    picture = p.screenshot(return_img=True)
    p.close()
    return picture, dict(window_px=WINDOW, parallel_scale_m=view_scale, specimen_color=color,
                        pixels_per_mm=pixels_per_mm, anchor_px=calibration[0].tolist(),
                        camera_position_m=(focal + offset).tolist(), focal_m=focal.tolist(),
                        view_up=up.tolist(), specimen_bounds_px=[projected.min(0).tolist(), projected.max(0).tolist()],
                        motion_arrows=arrows)


def panel_a(ax, folder, images, xml):
    label_panel(ax, "(a) Same force command, different shapes")
    data = {name: np.load(folder / f"inputs/same_force_{name}.npz") for name in "AB"}
    np.testing.assert_array_equal(data["A"]["initial"], data["B"]["initial"])
    ax.text(.17, .745, "Squeeze", ha="center", fontsize=7.4, weight="bold", color=MOTION_COLOR)
    ax.text(.665, .825, "After release (1 s)", ha="center", fontsize=7.4)
    rendered = {}
    for name, action in [("pinch", True), ("A", False), ("B", False)]:
        rendered[name] = probe_view(data["A" if action else name], action=action, robot_xml=xml)
    # Use the released specimens' visible height for all three pictures. Crop
    # excess margins, not material geometry. The action has its own zoom.
    released_bounds = [rendered[name][1]["specimen_bounds_px"] for name in "AB"]
    lower = min(bounds[0][1] for bounds in released_bounds)
    upper = max(bounds[1][1] for bounds in released_bounds)
    crop_height = int(np.ceil(upper - lower)) + 12
    top_released = int(np.floor(WINDOW[1] - upper)) - 6
    views = {}
    for name, cx, action in [("pinch", .17, True), ("A", .50, False), ("B", .83, False)]:
        picture, info = rendered[name]
        top = (int(np.ceil(WINDOW[1] - info["specimen_bounds_px"][0][1])) + 6 - crop_height
               if action else top_released)
        assert 0 <= top and top + crop_height <= WINDOW[1]
        assert top < WINDOW[1] - info["specimen_bounds_px"][1][1]
        assert top + crop_height > WINDOW[1] - info["specimen_bounds_px"][0][1]
        picture = picture[top:top + crop_height]
        info["display_crop_rows"] = [top, top + crop_height]
        info["display_window_px"] = [WINDOW[0], crop_height]
        plt.imsave(images / f"probe_{name}.png", picture)
        ia = ax.inset_axes([cx - .15, .255, .30, .39])
        ia.imshow(picture)
        ia.axis("off")
        if action:
            for arrow in info["motion_arrows"]:
                endpoints = np.array([arrow["tail_px"], arrow["tip_px"]])
                endpoints[:, 1] = WINDOW[1] - endpoints[:, 1] - top
                assert np.all(endpoints > 0) and np.all(endpoints < [WINDOW[0], crop_height])
                mark = FancyArrowPatch(endpoints[0], endpoints[1], arrowstyle="->",
                                      mutation_scale=8, linewidth=1.3, color=MOTION_COLOR,
                                      shrinkA=0, shrinkB=0, zorder=20)
                ia.add_patch(mark)
        if not action:
            # Analytic footprint of the shared initial 60 mm square, projected
            # with the same calibrated top camera as the released specimen.
            scale = info["pixels_per_mm"]
            center = np.array([WINDOW[0] / 2, WINDOW[1] * (1 - ANCHOR_HEIGHT)])
            square = center + np.array([[-30, -30], [30, -30], [30, 30], [-30, 30], [-30, -30]]) * scale
            square[:, 1] -= top
            ia.plot(square[:, 0], square[:, 1], color="#425868", lw=.6, dashes=(3, 2))
            ax.text(cx, .745, name, ha="center", fontsize=8, color=COLORS[name], weight="bold", zorder=20)
        views[name] = info
    ax.text(.5, .115, "1.25 N per finger; 2 s pulse", ha="center", fontsize=7.2, zorder=20)
    ax.text(.5, .025, "Dashed: the common initial outline.", ha="center", fontsize=6.8, color="#46515a")
    save_json(images / "probe_view_calibration.json", dict(views=views,
        inset_size_axes=[.30, .39], inset_bottom_axes=.255,
        scope="Equal visible image heights with labels outside the image band. Released A/B share a physical scale; the action specimen is 0.733 times their screen scale to show more of the gripper. Action has a separate vertical crop and geometric foreshortening. Every specimen is complete."))
