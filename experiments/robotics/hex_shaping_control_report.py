"""Render and verify the simple six-squeeze hexagonal-shaping recipe."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

from experiments.robotics.hex_shaping_control import OUT, ROOT
from experiments.robotics.plastic_shaping_figure import surface
from experiments.robotics.plastic_shaping_study import chamfer_mm, save_json


def area(points):
    return .5*abs(np.sum(points[:, 0]*np.roll(points[:, 1], -1)
                         - points[:, 1]*np.roll(points[:, 0], -1)))


def clipped_to_hexagon(points, radius):
    """Clip a convex particle-slice polygon against the six analytic face planes."""
    for angle in np.deg2rad(np.arange(0, 360, 60)):
        normal = np.array([np.cos(angle), np.sin(angle)])
        output = []
        for start, stop in zip(points, np.roll(points, -1, axis=0), strict=True):
            ds, de = start@normal-radius, stop@normal-radius
            if (ds <= 0) != (de <= 0):
                output.append(start+ds/(ds-de)*(stop-start))
            if de <= 0:
                output.append(stop)
        if not output:
            return np.empty((0, 2))
        points = np.asarray(output)
    return points


def section_iou(data, target, key="x"):
    points = data[key].astype(float)-np.array([.15, .15, float(data["floor"])])
    values = []
    d = float(target["diameter"])
    for fraction in [.25, .5, .75]:
        xy = points[abs(points[:, 2]-fraction*float(target["height"])) < .003, :2]
        xy = xy[ConvexHull(xy).vertices]
        intersection = clipped_to_hexagon(xy, d/2)
        intersection_area = area(intersection) if len(intersection) else 0.
        values.append(intersection_area/(area(xy)+np.sqrt(3)/2*d*d-intersection_area))
    return values


def view(data, *, target=False):
    import pyvista as pv

    floor = float(data["floor"])
    plotter = pv.Plotter(off_screen=True, window_size=(640, 560))
    plotter.set_background("white")
    if target:
        angles = np.deg2rad(np.arange(30, 390, 60))
        xy = np.column_stack((np.cos(angles), np.sin(angles)))*float(data["diameter"])/np.sqrt(3)+.15
        bottom = np.column_stack((xy, np.full(6, floor)))
        top = bottom+np.array([0, 0, float(data["height"])])
        faces = [[6, *range(5, -1, -1)], [6, *range(6, 12)]]
        faces += [[4, i, (i+1) % 6, (i+1) % 6+6, i+6] for i in range(6)]
        mesh = pv.PolyData(np.vstack((bottom, top)), np.concatenate(faces))
        plotter.add_mesh(mesh, color="#b5b6b0", ambient=.35, diffuse=.75,
                         show_edges=True, edge_color="#868985", line_width=.6)
    else:
        mesh = surface(data["x_after_1s"], data["vol0"])
        plotter.add_mesh(mesh, color="#cfb58c", smooth_shading=True,
                         ambient=.35, diffuse=.75, specular=.12, specular_power=18)
    plotter.add_mesh(pv.Plane(center=(.15, .15, floor), direction=(0, 0, 1),
                    i_size=.16, j_size=.15), color="#eef0f0", lighting=False)
    focal = np.array([.15, .15, floor+.043])
    plotter.camera.position = focal+np.array([1., -2.8, 4.])*.20
    plotter.camera.focal_point = focal
    plotter.camera.up = (0, 0, 1)
    plotter.enable_parallel_projection()
    plotter.camera.parallel_scale = .074
    plotter.enable_anti_aliasing("ssaa")
    result = plotter.screenshot(return_img=True)
    plotter.close()
    return result


def report(folder):
    target = np.load(folder/"check_A/target_48.npz")
    summary = dict(target_diameter_mm=float(target["diameter"])*1000,
                   target_height_mm=float(target["height"])*1000, materials={})
    data_by_name = {}
    for name, subdir in [("A", "check_A"), ("B_revised", "check_B_E240")]:
        controls = json.loads((folder/subdir/"selected_controls.json").read_text())
        data = np.load(folder/subdir/"rounded.npz")
        data_by_name[name] = data
        checks = json.loads((folder/subdir/"checks.json").read_text())
        assert checks["exact"]["independent_replay_difference_mm"] < .05
        for tag in checks:
            d = np.load(folder/subdir/f"{tag}.npz")
            t = np.load(folder/subdir/f"target_{int(d['n_grid'])}.npz")
            assert int(d["inverted_count"]) == 0
            assert all(np.all(np.isfinite(d[k])) for k in d.files)
            np.testing.assert_allclose(chamfer_mm(d["x"], t["x"]), checks[tag]["error_mm"], rtol=1e-12)
            np.testing.assert_allclose(chamfer_mm(d["x_after_1s"], t["x"]), checks[tag]["error_after_1s_mm"], rtol=1e-12)
            checks[tag]["particle_slice_hull_iou_after_1s"] = section_iou(d, t, "x_after_1s")
        summary["materials"][name] = dict(controls=controls, checks=checks,
            mean_particle_slice_hull_iou_after_1s=float(np.mean(section_iou(data, target, "x_after_1s"))))
    # Check every search objective against its saved rollout and the independent target.
    count = 0
    for group in sorted(folder.glob("*/protocol.json")):
        case = group.parent
        for rel, digest in json.loads((case/"source_sha256.json").read_text()).items():
            assert hashlib.sha256((case/"source_snapshot"/rel).read_bytes()).hexdigest() == digest
        if not (case/"evaluations.json").exists():
            continue
        t = np.load(case/"target.npz")["x"]
        for row in json.loads((case/"evaluations.json").read_text()):
            np.testing.assert_allclose(chamfer_mm(np.load(case/row["file"])["x"], t), row["error_mm"], rtol=1e-12)
            count += 1
    before_path = folder/"manuscript_before.json"
    unchanged = None
    if before_path.exists():
        before = json.loads(before_path.read_text())
        unchanged = all((ROOT/rel).is_file()
                        and hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == digest
                        for rel, digest in before.items())
    summary["search_objectives_verified"] = count
    # Scientific reproduction must also work after the manuscript evolves, or
    # in a fresh output directory without an optional manuscript snapshot.
    summary["manuscript_and_current_figure_unchanged"] = unchanged
    summary["scope"] = "Known-parameter shaping feasibility, with E of B increased to 240 kPa; not an identification comparison"
    save_json(folder/"summary.json", summary)

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.45))
    fig.subplots_adjust(left=.015, right=.995, top=.86, bottom=.21, wspace=.01)
    for ax, name, data in zip(axes, ["Target", "A", "B_revised"],
                             [target, data_by_name["A"], data_by_name["B_revised"]], strict=True):
        ax.imshow(view(data, target=name == "Target")); ax.axis("off")
        if name == "Target":
            title, label = "Hexagonal target", "90 mm across flats; 81.7 mm high"
        else:
            controls = summary["materials"][name]["controls"]
            title = "Material A" if name == "A" else "Material B (revised stiffness)"
            label = (f"E = {controls['law']['E']/1000:.0f} kPa; yield = {controls['law']['yield_stress']/1000:.0f} kPa\n"
                     + "Gaps: "+" / ".join(f"{v:.0f}" for v in controls["gaps_mm"])+" mm")
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        ax.text(.5, -.045, label, transform=ax.transAxes, ha="center", va="top", fontsize=8)
    fig.text(.505, .02, "Squeeze at 0°, 60°, 120°; repeat once. Shown 1 s after the last release.",
             ha="center", fontsize=8)
    fig.savefig(folder/"hexagonal_shapes.png", dpi=200, facecolor="white")
    fig.savefig(folder/"hexagonal_shapes.pdf", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5), constrained_layout=True)
    for ax, name, subdir in zip(axes, ["A", "B (E = 240 kPa)"], ["check_A", "check_B_E240"], strict=True):
        for tag, label, color in [("rounded", "48³, 0.1 ms", "#333333"),
                                  ("half_dt", "48³, 0.05 ms", "#c8733b"),
                                  ("grid64", "64³, 0.1 ms", "#2778a4")]:
            d = np.load(folder/subdir/f"{tag}.npz")
            x = d["x_after_1s"].astype(float)-np.array([.15, .15, float(d["floor"])])
            xy = x[abs(x[:, 2]-.5*float(target["height"])) < .003, :2]*1000
            hull = ConvexHull(xy); xy = xy[np.r_[hull.vertices, hull.vertices[0]]]
            ax.plot(xy[:, 0], xy[:, 1], color=color, label=label)
        angles = np.deg2rad(np.arange(30, 391, 60))
        radius = float(target["diameter"])*1000/np.sqrt(3)
        ax.plot(radius*np.cos(angles), radius*np.sin(angles), "k--", lw=1, label="Target")
        ax.set(title=name, aspect="equal", xlim=(-60, 60), ylim=(-60, 60),
               xlabel="x (mm)", ylabel="y (mm)")
        ax.legend(fontsize=7); ax.grid(alpha=.12)
    fig.savefig(folder/"hexagonal_profiles.png", dpi=180)
    plt.close(fig)
    for rel in ["experiments/robotics/hex_shaping_control_report.py",
                "experiments/robotics/plastic_shaping_figure.py",
                "tests/test_hex_shaping_control.py", "tests/test_hex_shaping_pilot.py",
                "docs/hex_shaping_control.md", "paper/evidence/hex-shaping-control.md"]:
        if not (ROOT/rel).exists():
            continue
        dest = folder/"report_source"/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/rel, dest)
    hashes = {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in folder.rglob("*") if p.is_file() and p.name != "checksums.json"}
    save_json(folder/"checksums.json", hashes)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    report(args.out.resolve())


if __name__ == "__main__":
    main()
