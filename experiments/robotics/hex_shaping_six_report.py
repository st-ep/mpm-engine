"""Inspect six-gap outcomes and distinguish six-sided profiles from round ones."""
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

from experiments.robotics.hex_shaping_six_gaps import OUT, ROOT
from experiments.robotics.hex_shaping_figure import render
from experiments.robotics.hex_shaping_control_report import section_iou
from experiments.robotics.plastic_shaping_figure import METHOD_COLORS
from experiments.robotics.plastic_shaping_study import chamfer_mm, save_json


def slices(data, target, key="x_after_1s"):
    x = data[key].astype(float)-np.array([.15, .15, float(data["floor"])])
    return [x[abs(x[:, 2]-q*float(target["height"])) < .003, :2]
            for q in [.25, .5, .75]]


def corner_to_face_ratio(points):
    """Radial size at six target corners divided by size at the six face normals.

    This outline diagnostic is 1 for a centered circle and 2/sqrt(3) for the
    exact centered hexagon. It is not a surface-error metric or selection loss.
    """
    hull = ConvexHull(points)
    angles = np.deg2rad(np.arange(0, 360, 30))
    rays = np.column_stack((np.cos(angles), np.sin(angles)))
    den = hull.equations[:, :2]@rays.T
    radii = np.min(np.divide(-hull.equations[:, 2, None], den,
                  out=np.full(den.shape, np.inf), where=den > 1e-12), axis=0)
    return float(radii[1::2].mean()/radii[::2].mean())


def release_profiles(folder, name, method, target):
    """Inspect raw particle slices under the last squeeze and after release."""
    data = np.load(folder/f"baseline_{name}_{method}.npz")
    keys = ["stage_5_pressed", "x", "x_after_1s"]
    titles = ["Last squeeze", "0.30 s after release", "1.00 s after release"]
    colors = ["#33658a", "#d28a35", "#638c4d"]
    angles = np.deg2rad(np.arange(30, 391, 60))
    radius = float(target["diameter"])/np.sqrt(3)*1000
    fig, axes = plt.subplots(1, 3, figsize=(8., 2.9))
    for col, (key, title) in enumerate(zip(keys, titles, strict=True)):
        ax = axes[col]
        for fraction, color, xy in zip([25, 50, 75], colors, slices(data, target, key), strict=True):
            h = ConvexHull(xy)
            xy = xy[np.r_[h.vertices, h.vertices[0]]]*1000
            ax.plot(xy[:, 0], xy[:, 1], color=color, lw=1.2, label=f"{fraction}% height")
        ax.plot(radius*np.cos(angles), radius*np.sin(angles), "k--", lw=1, label="Target")
        ax.set(aspect="equal", xlim=(-60, 60), ylim=(-60, 60), title=title)
        ax.axis("off")
    axes[1].legend(loc="center", frameon=False, fontsize=8)
    fig.suptitle(f"Material {name}, {method} plan: cross-sections at fixed target heights", fontsize=11)
    fig.tight_layout()
    fig.savefig(folder/f"release_{name}_{method}.png", dpi=180)
    plt.close(fig)


def geometry_checks(folder):
    """Audit selected numerical replays and all completed true-model trials.

    These post-search diagnostics never change the selected action or objective.
    """
    executions = []
    for tag in ["baseline", "half_dt", "grid64"]:
        for path in sorted(folder.glob(f"{tag}_?_*.json")):
            row = json.loads(path.read_text())
            data = np.load(path.with_suffix(".npz"))
            target = np.load(folder/f"target_{tag}.npz")
            row["slice_hull_iou"] = section_iou(data, target, "x_after_1s")
            row["corner_to_face_ratio"] = [corner_to_face_ratio(xy) for xy in slices(data, target)]
            row["rms_speed_1s_mm_s"] = float(np.sqrt((data["v_after_1s"]**2).sum(axis=1).mean())*1000)
            executions.append(row)
    trials = {}
    target = np.load(folder/"target.npz")
    for selected in sorted(folder.glob("plans/true_*/selected.json")):
        rows = json.loads((selected.parent/"evaluations.json").read_text())
        for row in rows:
            data = np.load(folder/row["file"])
            row["slice_hull_iou"] = section_iou(data, target, "x")
            row["corner_to_face_ratio"] = [corner_to_face_ratio(xy) for xy in slices(data, target, "x")]
        trials[selected.parent.name] = rows
    save_json(folder/"geometry_checks.json", dict(executions=executions, true_model_trials=trials,
        interpretation="Post-search diagnostics, not action-selection criteria. Trials at 0.30 s; independent executions at 1.00 s. A fixed observation time does not imply mechanical equilibrium."))


def report(folder, materials, methods):
    target = np.load(folder/"target.npz")
    images = folder/"renders"/hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    images.mkdir(parents=True, exist_ok=True)
    colors = dict(METHOD_COLORS, target="#b5b6b0")
    titles = dict(nominal="Nominal", identified="Identified", oracle="True parameters", target="Target")
    columns = ["target", *methods]
    fig, axes = plt.subplots(len(materials), len(columns),
        figsize=(2.25*len(columns), 2.6*len(materials)), squeeze=False)
    fig.subplots_adjust(left=.035, right=.99, bottom=.08, top=.88, wspace=.01, hspace=.18)
    metrics = {}
    for row, name in enumerate(materials):
        metrics[name] = {}
        for col, method in enumerate(columns):
            ax = axes[row, col]
            path = images/f"{name}_{method}.png"
            data = target if method == "target" else np.load(folder/f"baseline_{name}_{method}.npz")
            img = render(data, target, path, color=colors[method], is_target=method == "target")
            ax.imshow(img); ax.axis("off")
            if row == 0: ax.set_title(titles[method], fontsize=11)
            if col == 0: ax.text(-.02, .5, name, transform=ax.transAxes, fontsize=12, fontweight="bold")
            if method != "target":
                ax.contour(np.load(path.with_suffix(".target.npy")), levels=[.5], colors=["black"],
                           linewidths=.7, linestyles=[(0,(2.4,1.8))])
                metrics[name][method] = dict(error_mm=chamfer_mm(data["x_after_1s"], target["x"]),
                    slice_hull_iou=section_iou(data,target,"x_after_1s"),
                    corner_to_face_ratio=[corner_to_face_ratio(p) for p in slices(data,target)])
                ax.text(.5, -.01, f"{metrics[name][method]['error_mm']:.3f} mm", transform=ax.transAxes,
                        ha="center", va="top", fontsize=10)
    fig.text(.5,.01,"Original materials, six independent gaps. Shown 1 s after release.",ha="center",fontsize=9)
    stem="comparison_"+"".join(materials)+"_"+"_".join(methods)
    fig.savefig(folder/f"{stem}.png",dpi=200); fig.savefig(folder/f"{stem}.pdf");plt.close(fig)

    fig, axes = plt.subplots(len(materials),3,figsize=(8.,2.6*len(materials)),squeeze=False)
    angles=np.deg2rad(np.arange(30,391,60)); r=float(target["diameter"])/np.sqrt(3)*1000
    for row,name in enumerate(materials):
        for method in methods:
            d=np.load(folder/f"baseline_{name}_{method}.npz")
            for col,xy in enumerate(slices(d,target)):
                h=ConvexHull(xy); xy=xy[np.r_[h.vertices,h.vertices[0]]]*1000
                axes[row,col].plot(xy[:,0],xy[:,1],color=colors[method],lw=1.5,label=titles[method])
        for col in range(3):
            ax=axes[row,col];ax.plot(r*np.cos(angles),r*np.sin(angles),"k--",lw=1,label="Target")
            ax.set(aspect="equal",xlim=(-60,60),ylim=(-60,60),title=f"{name}: {[25,50,75][col]}% of target height")
            ax.axis("off");ax.legend(loc="center",frameon=False,fontsize=8)
    fig.tight_layout();fig.savefig(folder/f"{stem}_profiles.png",dpi=180);plt.close(fig)
    save_json(folder/f"{stem}_metrics.json",dict(metrics=metrics,
        ideal_corner_to_face_ratio=2/np.sqrt(3),circle_corner_to_face_ratio=1.,
        interpretation="Convex slice hull diagnostics at fixed physical heights; not full 3D accuracy or search objectives"))
    for name in materials:
        for method in methods:
            release_profiles(folder, name, method, target)
    geometry_checks(folder)
    for rel in ["experiments/robotics/hex_shaping_six_report.py", "experiments/robotics/hex_shaping_figure.py",
                "experiments/robotics/hex_shaping_control_report.py", "experiments/robotics/plastic_shaping_figure.py"]:
        dest=folder/"report_source"/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
    print(json.dumps(metrics,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=OUT)
    parser.add_argument("--materials",nargs="+",choices=["A","B"],default=["A","B"])
    parser.add_argument("--methods",nargs="+",choices=["nominal","identified","oracle"],default=["nominal","identified"])
    args=parser.parse_args();report(args.out.resolve(),args.materials,args.methods)


if __name__=="__main__":
    main()
