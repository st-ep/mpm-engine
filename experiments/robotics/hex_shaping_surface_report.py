"""Paper figure and geometric checks for surface-optimized work control."""
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

from experiments.robotics.hex_shaping_surface_study import OUT, ROOT, MODELS, check_sources
from experiments.robotics.hex_shaping_surface import surface_error_mm, target_surface
from experiments.robotics.hex_shaping_figure import render, panel_b
from experiments.robotics.hex_shaping_six_report import slices, corner_to_face_ratio
from experiments.robotics.hex_shaping_control_report import section_iou
from experiments.robotics.plastic_shaping_figure import METHOD_COLORS, COLORS, COLUMNS, ROWS, panel_a, label_panel
from experiments.robotics.plastic_shaping_study import chamfer_mm,save_json


def compare_previous(folder, previous):
    """Compare both protocols with the same metric, target, view, and raw slices."""
    target=np.load(folder/"target.npz");mesh=target_surface(target)
    old_target=np.load(previous/"target.npz")
    for key in ["floor","diameter","height","volume"]:
        np.testing.assert_array_equal(target[key],old_target[key])
    images=folder/"previous_comparison_renders";images.mkdir(exist_ok=True)
    rows=[]
    for name in "AB":
        for method in ["nominal","identified","oracle"]:
            row=dict(material=name,method=method)
            for label,source in [("previous",previous),("updated",folder)]:
                path=source/f"baseline_{name}_{method}.npz";data=np.load(path)
                row[label]=dict(surface_error_1s_mm=surface_error_mm(data,mesh),
                    particle_error_1s_mm=chamfer_mm(data["x_after_1s"],target["x"]),
                    slice_iou=section_iou(data,target,"x_after_1s"),
                    path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            rows.append(row)
    fig,axes=plt.subplots(2,3,figsize=(8.,5.6))
    fig.subplots_adjust(left=.035,right=.99,bottom=.07,top=.91,wspace=.01,hspace=.19)
    for row,name in enumerate("AB"):
        record=next(r for r in rows if r["material"]==name and r["method"]=="identified")
        for col,(label,source) in enumerate([("Target",None),("Previous identified",previous),("Updated identified",folder)]):
            ax=axes[row,col]
            data=target if source is None else np.load(source/f"baseline_{name}_identified.npz")
            path=images/f"{name}_{col}.png"
            img=render(data,target,path,color="#b5b6b0" if source is None else METHOD_COLORS["identified"],is_target=source is None)
            ax.imshow(img);ax.axis("off")
            if row==0:ax.set_title(label,fontsize=11)
            if col==0:ax.text(-.015,.5,name,transform=ax.transAxes,fontsize=12,fontweight="bold")
            else:
                ax.contour(np.load(path.with_suffix(".target.npy")),levels=[.5],colors=["black"],linewidths=.7,linestyles=[(0,(2.4,1.8))])
                err=record["previous" if col==1 else "updated"]["surface_error_1s_mm"]
                ax.text(.5,-.015,f"Surface error {err:.3f} mm",ha="center",va="top",transform=ax.transAxes,fontsize=10)
    fig.text(.5,.02,"Both work policies executed in the true material; same metric and observation time.",ha="center",fontsize=9)
    fig.savefig(folder/"identified_before_after.png",dpi=200);fig.savefig(folder/"identified_before_after.pdf");plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(8.,5.))
    angles=np.deg2rad(np.arange(30,391,60));radius=float(target["diameter"])/np.sqrt(3)*1000
    for row,name in enumerate("AB"):
        for label,source,color in [("Previous",previous,"#8c8883"),("Updated",folder,METHOD_COLORS["identified"])]:
            data=np.load(source/f"baseline_{name}_identified.npz")
            for col,xy in enumerate(slices(data,target)):
                hull=ConvexHull(xy);polygon=xy[np.r_[hull.vertices,hull.vertices[0]]]*1000
                axes[row,col].plot(polygon[:,0],polygon[:,1],color=color,lw=1.5,label=label)
        for col in range(3):
            ax=axes[row,col];ax.plot(radius*np.cos(angles),radius*np.sin(angles),"k--",lw=1,label="Target")
            ax.set(aspect="equal",xlim=(-60,60),ylim=(-60,60),title=f"{name}: {[25,50,75][col]}% of target height")
            ax.axis("off");ax.legend(loc="center",frameon=False,fontsize=8)
    fig.tight_layout();fig.savefig(folder/"identified_profiles_before_after.png",dpi=180);plt.close(fig)
    save_json(folder/"before_after.json",dict(previous=str(previous),updated=str(folder),results=rows))
    progress={}
    for model in MODELS:
        old=np.load(previous/f"calibration_{model}.npz")
        evaluations=json.loads((folder/"plans"/model/"evaluations.json").read_text())
        assert len(evaluations)==64
        progress[model]=dict(previous_selected_plan_surface_mm=surface_error_mm(old,mesh),
            new_initial_surface_mm=evaluations[0]["surface_error_1s_mm"],
            new_best_32_surface_mm=min(r["surface_error_1s_mm"] for r in evaluations[:32]),
            new_best_64_surface_mm=min(r["surface_error_1s_mm"] for r in evaluations),
            scope="Model predictions at 1 s; not true-material execution results")
    save_json(folder/"optimization_progress.json",progress)


def paper_preview(folder, images):
    """Prepare the established four-panel layout without replacing the manuscript asset."""
    source=folder/"inputs"
    press_inputs=[*[f"probe_{n}.npz" for n in "AB"],
                  *[f"validation_{m}.npz" for m in ["nominal","true_A","true_B","identified_A","identified_B"]]]
    save_json(folder/"press_panel_inputs.json",{rel:dict(path=str(source/rel),sha256=hashlib.sha256((source/rel).read_bytes()).hexdigest()) for rel in press_inputs})
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8.5,"pdf.fonttype":42,"ps.fonttype":42})
    fig=plt.figure(figsize=(7.2,4.15))
    grid=fig.add_gridspec(2,2,left=.02,right=.985,bottom=.025,top=.99,wspace=.12,hspace=.12)
    ax=fig.add_subplot(grid[0,0]);panel_a(ax,source,images)
    ax.texts[-1].set_text("A: E = 80 kPa, yield = 1 kPa. B: E = 240 kPa, yield = 10 kPa.")
    panel_b(fig.add_subplot(grid[0,1]),source)
    ax=fig.add_subplot(grid[1,0]);label_panel(ax,"(c) Work-limited shaping")
    data=np.load(folder/"baseline_A_identified.npz");target=np.load(folder/"target.npz")
    for i,angle in enumerate([0,60,120]):
        path=images/f"action_{i}.png"
        img=render(data,target,path,color=METHOD_COLORS["identified"],stage=i)
        ax.text(COLUMNS[i],.83,f"{i+1}. {angle}° squeeze",ha="center",va="center",fontsize=7.4)
        ia=ax.inset_axes([COLUMNS[i]-.16,.29,.32,.48]);ia.imshow(img);ia.axis("off")
        ax.text(COLUMNS[i],.22,f"Work limit {data['requested_work_j'][i]:.2f} J",ha="center",va="center",fontsize=7.1)
    ax.text(.5,.075,"Six squeezes; force and travel determine when each stops.",ha="center",va="bottom",fontsize=6.8,color="#46515a")
    ax.text(.5,.005,"First three shown. Material A, identified model.",ha="center",va="bottom",fontsize=6.8,color="#46515a")
    ax=fig.add_subplot(grid[1,1]);label_panel(ax,"(d) Target and executed shapes")
    for x,label in zip(COLUMNS,["Target","Nominal","Identified"],strict=True):
        ax.text(x,.83,label,ha="center",va="center",fontsize=7.4)
    for row,name in enumerate("AB"):
        ax.text(.005,ROWS[row]+.11,name,color=COLORS[name],fontsize=8,fontweight="bold",va="center",ha="left")
        for col,method in enumerate(["target","nominal","identified"]):
            path=images/f"{name}_{method}.png";img=plt.imread(path)
            ia=ax.inset_axes([COLUMNS[col]-.145,ROWS[row],.29,.29]);ia.imshow(img);ia.axis("off")
            if method!="target":
                ia.contour(np.load(path.with_suffix(".target.npy")),levels=[.5],colors=["black"],linewidths=.55,linestyles=[(0,(2.4,1.8))],zorder=5)
                record=json.loads((folder/f"baseline_{name}_{method}.json").read_text())
                ax.text(COLUMNS[col],ROWS[row]-.022,f"{record['surface_error_1s_mm']:.2f} mm",ha="center",va="top",fontsize=7.2,zorder=20)
    ax.text(.5,.025,"Surface error at 1 s. Nominal A reaches the stroke limit.",ha="center",va="bottom",fontsize=6.8,color="#46515a")
    stem=folder/"identification_plastic_shaping_surface"
    fig.savefig(stem.with_suffix(".pdf"),dpi=300,bbox_inches="tight",pad_inches=.02)
    fig.savefig(stem.with_suffix(".png"),dpi=260,bbox_inches="tight",pad_inches=.02);plt.close(fig)


def report(folder):
    check_sources(folder)
    target=np.load(folder/"target.npz");dest=folder/"renders";dest.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(8.,5.6))
    fig.subplots_adjust(left=.035,right=.99,bottom=.08,top=.90,wspace=.01,hspace=.19)
    labels={"target":"Target","nominal":"Nominal model","identified":"Identified model"}
    metrics={}
    for row,name in enumerate("AB"):
        metrics[name]={}
        for col,method in enumerate(["target","nominal","identified"]):
            ax=axes[row,col];path=dest/f"{name}_{method}.png"
            data=target if method=="target" else np.load(folder/f"baseline_{name}_{method}.npz")
            img=render(data,target,path,color="#b5b6b0" if method=="target" else METHOD_COLORS[method],is_target=method=="target")
            ax.imshow(img);ax.axis("off")
            if row==0: ax.set_title(labels[method],fontsize=12)
            if col==0: ax.text(-.015,.5,name,transform=ax.transAxes,fontsize=12,fontweight="bold")
            if method!="target":
                ax.contour(np.load(path.with_suffix(".target.npy")),levels=[.5],colors=["black"],linewidths=.7,linestyles=[(0,(2.4,1.8))])
                err=surface_error_mm(data,target_surface(target))
                ax.text(.5,-.015,f"{err:.3f} mm",ha="center",va="top",transform=ax.transAxes,fontsize=10)
                metrics[name][method]=dict(surface_error_mm=err,particle_error_mm=chamfer_mm(data["x_after_1s"],target["x"]),slice_iou=section_iou(data,target,"x_after_1s"),
                    corner_to_face_ratio=[corner_to_face_ratio(xy) for xy in slices(data,target)])
    fig.text(.5,.02,"Surface-optimized work limits. Surface errors at 1 s after release.",ha="center",fontsize=9)
    fig.savefig(folder/"work_control_comparison.png",dpi=200)
    fig.savefig(folder/"work_control_comparison.pdf");plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(8.,5.))
    a=np.deg2rad(np.arange(30,391,60));r=float(target["diameter"])/np.sqrt(3)*1000
    for row,name in enumerate("AB"):
        for method in ["nominal","identified"]:
            data=np.load(folder/f"baseline_{name}_{method}.npz")
            for col,xy in enumerate(slices(data,target)):
                h=ConvexHull(xy);xy=xy[np.r_[h.vertices,h.vertices[0]]]*1000
                axes[row,col].plot(xy[:,0],xy[:,1],color=METHOD_COLORS[method],lw=1.5,label=labels[method])
        for col in range(3):
            ax=axes[row,col];ax.plot(r*np.cos(a),r*np.sin(a),"k--",lw=1,label="Target")
            ax.set(aspect="equal",xlim=(-65,65),ylim=(-65,65),title=f"{name}: {[25,50,75][col]}% of target height")
            ax.axis("off");ax.legend(loc="center",frameon=False,fontsize=8)
    fig.tight_layout();fig.savefig(folder/"work_control_profiles.png",dpi=180);plt.close(fig)
    # The controller acts on measured work, not observed shape or inferred truth.
    fig,axes=plt.subplots(2,2,figsize=(9,6))
    for row,name in enumerate("AB"):
        for method in ["nominal","identified"]:
            d=np.load(folder/f"baseline_{name}_{method}.npz")
            color=METHOD_COLORS[method]
            axes[row,0].plot(np.arange(1,7),1000*d["gaps"],"o-",color=color,label=labels[method])
            axes[row,1].plot(np.arange(1,7),d["requested_work_j"],"o--",color=color,label=f"{labels[method]}: requested")
            axes[row,1].plot(np.arange(1,7),d["work_j"],"x-",color=color,label=f"{labels[method]}: delivered")
        axes[row,0].axhline(65,color="gray",ls=":",label="Common stroke limit")
        for col in range(2):
            axes[row,col].set(xlabel="Squeeze",title=f"Material {name}")
            axes[row,col].legend(frameon=False,fontsize=8)
        axes[row,0].set_ylabel("Actual final gap (mm)");axes[row,1].set_ylabel("Positive tool work (J)")
    fig.tight_layout();fig.savefig(folder/"control_diagnostics.png",dpi=180);plt.close(fig)
    paper_preview(folder,dest)
    save_json(folder/"figure_metrics.json",metrics)
    source=folder/"report_source";source.mkdir(exist_ok=True)
    shutil.copy2(__file__,source/Path(__file__).name)
    save_json(folder/"figure_provenance.json",dict(renderer=str(Path(__file__).relative_to(ROOT)),
        data="baseline_{A,B}_{nominal,identified}.npz; raw x_after_1s",geometry="Existing common analytic target",
        camera="Same camera and scale for all outcomes, reused from active paper renderer",
        annotations="Dashed target silhouette; no alignment, shape rescaling, or altered simulation geometry"))
    print(json.dumps(metrics,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--out",type=Path,default=OUT)
    parser.add_argument("--dest",type=Path,help="Explicit manuscript asset stem; rebuild the paper after copying")
    parser.add_argument("--previous",type=Path,help="Earlier work-controlled study for a common-metric comparison")
    args=parser.parse_args();folder=args.out.resolve();report(folder)
    if args.previous:
        compare_previous(folder,args.previous.resolve())
    if args.dest:
        for suffix in [".pdf", ".png"]:
            dest=args.dest.with_suffix(suffix);dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2((folder/"identification_plastic_shaping_surface").with_suffix(suffix),dest)


if __name__=="__main__": main()
