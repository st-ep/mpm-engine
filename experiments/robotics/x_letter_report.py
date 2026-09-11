"""Images of saved material-A letter-shaping results; no simulated data edits."""
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
import pyvista as pv

from experiments.robotics.x_letter_pilot import OUT, check
from experiments.robotics.x_letter_target import mesh as target_mesh
from experiments.robotics.plastic_shaping_figure import surface
from experiments.robotics.plastic_shaping_study import save_json


def render(mesh, *, top=False, mask=False, color="#4a91b4", tools=None, radius=.014):
    p = pv.Plotter(off_screen=True, window_size=(600, 620))
    p.set_background("white")
    p.add_mesh(mesh, color="black" if mask else color, lighting=not mask,
               smooth_shading=True, split_sharp_edges=True, ambient=.4, diffuse=.7, specular=.1)
    if not mask:
        p.add_mesh(pv.Plane(center=(.08,.08,.0099),i_size=.115,j_size=.115),
                   color="#f1f2f2",lighting=False)
    if tools is not None:
        for center in tools:
            p.add_mesh(pv.Cylinder(center=center,direction=(0,0,1),radius=radius,height=.045,resolution=96),
                       color="#9ba4ac",smooth_shading=True,ambient=.35,specular=.35)
    focal=np.array([.08,.08,.026])
    p.camera_position=[focal+np.array([0,0,.3] if top else [0,-.14,.24]),focal,(0,1,0)]
    p.enable_parallel_projection();p.camera.parallel_scale=.061
    if not mask:
        p.enable_anti_aliasing("ssaa")
    image=p.screenshot(return_img=True);p.close()
    return image


def load(folder,path):
    record=json.loads(path.read_text());data=np.load(folder/record["file"])
    return record,surface(data["x_after_1s"],data["vol0"],h=.00125)


def overview(folder):
    paths=sorted((folder/"rollouts").glob("*.json"))
    paths=[p for p in paths if not p.name.endswith((".config.json",".phases.json"))]
    items=[("Balanced target",target_mesh("balanced")),("Slimmer target",target_mesh("slimmer"))]
    for path in paths:
        r,m=load(folder,path)
        if r["config"]["n_grid"]!=64 or r["config"]["dt"]!=.00005:
            continue
        gaps=", ".join(f"{x:g}" for x in r["config"]["gaps_mm"])
        title=f"{path.stem[:5]}: {gaps} mm\nBalanced {r['errors_mm']['balanced']:.2f}; slimmer {r['errors_mm']['slimmer']:.2f} mm"
        items.append((title,m))
    fig,axes=plt.subplots(2,len(items),figsize=(2.7*len(items),5.8))
    for col,(title,m) in enumerate(items):
        for row in range(2):
            axes[row,col].imshow(render(m,top=bool(row),color="#a9afa8" if col<2 else "#4a91b4"))
            axes[row,col].axis("off")
        axes[0,col].set_title(title,fontsize=8.5)
    fig.tight_layout();fig.savefig(folder/"overview.png",dpi=160);plt.close(fig)


def report(folder,path,target,stem,target_shape=None):
    r,actual=load(folder,path)
    if target_shape is None:
        target_shape=target_mesh(target)
    target_mask=render(target_shape,top=True,mask=True)[...,:3].mean(-1)<128
    actual_mask=render(actual,top=True,mask=True)[...,:3].mean(-1)<128
    iou=float(np.count_nonzero(target_mask&actual_mask)/np.count_nonzero(target_mask|actual_mask))
    fig,axes=plt.subplots(1,3,figsize=(9.5,4.))
    pictures=[render(target_shape,color="#b3b6ae"),render(actual),render(actual,top=True)]
    titles=["Target","Material A after withdrawal","Top view: dashed target outline"]
    for ax,img,title in zip(axes,pictures,titles,strict=True):
        ax.imshow(img);ax.axis("off");ax.set_title(title,fontsize=10.5,pad=9)
    axes[2].contour(target_mask,[.5],colors="#253139",linewidths=.9,linestyles=[(0,(3,2))])
    gaps=r["config"]["gaps_mm"]
    size=r["config"].get("initial_size_mm")
    offset=.035 if size else 0.
    fig.text(.5,.10+offset,f"Mean surface error: {r['errors_mm'][target]:.3f} mm     |     Top-view overlap (IoU): {100*iou:.1f}%",ha="center",fontsize=11)
    fig.text(.5,.045+offset,f"{len(gaps)} position-controlled pinches; {2*r['config']['radius_mm']:g} mm fingers; {r['config']['n_grid']}³ grid; 1 s after full withdrawal.",ha="center",fontsize=9,color="#46515a")
    if size:
        fig.text(.5,.025,f"Initial block: {size[0]:g} × {size[1]:g} × {size[2]:.2f} mm; same 90 mL volume and material A parameters.",ha="center",fontsize=9,color="#46515a")
    fig.subplots_adjust(left=.025,right=.975,top=.88,bottom=.18+offset,wspace=.07)
    for ext in ["png","pdf"]:
        fig.savefig(folder/f"{stem}.{ext}",dpi=200,facecolor="white")
    plt.close(fig)
    renderer_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    source=folder/"report_source"/renderer_hash
    source.mkdir(parents=True,exist_ok=True);shutil.copy2(__file__,source/Path(__file__).name)
    save_json(folder/f"{stem}.json",dict(record=r,target=target,footprint_iou=iou,
             metric="IoU of orthographic silhouette masks at common world scale; no registration",
             scope="True-parameter material A simulation, not an identification comparison",
             renderer_sha256=renderer_hash,
             renderer_source=str((source/Path(__file__).name).relative_to(folder)),
             raw_sha256=hashlib.sha256((folder/r["file"]).read_bytes()).hexdigest()))
    print(json.dumps(dict(case=r["file"],target=target,error_mm=r["errors_mm"][target],footprint_iou=iou)),flush=True)


def sequence(folder,path,stem):
    record=json.loads(path.read_text());data=np.load(folder/record["file"])
    gaps=record["config"]["gaps_mm"];radius=record["config"]["radius_mm"]/1000
    fig,axes=plt.subplots(1,len(gaps),figsize=(2.8*len(gaps),3.4),squeeze=False)
    for i,ax in enumerate(axes[0]):
        shape=surface(data[f"stage_{i}_pressed"],data["vol0"],h=.00125)
        ax.imshow(render(shape,tools=data[f"stage_{i}_centers"],radius=radius));ax.axis("off")
        axis="x" if abs(np.sin(np.deg2rad(data["angles_deg"][i])))<.5 else "y"
        ax.set_title(f"Pinch {i+1}: {axis}, gap {gaps[i]:.1f} mm",fontsize=10)
    fig.text(.5,.025,"Saved states at full closure; cylindrical contacts simulated. Full arm dynamics are not modeled.",
             ha="center",fontsize=9,color="#46515a")
    fig.subplots_adjust(left=.015,right=.985,top=.9,bottom=.08,wspace=.035)
    fig.savefig(folder/f"{stem}_actions.png",dpi=180,facecolor="white");plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--out",type=Path,default=OUT)
    p.add_argument("--case",type=Path);p.add_argument("--target",default="balanced",choices=["balanced","slimmer","compact","compact_thin"])
    p.add_argument("--refinement",action="store_true",help="Use the separately frozen refinement targets and controller")
    p.add_argument("--sequence",action="store_true",help="Also render the recorded cylindrical-finger pinch sequence")
    p.add_argument("--stem",default="result");a=p.parse_args()
    if a.refinement:
        from experiments.robotics import x_letter_refine
        x_letter_refine.check(a.out)
    else:
        check(a.out)
    if a.case:
        report(a.out,a.case,a.target,a.stem,x_letter_refine.target_mesh(a.target) if a.refinement else None)
        if a.sequence:
            sequence(a.out,a.case,a.stem)
    else:
        assert not a.refinement, "Select a recorded case for the refinement report"
        overview(a.out)
