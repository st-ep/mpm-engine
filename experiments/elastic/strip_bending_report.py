"""Figure preview, video and concise report from saved bending trajectories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image, ImageDraw

from experiments.elastic.block_drop_report import COLORS, INK, font
from experiments.elastic.block_drop_study import save, sha
from experiments.elastic.strip_bending_study import pusher_position
from experiments.robotics.plastic_shaping_figure import surface


def read(path):
    return json.loads(Path(path).read_text())


def signals(root,name):
    return np.genfromtxt(root/name/"signals.csv",delimiter=",",names=True)


def summarize(root):
    p=read(root/"protocol.json")
    result=dict(protocol=p,materials={})
    for label in COLORS:
        # The report requires complete main, prediction and resolution-check runs.
        checks={}
        for stage in ["truth", "prediction", "fine"]:
            case=root/f"{stage}_{label}"
            completed=read(case/"completion.json")
            if not completed["all_frames_completed"] or completed["inverted_count"]:
                raise RuntimeError(f"Incomplete or inverted simulation: {case}")
            data=signals(root,f"{stage}_{label}")
            force=np.column_stack([data[k] for k in ["reaction_x","reaction_y","reaction_z"]])
            maximum=float(np.linalg.norm(force[data["time"]>=p["withdraw_end"]],axis=1).max())
            if maximum>1e-8:
                raise RuntimeError(f"Pusher recontact during recovery in {case}: {maximum} N")
            checks[stage]=dict(max_recovery_tool_force_n=maximum,
                               min_J=completed["min_J"],
                               momentum_max_error_kg_m_s=completed["cumulative_momentum_max_error_kg_m_s"])
        fit=read(root/f"truth_{label}/identification.json")
        predicted_E=read(root/f"prediction_{label}/config.json")["E_pa"]
        if predicted_E!=fit["E_pa"]:
            raise RuntimeError("Prediction did not use the frozen fitted stiffness")
        truth=np.load(root/f"truth_{label}/x.npy",mmap_mode="r")
        pred=np.load(root/f"prediction_{label}/x.npy",mmap_mode="r")
        t=np.load(root/f"truth_{label}/time.npy")
        if not np.array_equal(truth[0],pred[0]):
            raise RuntimeError("Prediction must start from the original undeformed specimen")
        if not np.array_equal(t,np.load(root/f"prediction_{label}/time.npy")):
            raise RuntimeError("Prediction and truth must have identical observation times")
        mask=t>=p["withdraw_end"]
        rms=np.array([np.sqrt(np.mean(np.sum((a.astype(float)-b)**2,axis=1)))*1000
                      for a,b in zip(truth,pred,strict=True)])
        d=signals(root,f"truth_{label}")
        dp=signals(root,f"prediction_{label}")
        tip=np.stack([d[k]-dp[k] for k in ["tip_x","tip_y","tip_z"]],axis=1)*1000
        fit_mask=(t>=p["settle_end"])&(t<=p["bend_end"])
        row=dict(true_E_pa=p["E_pa"][label],identified_E_pa=fit["E_pa"],
                 signed_E_error_percent=100*(fit["E_pa"]/p["E_pa"][label]-1),
                 recovery_particle_rmse_mm=float(np.sqrt(np.mean(rms[mask]**2))),
                 recovery_max_frame_rmse_mm=float(rms[mask].max()),
                 recovery_tip_rmse_mm=float(np.sqrt(np.mean(np.sum(tip[mask]**2,axis=1)))),
                 peak_loading_force_n=float(d["reaction_x"][fit_mask].max()),
                 final_loading_force_n=float(d["reaction_x"][np.argmin(abs(t-p["bend_end"]))]),
                 max_recovery_tool_force_n=float(np.linalg.norm(np.column_stack(
                     [d[k][mask] for k in ["reaction_x","reaction_y","reaction_z"]]),axis=1).max()),
                 fit=fit,checks=checks)
        fine_dir=root/f"fine_{label}"
        if (fine_dir/"identification.json").exists():
            fine=signals(root,f"fine_{label}")
            row["fine_identification"]=read(fine_dir/"identification.json")
            row["grid_tip_rmse_mm"]=float(np.sqrt(np.mean(sum(
                (d[k][mask]-fine[k][mask])**2 for k in ["tip_x","tip_y","tip_z"]))))*1000
            row["fine_peak_loading_force_n"]=float(fine["reaction_x"][fit_mask].max())
        np.savetxt(root/f"recovery_error_{label}.csv",np.column_stack([t,rms]),
                   delimiter=",",header="time,particle_rmse_mm",comments="")
        result["materials"][label]=row
    save(root/"report.json",result)
    return result


def scene(pl,x,vol,pose,p,color,reference=None):
    import pyvista as pv
    pl.clear(); pl.enable_lightkit(); pl.set_background("white")
    mesh=surface(x,vol,h=.0008)
    mesh.compute_normals(auto_orient_normals=True,consistent_normals=True,inplace=True)
    pl.add_mesh(mesh,color=color,smooth_shading=True,ambient=.28,diffuse=.8,
                specular=.18,specular_power=25)
    # Fixture illustrates the ideal bonded clamp, not a simulated Franka mechanism.
    for bounds in [(.101,.139,.099,.111,.165,.193),(.101,.139,.129,.141,.165,.193),
                   (.101,.139,.099,.141,.193,.204)]:
        block=pv.Box(bounds=bounds).triangulate().subdivide(1)
        pl.add_mesh(block,color="#aab4bc",ambient=.25,specular=.5,specular_power=30)
    pl.add_mesh(pv.Cylinder(center=(.12,.12,.212),direction=(0,0,1),radius=.009,
                            height=.018,resolution=48),color="#364650",specular=.45)
    for y in [.103,.137]:
        pl.add_mesh(pv.Cylinder(center=(.110,y,.182),direction=(0,1,0),radius=.003,
                                height=.004,resolution=24),color="#4e5d67")
    tool=pv.Cylinder(center=pose,direction=(0,1,0),radius=p["pusher_radius"],
                     height=p["pusher_length"],resolution=64)
    pl.add_mesh(tool,color="#bac3c7",smooth_shading=True,ambient=.2,specular=.6,specular_power=40)
    # The dark collar and shaft are an illustration of the prescribed actuator.
    pl.add_mesh(pv.Cylinder(center=pose+np.array([.022,0,0]),direction=(1,0,0),
                            radius=.004,height=.032,resolution=36),color="#697a84",specular=.4)
    if reference is not None:
        # Initial centerline gives a neutral displacement reference.
        line=pv.Line((p["center"][0],.12,.082),(p["center"][0],.12,.161),resolution=40)
        pl.add_mesh(line,color="#9aa5ab",line_width=1.3,opacity=.6)
    focal=np.array([.137,.120,.145])
    pl.camera.position=focal+np.array([.12,-.50,.085])
    pl.camera.focal_point=focal; pl.camera.up=(0,0,1)
    pl.enable_parallel_projection(); pl.camera.parallel_scale=.080
    return pl.screenshot(return_img=True)


def phase(t,p):
    if t<=p["settle_end"]: return "Hold the upper end"
    if t<=p["bend_end"]: return "Bend with a prescribed pusher motion"
    if t<=p["hold_end"]: return "Hold the bend"
    if t<=p["withdraw_end"]: return "Withdraw the pusher"
    return "Free recovery · upper end remains held"


def render(root,r):
    import imageio.v2 as imageio
    import pyvista as pv
    p=r["protocol"]; media=root/"media"; media.mkdir(exist_ok=True)
    X={k:np.load(root/f"truth_{k}/x.npy",mmap_mode="r") for k in COLORS}
    S={k:signals(root,f"truth_{k}") for k in COLORS}
    P={k:signals(root,f"prediction_{k}") for k in COLORS}
    vol=np.load(root/"truth_A/vol0.npy"); times=np.load(root/"truth_A/time.npy")
    pl=pv.Plotter(off_screen=True,window_size=(600,580),lighting="light kit")
    pl.enable_anti_aliasing("ssaa")
    stills={}
    selected=[round(p["bend_end"]/p["tick"]),round(1.4/p["tick"])]
    def frame(i):
        t=times[i]
        canvas=Image.new("RGB",(1200,800),"white"); draw=ImageDraw.Draw(canvas)
        draw.text((28,15),"Elastic identification from bending",font=font(27,True),fill=INK)
        draw.text((992,21),f"{t:.2f} s",font=font(24),fill=INK)
        for col,k in enumerate(COLORS):
            im=Image.fromarray(scene(pl,X[k][i],vol,pusher_position(t,p),p,COLORS[k],reference=True))
            canvas.paste(im,(col*600,105))
            draw.text((col*600+28,67),f"{k}   {p['E_pa'][k]/1000:.0f} kPa",font=font(24,True),fill=COLORS[k])
            draw.text((col*600+28,683),f"Pusher force  {S[k]['reaction_x'][i]:.3f} N",font=font(21),fill=INK)
            if i in selected: stills[k,i]=im
        draw.line((600,110,600,674),fill="#dfe4e7",width=2)
        draw.text((28,727),phase(t,p),font=font(23,True),fill=INK)
        draw.text((28,769),"4× slow motion · same motion and scale · simulated strip and contacts; fixture illustration",
                  font=font(15),fill="#5e6c75")
        return np.asarray(canvas)
    with imageio.get_writer(media/"bend_and_release.mp4",fps=25,codec="libx264",quality=8,
                            macro_block_size=2,pixelformat="yuv420p") as writer:
        first=frame(0)
        for _ in range(13): writer.append_data(first)
        for i in range(0,len(times),5): writer.append_data(frame(i))
        last=frame(len(times)-1)
        for _ in range(13): writer.append_data(last)
    for i in selected:
        if ("A",i) not in stills: frame(i)
    pl.close()

    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                         "axes.spines.top":False,"axes.spines.right":False,
                         "axes.labelcolor":INK,"text.color":INK,"axes.titleweight":"bold"})
    fig=plt.figure(figsize=(14,9),facecolor="white")
    fig.text(.035,.962,"Identify stiffness from bending, predict the recovery",fontsize=20,weight="bold")
    fig.text(.035,.923,"Two elastic strips · identical pusher motion · identification uses bending data only",fontsize=12,color="#576772")
    for row,(fi,title) in enumerate(zip(selected,["(a) Same prescribed bend","(c) Free recovery at 1.40 s"],strict=True)):
        top=.873-row*.403
        fig.text(.035,top,title,fontsize=13,weight="bold")
        for col,k in enumerate(COLORS):
            ax=fig.add_axes([.015+col*.245,top-.345,.255,.334])
            ax.imshow(stills[k,fi]); ax.axis("off")
            ax.text(.12,.92,k,transform=ax.transAxes,color=COLORS[k],fontsize=15,weight="bold")
    force=fig.add_axes([.60,.616,.345,.235])
    for k in COLORS:
        d=S[k]; use=(d["time"]>=p["settle_end"])&(d["time"]<=p["bend_end"])
        force.plot((p["pusher_initial"][0]-d["tool_x"][use])*1000,d["reaction_x"][use],
                    color=COLORS[k],label=k,lw=1.8)
    force.set(title="(b) Measured loading force",xlabel="Pusher travel (mm)",ylabel="Force (N)")
    force.legend(frameon=False)
    for i,k in enumerate(COLORS):
        row=r["materials"][k]
        fig.text(.60,.545-i*.026,f"{k}:  E = {row['identified_E_pa']/1000:.3f} kPa "
                 f"({row['signed_E_error_percent']:+.3f}%)",fontsize=11,color=COLORS[k],weight="bold")
    recovery=fig.add_axes([.60,.204,.345,.222])
    for k in COLORS:
        use=S[k]["time"]>=p["withdraw_end"]
        for d,identified in [(S[k],False),(P[k],True)]:
            recovery.plot(d["time"][use]-p["withdraw_end"],
                           (d["tip_x"][use]-p["center"][0])*1000,
                           ls="none" if identified else "-",color=COLORS[k],lw=1.6,
                           marker="o" if identified else None,markevery=22,
                           ms=3.5,mfc="white",mew=1.0)
    fig.text(.60,.470,"(d) Recovery excluded from fitting",fontsize=13,weight="bold")
    recovery.set(xlabel="Time after withdrawal (s)",
                 ylabel="Tip displacement (mm)")
    recovery.legend(handles=[Line2D([],[],color=INK,label="True",lw=1.6),
                             Line2D([],[],color=INK,label="Identified",ls="none",
                                    marker="o",ms=4,mfc="white",mew=1)],
                     frameon=False,ncol=2,loc="lower left",bbox_to_anchor=(0,1.01),
                     borderaxespad=0,fontsize=9)
    fig.text(.60,.117,"Recovery 3D particle RMSE",fontsize=10,color="#576772")
    fig.text(.60,.087,"  ·  ".join(f"{k}: {r['materials'][k]['recovery_particle_rmse_mm']:.3f} mm" for k in COLORS),
             fontsize=12,weight="bold")
    fig.text(.035,.033,"Exact simulated particle states and force. Only E is fitted; ν = 0.45 and density = 1000 kg/m³ are known.",
             fontsize=10,color="#576772")
    fig.savefig(media/"figure_preview.png",dpi=160)
    fig.savefig(media/"figure_preview.pdf")
    plt.close(fig)
    save(media/"provenance.json",dict(renderer_sha256=sha(Path(__file__)),
         report_sha256=sha(root/"report.json"),playback_slowdown=4,
         displayed_times_s=[float(times[i]) for i in selected],
         surface_voxel_m=.0008,common_camera=True,physical_geometry_not_rescaled=True))


def report(root,r):
    rows=[]
    for k,v in r["materials"].items():
        rows.append(f"| {k} | {v['true_E_pa']/1000:.1f} | {v['identified_E_pa']/1000:.3f} | "
                    f"{v['signed_E_error_percent']:+.3f}% | {v['recovery_particle_rmse_mm']:.4f} |")
    text="""# Bending identification and unfitted recovery

Two identical 12 × 20 × 100 mm strips are held vertically at their upper end.
The upper 15 mm lie in an ideal fixed grip. A 20 mm diameter cylindrical pusher
moves sideways by 24 mm, holds briefly, and withdraws. The grip stays fixed
throughout recovery. Both materials receive exactly the same commanded motion.
These are the elastic A/B materials from the drop pilot, not the plastic-shaping materials.

**What is needed:** bending measurements identify stiffness E from deformation
and pusher force. Release is not required for this fit. It supplies an unfitted
prediction interval after the pusher withdraws; its measured contact force is zero.

| Material | True E (kPa) | Identified E (kPa) | Signed error | Recovery particle RMSE (mm) |
|---|---:|---:|---:|---:|
"""+"\n".join(rows)+"""

The model prediction starts from the original undeformed specimen and replays
the full frozen tool trajectory. Recovery is a continuation of the identification
episode excluded from fitting, not an independent experiment or geometry transfer.
RMSE compares all corresponding 3D particles from 1.17 to 2.00 s, without
alignment, rescaling, or true-state initialization at release.

![Figure preview](media/figure_preview.png)

[Synchronized bending and release video, 4× slow motion](media/bend_and_release.mp4)

## Identification and assumptions

The fit uses noise-free particle x, v and full F, reference volumes, known density,
and the measured net pusher reaction. Geometry, Poisson ratio 0.45, law family,
grip and tool motion are known. Only E is estimated. The material is ideal
fixed-corotated elasticity, with no plasticity or modeled viscosity.

Three nodal bending virtual fields are interpolated with the same quadratic
B-splines used by MPM. Each is zero through the grip and equal to the horizontal
unit vector throughout the pusher contact. Consequently the unknown grip force
vanishes and the pusher contributes its net force without a pressure-distribution
assumption. Tests check these traces, the spatial derivatives and equivalence
to full 3D MPM nodal interpolation. Time integration removes acceleration from
the data. The scalar least-squares fit uses 0.32–0.88 s only. Grip forces are
recorded solely for a global momentum audit and never enter identification.

Recorded pusher forces are interval averages; adjacent intervals are averaged to
center the force on each observed state. Every interval used is within loading.
True stiffness, stress and hold/withdrawal/recovery measurements are excluded.
Dropping the force term is retained as a diagnostic and produces nonphysical
negative stiffness for this probe; we do not use that incorrect fit.

## Numerical checks and limits

The main simulation uses a 96³ grid over 0.24 m, 20 µs time steps, and 2 ms
observations. A 128³ replay checks sensitivity. Completion files contain finite
state, Jacobian, mass, and momentum-balance checks. Exact physical dimensions are
preserved when resolution changes. Explicit grid damping is disabled, but MPM
has numerical dissipation. The matched-resolution prediction errors do not
establish physical accuracy or convergence.

The video shows raw-motion surface reconstructions at 4× slow motion. Colors
identify the materials. The clamp fixture and pusher shaft are illustrations of
the ideal boundary conditions, not a simulated robot-arm mechanism. No hardware,
camera reconstruction, insertion planning, or geometry transfer is demonstrated.
No deformation is exaggerated and no trajectory is aligned or rescaled.

## Development record

The first run in ../strip_bending_20260912 is retained. Its pusher parking
position allowed A to strike it again. This run moves the parked pusher farther
away for both materials. The material parameters, loading motion and estimator
settings are unchanged. Recovery must have zero pusher contact in every case.

## Reproduce

Use a fresh directory. From the repository root:

```bash
.venv/bin/python -m experiments.elastic.strip_bending_study init --out out/new_bending
```

For each material A and B, run the following stages in order. Assign cuda:0 or
cuda:1 through --device. Each stage uses the protocol saved at initialization.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.elastic.strip_bending_study truth --out out/new_bending --material A --device cuda:0
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.elastic.strip_bending_study prediction --out out/new_bending --material A --device cuda:0
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.elastic.strip_bending_study fine --out out/new_bending --material A --device cuda:0
```

Repeat those three commands with material B. Then render:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.elastic.strip_bending_report --out out/new_bending
```

Raw states, forces, fitted equations, protocol and source snapshots are retained.
GPU atomic operations can introduce small floating-point differences between
reruns; bitwise identity is not promised.
"""
    for k,v in r["materials"].items():
        if "fine_identification" in v:
            text+=f"\nGrid check {k}: E = {v['fine_identification']['E_pa']/1000:.3f} kPa; "
            text+=f"recovery tip difference between grids = {v['grid_tip_rmse_mm']:.3f} mm RMS.\n"
            text+=f"Loading force peaks at grid 96/128: {v['peak_loading_force_n']:.3f}/"
            text+=f"{v['fine_peak_loading_force_n']:.3f} N.\n"
    (root/"REPORT.md").write_text(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    r=summarize(args.out); report(args.out,r); render(args.out,r)


if __name__=="__main__":
    main()
