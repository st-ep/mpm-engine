"""A-only letter refinement: explicit pinch order and repeated position commands.

The MPM solver, contact primitives, material and particle sampling are unchanged.
This extends the recorded motion loop to two, four or six pinches. Targets are
declared before this refinement batch and scored separately at equal volume.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pyvista as pv
from scipy.optimize import minimize

from experiments.robotics import x_shaping as core
from experiments.robotics.x_letter_pilot import LAW, ROOT, scene
from experiments.robotics.x_letter_target import outline, TARGETS as FIRST_TARGETS
from experiments.robotics.hex_shaping_surface import mesh_distance_mm
from experiments.robotics.plastic_shaping_figure import surface
from experiments.robotics.plastic_shaping_study import save_json, save_npz

OUT=ROOT/"out/x_letter_A_refine_20260909"
TARGETS={"balanced":FIRST_TARGETS["balanced"],
         "compact":dict(width=74.,height=80.,end_width=19.,cx=11.,cy=7.,inner_radius=12.,outer_radius=2.),
         "compact_thin":dict(width=72.,height=78.,end_width=18.,cx=10.,cy=6.,inner_radius=10.,outer_radius=2.)}
MESHES={}


def target_mesh(name):
    if name not in MESHES:
        xy,area=outline(TARGETS[name]);xy+=.08
        p=np.column_stack([xy,np.full(len(xy),.01)])
        m=pv.PolyData(p,np.r_[len(p),np.arange(len(p))]).extrude((0,0,.00009/area),capping=True).triangulate().clean()
        m=m.compute_normals(auto_orient_normals=True,consistent_normals=True).subdivide(3,subfilter="linear")
        assert m.n_open_edges==0
        np.testing.assert_allclose(m.volume,.00009,rtol=2e-6)
        MESHES[name]=m
    return MESHES[name]


def execute(gaps,first_angle,grid,dt,device):
    cfg=core.CONFIG;tick=cfg["tick"];substeps=round(tick/dt)
    assert abs(substeps*dt-tick)<1e-12
    gaps=np.asarray(gaps,float)/1000
    assert 1<=len(gaps)<=8 and np.all((gaps>=.016)&(gaps<=.060))
    angles=np.deg2rad(first_angle+np.tile([0.,90.],(len(gaps)+1)//2)[:len(gaps)])
    x,vol=core.specimen(grid)
    with contextlib.redirect_stdout(io.StringIO()):
        solver=core.Solver(core.GridConfig(n_grid=grid,grid_lim=cfg["domain"]),device=device,inversion_policy="raise").load_particles(x,vol)
        solver.set_material(core.vonmises(**LAW,density=cfg["density"]))
    solver.add_plane(point=(0,0,cfg["floor"]),normal=(0,0,1),surface="separable",friction=cfg["floor_friction"])
    sdf=core.cylinder_sdf(cfg["radius"],cfg["finger_height"])
    pose=core.centers(angles[0],cfg["opening"],cfg["hover_bottom"])
    handles=[solver.add_sdf_collider(sdf,center=c,band=cfg["band"],surface="separable",friction=cfg["tool_friction"]) for c in pose]
    data=dict(initial=x,vol0=vol,floor=np.array(cfg["floor"]),n_grid=np.array(grid),dt=np.array(dt),gaps_mm=gaps*1000,angles_deg=np.rad2deg(angles))
    elapsed=0.;frames=[];ids=[];forces=[];times=[];phases=[]

    def move(end,duration,name,angle_start=None,angle_end=None):
        nonlocal pose,elapsed
        start=pose.copy();count=max(1,int(np.ceil(duration/tick)))
        phases.append(dict(name=name,start_s=elapsed,duration_s=count*tick,start_centers=start.tolist(),end_centers=end.tolist()))
        for j in range(count):
            nxt=start+(end-start)*(j+1)/count if angle_start is None else core.centers(angle_start+(angle_end-angle_start)*(j+1)/count,cfg["opening"],cfg["hover_bottom"])
            velocity=(nxt-pose)/tick
            for handle,c,v in zip(handles,pose,velocity,strict=True):
                solver.set_sdf_pose(handle,center=c,velocity=v);solver.reset_sdf_force(handle)
            solver.step(dt,substeps=substeps)
            forces.append(np.stack([solver.sdf_wrench(h,tick)["force"] for h in handles]))
            frames.append(nxt.copy());times.append(elapsed+tick);ids.append(len(phases)-1)
            pose=nxt;elapsed+=tick

    for i,(gap,angle) in enumerate(zip(gaps,angles,strict=True)):
        if i:
            move(core.centers(angle,cfg["opening"],cfg["hover_bottom"]),abs(angle-angles[i-1])/cfg["rotation_speed"],f"{i}:rotate",angles[i-1],angle)
        move(core.centers(angle,cfg["opening"],cfg["finger_bottom"]),(cfg["hover_bottom"]-cfg["finger_bottom"])/cfg["vertical_speed"],f"{i}:approach")
        move(core.centers(angle,gap,cfg["finger_bottom"]),(cfg["opening"]-gap)/(2*cfg["finger_speed"]),f"{i}:close")
        data[f"stage_{i}_pressed"]=solver.x().copy();data[f"stage_{i}_centers"]=pose.copy()
        move(core.centers(angle,cfg["opening"],cfg["finger_bottom"]),(cfg["opening"]-gap)/(2*cfg["finger_speed"]),f"{i}:open")
        move(core.centers(angle,cfg["opening"],cfg["hover_bottom"]),(cfg["hover_bottom"]-cfg["finger_bottom"])/cfg["vertical_speed"],f"{i}:withdraw")
        data[f"stage_{i}_released"]=solver.x().copy()
    data["x_after_withdrawal"]=solver.x().copy()
    move(pose.copy(),cfg["release_s"],"final:wait")
    data.update(x_after_1s=solver.x().copy(),v_after_1s=solver.v().copy(),inverted_count=np.array(solver.inverted_count()),time=np.array(times),tool_centers=np.array(frames),reaction_force=np.array(forces),phase_id=np.array(ids))
    assert int(data["inverted_count"])==0 and all(np.all(np.isfinite(v)) for v in data.values())
    return data,phases


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(folder):
    folder.mkdir(parents=True,exist_ok=True);assert not (folder/"protocol.json").exists()
    save_json(folder/"protocol.json",dict(targets=TARGETS,law=LAW,scene=dict(core.CONFIG,rotation_speed=np.pi/2),scope="True-parameter A-only adaptive feasibility; no identification comparison"))
    previous=ROOT/"out/x_letter_A_20260909"
    hashes=json.loads((previous/"source_sha256.json").read_text())
    hashes[str(Path(__file__).relative_to(ROOT))]=digest(__file__)
    for rel,h in hashes.items():
        assert digest(ROOT/rel)==h
        dest=folder/"source_snapshot"/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
    save_json(folder/"source_sha256.json",hashes)
    target_hashes={}
    for name in TARGETS:
        path=folder/f"target_{name}.vtp";target_mesh(name).save(path);target_hashes[path.name]=digest(path)
    save_json(folder/"target_sha256.json",target_hashes)
    for rel in ["packages.txt","gpu.txt","git.txt"]:
        shutil.copy2(previous/rel,folder/rel)


def check(folder):
    for rel,h in json.loads((folder/"source_sha256.json").read_text()).items():
        assert digest(ROOT/rel)==digest(folder/"source_snapshot"/rel)==h,rel
    for rel,h in json.loads((folder/"target_sha256.json").read_text()).items():
        assert digest(folder/rel)==h
    assert json.loads((folder/"protocol.json").read_text())["targets"]==TARGETS


def scores(data,voxel=.00125):
    actual=surface(data["x_after_1s"],data["vol0"],h=voxel)
    return {name:mesh_distance_mm(actual,target_mesh(name)) for name in TARGETS}


def run_case(folder,gaps,radius=14.,first_angle=90.,grid=64,dt=.00005,device="cuda:0"):
    check(folder)
    config=dict(law=LAW,gaps_mm=np.round(gaps,6).tolist(),radius_mm=radius,first_angle_deg=first_angle,n_grid=grid,dt=dt,seed=0,device=device)
    key=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
    path=folder/"rollouts"/f"{key}.npz";path.parent.mkdir(exist_ok=True);start=time.monotonic()
    if not path.exists():
        save_json(path.with_suffix(".config.json"),config)
        with scene(radius):
            data,phases=execute(config["gaps_mm"],first_angle,grid,dt,device)
        save_npz(path,data);save_json(path.with_suffix(".phases.json"),phases)
    else:
        assert json.loads(path.with_suffix(".config.json").read_text())==config
    data=np.load(path)
    r=dict(config=config,file=str(path.relative_to(folder)),errors_mm=scores(data),rms_speed_m_s=float(np.sqrt(np.mean(np.sum(data["v_after_1s"]**2,axis=1)))))
    save_json(path.with_suffix(".json"),r);print(json.dumps(dict(seconds=time.monotonic()-start,**r)),flush=True)
    return r


def search(folder,tag,target,initial,passes,radius,budget,first_angle):
    dest=folder/"searches"/tag;dest.mkdir(parents=True,exist_ok=True)
    protocol=dict(target=target,initial_mm=initial,passes=passes,radius_mm=radius,budget=budget,first_angle_deg=first_angle,bounds_mm=[16.,54.])
    if (dest/"search.json").exists():
        assert json.loads((dest/"search.json").read_text())==protocol
    else:
        save_json(dest/"search.json",protocol)
    rows=[]
    def objective(base):
        r=run_case(folder,list(base)*passes,radius,first_angle)
        rows.append(r);save_json(dest/"evaluations.json",rows)
        print(json.dumps(dict(search=tag,evaluation=len(rows),error_mm=r["errors_mm"][target])),flush=True)
        return r["errors_mm"][target]
    x=np.array(initial,float);simplex=np.tile(x,(len(x)+1,1));simplex[1:]+=-3.*np.eye(len(x))
    minimize(objective,x,method="Nelder-Mead",bounds=[(16.,54.)]*len(x),options=dict(maxfev=budget,initial_simplex=simplex,xatol=0.,fatol=0.))
    save_json(dest/"selected.json",min(rows,key=lambda r:r["errors_mm"][target]))


def verify(folder):
    check(folder);rows=[]
    for path in sorted((folder/"rollouts").glob("*.npz")):
        data=np.load(path);r=json.loads(path.with_suffix(".json").read_text())
        assert int(data["inverted_count"])==0 and all(np.all(np.isfinite(data[k])) for k in data.files)
        np.testing.assert_allclose(data["vol0"].astype(float).sum(),.00009,rtol=1e-6)
        np.testing.assert_allclose(data["gaps_mm"],r["config"]["gaps_mm"],atol=1e-10)
        for name,v in scores(data).items():
            np.testing.assert_allclose(v,r["errors_mm"][name],rtol=1e-10)
        phases=json.loads(path.with_suffix(".phases.json").read_text())
        for i,phase in enumerate(phases):
            np.testing.assert_allclose(data["tool_centers"][data["phase_id"]==i][-1],phase["end_centers"],atol=1e-10)
            if phase["name"].endswith("rotate"):
                assert phase["duration_s"]==1.
                assert np.max(np.abs(data["reaction_force"][data["phase_id"]==i]))==0.
        assert phases[-1]["name"]=="final:wait" and phases[-1]["duration_s"]==1.
        rows.append(r)
    for search_dir in (folder/"searches").glob("*"):
        if not (search_dir/"selected.json").exists():continue
        p=json.loads((search_dir/"search.json").read_text());evaluations=json.loads((search_dir/"evaluations.json").read_text())
        assert len(evaluations)==p["budget"]
        assert json.loads((search_dir/"selected.json").read_text())==min(evaluations,key=lambda r:r["errors_mm"][p["target"]])
    save_json(folder/"summary.json",dict(rollouts=len(rows),results=rows));print(json.dumps(dict(verified_rollouts=len(rows))),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",choices=["prepare","run","search","verify"])
    p.add_argument("--out",type=Path,default=OUT);p.add_argument("--gaps",nargs="+",type=float,default=[29.,47.])
    p.add_argument("--radius",type=float,default=14.);p.add_argument("--first-angle",type=float,default=90.)
    p.add_argument("--grid",type=int,default=64);p.add_argument("--dt",type=float,default=.00005)
    p.add_argument("--tag",default="two_gap");p.add_argument("--target",choices=TARGETS,default="balanced")
    p.add_argument("--passes",type=int,default=1);p.add_argument("--budget",type=int,default=24)
    a=p.parse_args();folder=a.out.resolve()
    if a.stage=="prepare":prepare(folder)
    elif a.stage=="run":run_case(folder,a.gaps*a.passes,a.radius,a.first_angle,a.grid,a.dt)
    elif a.stage=="search":search(folder,a.tag,a.target,a.gaps,a.passes,a.radius,a.budget,a.first_angle)
    else:verify(folder)
