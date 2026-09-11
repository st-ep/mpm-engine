"""Model-planned work limits for six squeezes toward the existing hexagon.

This changes the execution policy, not the materials, nominal model, or target.
It imports the five equally budgeted gap plans from the active paper study,
predicts their positive tool work, and executes those work limits using force
feedback. No true-material geometry enters the stopping rule.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil

import numpy as np

from experiments.robotics.hex_shaping_control import snapshot_sources, target_prism
from experiments.robotics.hex_shaping_pilot import ANGLES, BAND, HALF, box_sdf, jaw_pose
from experiments.robotics.hex_shaping_study import TRUTH, MODELS
from experiments.robotics.plastic_shaping_study import ROOT, NOMINAL, chamfer_mm, save_json, save_npz
from warpmpm import GridConfig, Solver
from warpmpm.materials import vonmises
from warpmpm.scenes import block

OUT = ROOT/"out/hex_shaping_work_20260909"
SOURCE = ROOT/"out/hex_shaping_study_20260908"
PROTOCOL = dict(truth=TRUTH, nominal=NOMINAL, angles_deg=list(ANGLES)*2,
    policy="Constant-speed closing until cumulative positive work of both jaws reaches the model-predicted stage budget",
    work="Sum over ticks and jaws of max(-reaction_force dot jaw_velocity, 0) * control_dt, joules",
    planning="Existing common-target gap planning, followed by model-only work prediction; no retuning on actual outcomes",
    force_feedback=True, shape_feedback=False, opening_mm=160., minimum_gap_mm=65.,
    minimum_gap_reason="A volume-matched 65 mm hexagon fits approximately within the 160 mm jaw height",
    speed_mm_s_per_jaw=100., n_grid=48, domain_m=.30, dt=1e-4, control_dt=.0008,
    release_between_s=.30, final_observation_s=1., ppc=2, seed=0,
    target_diameter_m=.090, initial_size_m=[.12,.08,.06], density=1000.,
    contact="Sticky floor and SDF jaws; idealized pair insertion and removal",
    budget_overshoot="Stop on first control tick reaching budget; at most one tick of work overshoot",
    stroke_limit="All methods stop at 65 mm if the work budget has not been reached",
    scope="Simulation control experiment; work feedback is separate from supplied-state identification")


def work_increment(reaction_forces, jaw_velocities, tick):
    """Positive mechanical work supplied by both translating jaws, in joules."""
    forces = np.asarray(reaction_forces, dtype=float)
    velocity = np.asarray(jaw_velocities, dtype=float)
    if forces.shape != (2, 3) or velocity.shape != (2, 3) or tick <= 0:
        raise ValueError("Expected two 3D forces and velocities and a positive time interval")
    return float(np.maximum(-np.sum(forces*velocity, axis=1), 0.).sum()*tick)


def execute(law, *, device, gaps_mm=None, budgets_j=None, n_grid=48, dt=1e-4):
    """Calibrate with gaps, or execute work limits, retaining forces and all stops."""
    if (gaps_mm is None) == (budgets_j is None):
        raise ValueError("Provide exactly one of gaps_mm and budgets_j")
    planned = None if gaps_mm is None else np.tile(np.asarray(gaps_mm, dtype=float), 2)
    budgets = None if budgets_j is None else np.asarray(budgets_j, dtype=float)
    if planned is not None and (planned.shape != (6,) or np.any((planned < 65.) | (planned >= 160.))):
        raise ValueError("Expected three gaps in [65, 160) mm")
    if budgets is not None and (budgets.shape != (6,) or np.any(budgets <= 0) or not np.all(np.isfinite(budgets))):
        raise ValueError("Expected six positive finite work limits")
    grid = GridConfig(n_grid=n_grid, grid_lim=.30)
    pos, vol0, floor = block(grid, size=(.12,.08,.06), ppc=2, seed=0)
    with contextlib.redirect_stdout(io.StringIO()):
        solver = Solver(grid, device=device, inversion_policy="raise").load_particles(pos, vol0)
        solver.set_material(vonmises(**law, density=1000.))
    solver.add_plane(point=(0,0,floor), normal=(0,0,1), surface="sticky")
    sdf = box_sdf(); tick=.0008; opening=.160; maximum_speed=.10
    data = dict(initial=pos, vol0=vol0, floor=np.array(floor), half=HALF,
                angles=np.tile(ANGLES,2), n_grid=np.array(n_grid), dt=np.array(dt))
    elapsed=0.; actual_gaps=[]; work_totals=[]; stop_reasons=[]
    for stage, angle in enumerate(data["angles"]):
        centers, normal, quat = jaw_pose(angle, opening, floor)
        handles=[solver.add_sdf_collider(sdf, center=c, quat=quat, band=BAND,
                 surface="sticky", friction=0., start_time=elapsed-.5*dt) for c in centers]
        final_gap=.065 if planned is None else planned[stage]/1000
        count=int(np.ceil((opening-final_gap)/(2*maximum_speed*tick)))
        # Calibration matches the preceding gap planner's quantized motion.
        speed=maximum_speed if planned is None else (opening-final_gap)/(2*count*tick)
        gap=opening; total=0.; trace=[]
        for step in range(count):
            v=min(speed,(gap-final_gap)/(2*tick))
            velocity=np.array([1.,-1.])[:,None]*normal*v
            starts=jaw_pose(angle,gap,floor)[0]
            for h,c,vel in zip(handles,starts,velocity,strict=True):
                solver.set_sdf_pose(h,center=c,velocity=vel)
                solver.reset_sdf_force(h)
            solver.step(dt,substeps=round(tick/dt))
            forces=np.stack([solver.sdf_wrench(h,tick)["force"] for h in handles])
            increment=work_increment(forces,velocity,tick)
            total+=increment;gap-=2*v*tick
            trace.append(dict(time=elapsed+(step+1)*tick,gap=gap,force=forces,
                              velocity=velocity,increment=increment,work=total))
            if budgets is not None and total >= budgets[stage]:
                break
        stopped_on_work=budgets is not None and total >= budgets[stage]
        actual_gaps.append(gap);work_totals.append(total)
        stop_reasons.append("gap_calibration" if planned is not None else ("work" if stopped_on_work else "stroke_limit"))
        data[f"stage_{stage}_pressed"]=solver.x().copy()
        data[f"stage_{stage}_centers"]=jaw_pose(angle,gap,floor)[0]
        data[f"stage_{stage}_quat"]=quat
        for key in trace[0]:
            data[f"stage_{stage}_{key}"]=np.array([r[key] for r in trace])
        for h in handles:
            solver.set_sdf_pose(h,center=(.9,.9,.9),velocity=(0,0,0))
        for _ in range(round(.30/tick)):
            solver.step(dt,substeps=round(tick/dt))
        elapsed+=len(trace)*tick+.30
        data[f"stage_{stage}_released"]=solver.x().copy()
    data.update(x=solver.x().copy(),v=solver.v().copy(),gaps=np.array(actual_gaps),
                work_j=np.array(work_totals),elapsed=np.array(elapsed))
    for _ in range(round(.70/tick)):
        solver.step(dt,substeps=round(tick/dt))
    data.update(x_after_1s=solver.x().copy(),v_after_1s=solver.v().copy(),
                inverted_count=np.array(solver.inverted_count()))
    if int(data["inverted_count"]) or not all(np.all(np.isfinite(v)) for v in data.values()):
        raise RuntimeError("Invalid work-controlled execution")
    if budgets is not None:
        data["requested_work_j"]=budgets
    return data,stop_reasons


def prepare(folder, source, device):
    snapshot_sources(folder,PROTOCOL)
    files=[Path(__file__),ROOT/"experiments/robotics/hex_shaping_study.py",
           ROOT/"experiments/robotics/hex_shaping_check.py",
           ROOT/"experiments/robotics/hex_shaping_six_report.py",
           ROOT/"experiments/robotics/hex_shaping_six_gaps.py",
           ROOT/"experiments/robotics/hex_shaping_control_report.py",
           ROOT/"experiments/robotics/hex_shaping_figure.py",
           ROOT/"experiments/robotics/plastic_shaping_figure.py"]
    hashes=json.loads((folder/"source_sha256.json").read_text())
    for p in files:
        rel=str(p.relative_to(ROOT));dest=folder/"source_snapshot"/rel
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()!=digest:
            raise RuntimeError("Source changed; use a fresh output folder")
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);hashes[rel]=digest
    save_json(folder/"source_sha256.json",hashes)
    # Confirm that the shared source study uses the current physics and input laws.
    old_hashes=json.loads((source/"source_sha256.json").read_text())
    for rel,digest in old_hashes.items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,rel
    models=json.loads((source/"models.json").read_text())
    assert models["nominal"]==NOMINAL and all(models[f"true_{n}"]==law for n,law in TRUTH.items())
    imports={}
    for rel in ["models.json","target.npz",*[f"plans/{m}/selected.json" for m in MODELS]]:
        dest=folder/"inputs"/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/rel,dest)
        imports[rel]=dict(source=str(source/rel),sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
    for model in MODELS:
        rows=json.loads((source/"plans"/model/"evaluations.json").read_text())
        assert len(rows)==32 and all(min(r["gaps_mm"])>=65 for r in rows)
    save_json(folder/"imported_inputs.json",imports)
    save_json(folder/"device.json",dict(device=device))
    save_npz(folder/"target.npz",target_prism())
    paths=["paper/icra2027/paper.tex","paper/icra2027/paper.pdf",
           "paper/icra2027/figs/identification_plastic_shaping.pdf",
           "paper/icra2027/figs/identification_plastic_shaping.png"]
    save_json(folder/"manuscript_before.json",{rel:hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() for rel in paths})


def check_sources(folder):
    assert json.loads((folder/"protocol.json").read_text())==PROTOCOL
    for rel,h in json.loads((folder/"source_sha256.json").read_text()).items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h,rel
        assert hashlib.sha256((folder/"source_snapshot"/rel).read_bytes()).hexdigest()==h,rel
    for rel,record in json.loads((folder/"imported_inputs.json").read_text()).items():
        assert hashlib.sha256((folder/"inputs"/rel).read_bytes()).hexdigest()==record["sha256"],rel


def calibrate(folder,model,device):
    check_sources(folder)
    selected=json.loads((folder/"inputs/plans"/model/"selected.json").read_text())
    path=folder/f"calibration_{model}.npz"
    if not path.exists():
        data,_=execute(selected["law"],gaps_mm=selected["gaps_mm"],device=device)
        save_npz(path,data)
    data=np.load(path)
    source=json.loads((folder/"imported_inputs.json").read_text())[f"plans/{model}/selected.json"]["source"]
    old_root=Path(source).parents[2]
    old=np.load(old_root/selected["file"])
    diff=chamfer_mm(data["x"],old["x"])
    assert diff<.05,(model,diff)
    result=dict(model=model,law=selected["law"],planned_gaps_mm=selected["gaps_mm"],
                work_limits_j=data["work_j"].tolist(),gap_replay_difference_mm=diff)
    save_json(path.with_suffix(".json"),result);print(json.dumps(result),flush=True)


def run_selected(folder,model,device,stage):
    check_sources(folder)
    calibration=json.loads((folder/f"calibration_{model}.json").read_text())
    method="nominal" if model=="nominal" else ("oracle" if model.startswith("true_") else "identified")
    names="AB" if model=="nominal" else model[-1]
    settings=[("baseline",48,1e-4)] if stage!="numerics" else [("half_dt",48,5e-5),("grid64",64,1e-4)]
    if stage=="selfcheck": names=[model]
    for tag,grid,dt in settings:
        target=target_prism(grid);save_npz(folder/f"target_{tag}.npz",target)
        for name in names:
            stem=f"selfcheck_{model}" if stage=="selfcheck" else f"{tag}_{name}_{method}"
            path=folder/f"{stem}.npz"
            if not path.exists():
                law=calibration["law"] if stage=="selfcheck" else TRUTH[name]
                data,stops=execute(law,budgets_j=calibration["work_limits_j"],device=device,n_grid=grid,dt=dt)
                save_npz(path,data);save_json(folder/f"{stem}_stops.json",stops)
            data=np.load(path)
            record=dict(model=model,material=name,method=method,setting=tag,
                work_limits_j=calibration["work_limits_j"],actual_work_j=data["work_j"].tolist(),
                actual_gaps_mm=(data["gaps"]*1000).tolist(),
                stops=json.loads((folder/f"{stem}_stops.json").read_text()),
                error_03s_mm=chamfer_mm(data["x"],target["x"]),
                error_1s_mm=chamfer_mm(data["x_after_1s"],target["x"]),
                displacement_03_to_1s_mm=chamfer_mm(data["x"],data["x_after_1s"]))
            if stage=="selfcheck":
                ref=np.load(folder/f"calibration_{model}.npz")
                record["gap_difference_mm"]=(1000*(data["gaps"]-ref["gaps"])).tolist()
                record["cloud_difference_1s_mm"]=chamfer_mm(data["x_after_1s"],ref["x_after_1s"])
            save_json(folder/f"{stem}.json",record);print(json.dumps(record),flush=True)


def verify(folder):
    from experiments.robotics.hex_shaping_six_report import slices,corner_to_face_ratio
    from experiments.robotics.hex_shaping_control_report import section_iou
    check_sources(folder);results=[];counts={}
    for path in sorted(folder.glob("*.npz")):
        if path.name.startswith("target"): continue
        data=np.load(path)
        assert not int(data["inverted_count"])
        assert all(np.all(np.isfinite(data[k])) for k in data.files)
        counts[path.stem]=6
        for i in range(6):
            forces=data[f"stage_{i}_force"];velocities=data[f"stage_{i}_velocity"]
            steps=np.maximum(-np.sum(forces*velocities,axis=2),0).sum(axis=1)*.0008
            np.testing.assert_allclose(steps,data[f"stage_{i}_increment"],rtol=1e-12)
            np.testing.assert_allclose(steps.cumsum(),data[f"stage_{i}_work"],rtol=1e-12)
            np.testing.assert_allclose(steps.sum(),data["work_j"][i],rtol=1e-12)
            assert data["gaps"][i]>=.065-1e-12
            if "requested_work_j" in data:
                budget=data["requested_work_j"][i]
                if data["work_j"][i]>=budget:
                    assert data["work_j"][i]-steps[-1]<budget+1e-12
                else:
                    np.testing.assert_allclose(data["gaps"][i],.065,atol=1e-12)
        if path.stem.startswith(("baseline_","half_dt_","grid64_")):
            row=json.loads(path.with_suffix(".json").read_text())
            target=np.load(folder/f"target_{row['setting']}.npz")
            np.testing.assert_allclose(chamfer_mm(data["x_after_1s"],target["x"]),row["error_1s_mm"],rtol=1e-12)
            row["slice_hull_iou"]=section_iou(data,target,"x_after_1s")
            row["corner_to_face_ratio"]=[corner_to_face_ratio(xy) for xy in slices(data,target)]
            results.append(row)
    before=json.loads((folder/"manuscript_before.json").read_text())
    summary=dict(results=results,verified_traces=counts,
        manuscript_unchanged=all(hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h for rel,h in before.items()))
    save_json(folder/"summary.json",summary);print(json.dumps(summary,indent=2),flush=True)
    hashes={str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob("*")
            if p.is_file() and p.name!="checksums.json" and p.suffix!=".log"}
    save_json(folder/"checksums.json",hashes)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["prepare","calibrate","selfcheck","execute","numerics","verify"])
    parser.add_argument("--out",type=Path,default=OUT)
    parser.add_argument("--source",type=Path,default=SOURCE)
    parser.add_argument("--device",default="cuda:1")
    parser.add_argument("--model",choices=MODELS,default="nominal")
    args=parser.parse_args();folder=args.out.resolve()
    if args.stage=="prepare": prepare(folder,args.source.resolve(),args.device)
    elif args.stage=="calibrate": calibrate(folder,args.model,args.device)
    elif args.stage=="verify": verify(folder)
    else: run_selected(folder,args.model,args.device,args.stage)


if __name__=="__main__":
    main()
