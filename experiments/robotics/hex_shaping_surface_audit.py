"""Geometry, input-integrity, and reconstruction-sensitivity audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from experiments.robotics.hex_shaping_pilot import BAND, jaw_pose
from experiments.robotics.hex_shaping_surface import mesh_distance_mm, target_surface, surface_error_mm
from experiments.robotics.hex_shaping_surface_study import OUT, MODELS, ROOT, check_sources
from experiments.robotics.plastic_shaping_figure import surface
from experiments.robotics.plastic_shaping_study import save_json


def audit(folder, previous):
    check_sources(folder)
    current_sources=json.loads((folder/"source_sha256.json").read_text())
    previous_sources=json.loads((previous/"source_sha256.json").read_text())
    shared=set(current_sources)&set(previous_sources)
    for rel in shared:
        assert current_sources[rel]==previous_sources[rel],rel
    imports=json.loads((folder/"imported_inputs.json").read_text())
    for rel,record in imports.items():
        assert hashlib.sha256(Path(record["source"]).read_bytes()).hexdigest()==record["sha256"],rel
    unique={model:len(list((folder/"plans"/model).glob("*.npz"))) for model in MODELS}
    work_files=[p for p in folder.glob("*.npz") if not p.name.startswith("target")]
    assert len(work_files)==28
    for path in work_files:
        d=np.load(path)
        for key in ["initial","x_after_1s"]:
            assert np.all((d[key]>=0.)&(d[key]<=.30)),(path.name,key)
        if "requested_work_j" in d:
            record=json.loads(path.with_suffix(".json").read_text())
            expected=["work" if w>=b else "stroke_limit" for w,b in zip(d["work_j"],d["requested_work_j"],strict=True)]
            assert record["stops"]==expected,path.name
            calibration=json.loads((folder/f"calibration_{record['model']}.json").read_text())
            np.testing.assert_array_equal(d["requested_work_j"],calibration["work_limits_j"])
        for i,angle in enumerate(d["angles"]):
            centers,normal,quat=jaw_pose(angle,float(d["gaps"][i]),float(d["floor"]))
            np.testing.assert_allclose(d[f"stage_{i}_centers"],centers,atol=1e-12)
            np.testing.assert_allclose(d[f"stage_{i}_quat"],quat,atol=1e-12)
            contact_gap=float(np.dot(centers[1]-centers[0],normal)-2*d["half"][0]-2*BAND)
            np.testing.assert_allclose(contact_gap,d["gaps"][i],atol=1e-12)
            for key in [f"stage_{i}_pressed",f"stage_{i}_released"]:
                assert np.all((d[key]>=0.)&(d[key]<=.30)),(path.name,key)
            np.testing.assert_allclose(d[f"stage_{i}_gap"][-1],d["gaps"][i],atol=1e-12)
            assert np.all(np.diff(d[f"stage_{i}_gap"])<0.)
    target=np.load(folder/"target.npz");mesh=target_surface(target)
    refined=mesh.subdivide(1)
    sensitivity={};quadrature={}
    for label,source in [("previous",previous),("updated",folder)]:
        sensitivity[label]={};quadrature[label]={}
        for name in "AB":
            for method in ["nominal","identified","oracle"]:
                d=np.load(source/f"baseline_{name}_{method}.npz")
                key=f"{name}_{method}"
                sensitivity[label][key]={str(h):mesh_distance_mm(surface(d["x_after_1s"],d["vol0"],h=h),mesh)
                                         for h in [.002,.0025,.003]}
                if method=="identified":
                    actual=surface(d["x_after_1s"],d["vol0"])
                    quadrature[label][name]=[mesh_distance_mm(actual,mesh),mesh_distance_mm(actual,refined)]
    target_reconstruction=surface_error_mm(dict(x_after_1s=target["x"],vol0=target["vol0"]),mesh)
    result=dict(unique_planning_rollouts=unique,work_rollouts=len(work_files),
        unchanged_shared_source_files=len(shared),
        verified_work_stages=6*len(work_files),imported_originals_unchanged=True,
        reconstruction_sensitivity_mm=sensitivity,target_quadrature_refinement_mm=quadrature,
        target_particle_reconstruction_error_mm=target_reconstruction,
        caveat="Reconstruction diagnostic is not a lower bound or correction to subtract")
    save_json(folder/"additional_audit.json",result)
    dest=folder/"audit_source";dest.mkdir(exist_ok=True);shutil.copy2(__file__,dest/Path(__file__).name)
    save_json(dest/"sha256.json",{str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=OUT)
    parser.add_argument("--previous",type=Path,default=ROOT/"out/hex_shaping_work_20260909")
    args=parser.parse_args();audit(args.out.resolve(),args.previous.resolve())


if __name__=="__main__":
    main()
