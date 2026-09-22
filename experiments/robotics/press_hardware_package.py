"""Write the business report, verify media, and freeze hardware-study provenance."""
import argparse
import hashlib
import platform
from pathlib import Path
import shutil
import sys
import cv2
import numpy as np
import scipy
import warp as wp
from experiments.robotics.press_hardware_observe import read,save
from experiments.robotics.press_hardware_report import LABELS

ROOT=Path(__file__).resolve().parents[2]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(1024*1024):h.update(chunk)
    return h.hexdigest()


def package(out):
    out=out.resolve();p=read(out/'protocol.json');s=read(out/'validation_summary.json')['materials']
    assert read(out/'implementation_checks.json')['status']=='passed'
    assert read(out/'identification_isolation_check.json')['status']=='passed'
    required=['validation','check_grid96','check_friction0','check_friction0.3']
    for m in p['materials']:
        fit=read(out/m/'fit.json')
        for name in required:
            c=read(out/m/name/'completion.json');assert c['complete']
            for k in ['E_Pa','Y_Pa','tau_s']:
                assert c['parameters'][k]==fit['selected'][k],(m,name,k)
        shutil.copy2(out/m/'fit.json',out/m/'frozen_fit.json')
    media=[]
    for path in sorted((out/'media').glob('*.mp4')):
        cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);size=[int(cap.get(3)),int(cap.get(4))]
        count=0;first=None;last=None
        while True:
            ok,frame=cap.read()
            if not ok:break
            if first is None:first=frame.copy()
            last=frame;count+=1
        cap.release();assert count==269 and abs(fps-20)<1e-6,(path,count,fps)
        assert np.mean(abs(first.astype(float)-last.astype(float)))>1
        media.append(dict(path=str(path),frames=count,fps=fps,size=size,sha256=digest(path),fully_decoded=True))
    assert len(media)==7
    save(out/'media_validation.json',dict(status='passed',videos=media,visual_inspection='Comparison contact sheets at 0, 5, 10 and 13.4 seconds inspected separately'))
    params=[];metrics=[];clips=[]
    for m in p['materials']:
        d=s[m];tau=f"{d['tau_s']:.2f}" if d['tau_s'] else 'Not selected'
        y=f"{d['Y_numeric_Pa']/1000:.3g}" if d['yield_resolved_in_sampled_search'] else ('Unresolved' if d['yield_weak_log_sensitivity']<1e-6 else f"{d['Y_numeric_Pa']/1000:.3g}, weak")
        law='Hencky + rate-dependent von Mises flow' if d['tau_s'] else 'Hencky / perfect von Mises candidate'
        params.append(f"| {LABELS[m]} | {law} | {d['E_equivalent_Pa']/1000:.1f} | {y} | {tau} |")
        metrics.append(f"| {LABELS[m]} | {d['loading_gap_rmse_mm']:.2f} | {d['loading_width_rmse_mm']:.2f} | {d['unloading_gap_rmse_mm']:.2f} |")
        clips.append(f"| {LABELS[m]} | [Reality]({out}/media/{m}_reality.mp4) | [Simulation]({out}/media/{m}_simulation.mp4) |")
    report=f'''# Real pressing: business report

**Decision: the end-to-end study is complete, but these models are not yet validated for final X-shaping gaps.** Play-Doh is the strongest starting candidate. Butter slime needs a better-constrained yield response; plasticine's yield stress remains unidentified by this reconstruction and fit.

## Delivered

- Prepared all 12 presses; fitted on 6, 11 and 31 N, reserving each material's 21 N recording for prediction.
- Reconstructed a smooth, volume-preserving axisymmetric motion field from visible shape and robot motion. Its interior correspondence is an explicit assumption.
- Compared two constitutive candidates and ran force-driven **3D MPM** with frozen estimates. Recorded displacement was not prescribed to the simulator.
- Produced six individual clips and the synchronized [three-by-two comparison video]({out}/media/pressing_reality_vs_simulation.mp4), including low-load unloading before separation.

This is an exploratory comparison excluded from parameter fitting, not a new independent hardware experiment. Prediction runs were inspected during numerical and input-processing development.

## Candidate parameters and constitutive laws

These are conditional model estimates, not independently measured material constants.

| Material | Constitutive law used | Equivalent E (kPa) | Y (kPa) | Relaxation time (s) |
|---|---|---:|---:|---:|
{chr(10).join(params)}

E is derived from fitted shear stiffness with **bulk modulus fixed at 1 MPa**. Y follows the repository's deviatoric Kirchhoff-stress norm convention. Plasticine's numerical threshold in the simulation is 54.3 kPa, but doubling it leaves the fitted weak response unchanged: that value is **not an identified yield stress**. Density, bulk stiffness, friction and adhesion were not identified.

## What the withheld predictions achieved

| Material | Loading gap RMS error (mm) | Loading width RMS error (mm) | Unloading gap RMS error (mm) |
|---|---:|---:|---:|
{chr(10).join(metrics)}

Loading covers 0.4–10.4 s; unloading covers 11–13.4 s. Gap uses measured robot displacement with an assumed initial height; width is an RGB silhouette estimate. Plasticine's raw width includes isolated segmentation failures; its late-hold width error is 2.56 mm, while its late-hold gap error remains 5.70 mm. Late-hold force tracking error is below 0.3% for all three. Force agreement verifies the imposed load, not material accuracy. **None passes the combined provisional gap/width screen.**

## Recommendation

Do not send final finger gaps yet. First resolve initial specimen/plate geometry and improve or bound the surface-to-interior reconstruction. A ±2 mm height assumption changes fitted shear stiffness by roughly 7–21%; an added 1 N force offset substantially changes the inferred yield response, especially for slime. These are sensitivity tests, not uncertainty confidence intervals. Slight stickiness remains unmeasured; contact tests do not measure or eliminate adhesion.

The next useful result is a model that predicts both compression and recovery across withheld presses under those uncertainties. Re-recording EP005/EP010 solely to restore their second depth stream is not the first priority.

| Material | Recorded clip | Predicted clip |
|---|---|---|
{chr(10).join(clips)}

[Technical details]({out}/TECHNICAL.md) · [Validation curves]({out}/validation_curves.png) · [Exact parameters and metrics]({out}/validation_summary.json) · [Reproduction]({out}/REPRODUCE.md)

Raw recordings and the manuscript were preserved. No commits, pushes or messages to Philip were sent.
'''
    (out/'REPORT.md').write_text(report)
    with (out/'PARAMETERS.csv').open('w') as f:
        f.write('material,G_Pa,E_equivalent_Pa,K_assumed_Pa,nu_equivalent,Y_numeric_Pa,yield_status,tau_s,eta_Pa_s\n')
        for m in p['materials']:
            d=s[m];f.write(','.join(map(str,[m,d['G_Pa'],d['E_equivalent_Pa'],d['bulk_assumed_Pa'],d['nu_equivalent'],d['Y_numeric_Pa'],d['yield_status'],d['tau_s'],d['eta_Pa_s']]))+'\n')
    # Numerical source snapshot; contains no raw imagery or manuscript.
    source_paths=list((ROOT/'experiments/robotics').glob('press_hardware_*.py'))
    source_paths+=list((ROOT/'src/warpmpm').rglob('*.py'))
    snapshot=out/'source_snapshot'
    for src in source_paths:
        dest=snapshot/src.relative_to(ROOT);dest.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(src,dest)
    manifest=dict(status='End-to-end study complete; physical validation screen not passed',protocol=str(out/'protocol.json'),
        material_parameters='Conditional estimates; plasticine yield unresolved; see REPORT.md',
        sources={str(src.relative_to(ROOT)):digest(src) for src in source_paths},
        environment=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,opencv=cv2.__version__,warp=wp.__version__),
        inputs={},outputs={},raw_inventory=str(ROOT/'out/press_data_inventory_20260913/REPORT.md'),
        observation_assessment=str(ROOT/'out/press_observation_assessment_20260913/dataset_manifest.json'))
    for i in range(12):
        ep=f'ep{i:04}'
        for name in ['motion.npz','geometry.json','load_only.npz']:manifest['inputs'][str(out/ep/name)]=digest(out/ep/name)
        for name in ['meta.json','actions.jsonl','states.jsonl','frames_hand.jsonl','hand_rgb.mp4']:
            path=ROOT/'press_real_data'/ep/name;manifest['inputs'][str(path)]=digest(path)
    for m in p['materials']:
        for path in (out/m).rglob('*'):
            if path.is_file() and ('check_' in str(path.parent.name) or path.parent.name=='validation' or path.parent.name==m):
                manifest['outputs'][str(path)]=digest(path)
    for path in out.iterdir():
        if path.is_file() and path.name!='manifest.json':manifest['outputs'][str(path)]=digest(path)
    manifest['outputs'].update({v['path']:v['sha256'] for v in media})
    save(out/'manifest.json',manifest)
    print('Packaged report, 7 decoded videos, source snapshot and manifest',flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--out',type=Path,required=True);package(a.parse_args().out)
