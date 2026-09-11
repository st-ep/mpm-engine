"""Frozen reduced-law diagnostic on actual joint trajectories (not an MPM table).

This checks transfer of the identified LOCAL spout law separately from MPM's
different wall physics. It assumes the 60-degree hand-to-cup transform persists.
No other video's fluid measurements are used, and no endpoint is read or fitted.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

from examples.pour_recorded_twin import RecordedPanda, load_episode, quat_to_mat
from experiments.pour.pour_perception import build_cavity_lattice,rim_curve_local

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'out/pour_weakform_recovery/identified'
OUT=ROOT/'out/pour_physics_audit/frozen_brink_diagnostic'


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    fitpath=ROOT/'out/pour_physics_audit/optical_resolution_fine/results.json'
    fits=json.loads(fitpath.read_text())
    fit=next(r for r in fits['cases'] if r['ray_steps']==4096)
    eta=fit['eta_pa_s']
    geometry=json.loads((BASE/'geometry.json').read_text())
    paths=[Path(__file__),fitpath,BASE/'geometry.json',
           ROOT/'experiments/pour/pour_perception.py',ROOT/'src/warpmpm/geometry/measuring_cup.py']
    protocol=dict(eta_pa_s=eta,identification_pour='09-04-60-2s',calibration_pour_count=1,
                  weak_method='Conservative time-weak spout-edge law',
                  model='Hydrostatic-head brink flux integrated forward; NOT MPM',
                  cases=[['09-04-60-2s',.02],['09-04-60-2s',.01],
                         ['09-04-45-2s',.02],['09-04-50-2s',.02]],
                  motion='Actual joint logs; fixed rigid grasp from 60-degree recording',
                  parameters_frozen_before_other_motion_evaluation=True,
                  endpoint_fitting=False,other_fluid_videos_used=False,
                  previously_seen_diagnostic_episodes=['09-04-45-2s','09-04-50-2s'],
                  blind_validation=False,release_status='Diagnostic only; no robot commands',
                  input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (OUT/'frozen_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    mesh=ROOT/'out/pour_wf/09-04-60-2s/cup_render.obj'
    ep60=load_episode(ROOT/'pouring_real_data/09-04-60-2s',1.,2.5)
    reference=RecordedPanda(ep60,mesh,height=64,width=64,max_geom=4000,
                            cup_reference_pos=geometry['cup_reference_pos'],
                            cup_reference_quat=geometry['cup_reference_quat'])
    hand_cup,grasp=reference._hand_cup.copy(),reference._grasp.copy();reference.close()
    lattice,cell=build_cavity_lattice();rim=rim_curve_local();rim=rim[abs(rim[:,1])<=.019]
    results=[]
    for episode,dt in protocol['cases']:
        path=ROOT/'pouring_real_data'/episode
        ep=load_episode(path,1.,2.5)
        arm=RecordedPanda(ep,mesh,height=64,width=64,max_geom=4000)
        arm._hand_cup=hand_cup.copy();arm._grasp=grasp.copy()
        n=int(np.ceil(ep['duration']/dt));times=np.linspace(0,ep['duration'],n+1);dt=times[1]-times[0]
        source=300.;depletion=[0.];tilts=[]
        for t in times[:-1]:
            pos,quat=arm.cup_pose_at(t+dt/2)
            rot=quat_to_mat(quat);wz=np.sort(lattice@rot[2]);rz=(rim@rot.T)[:,2]
            sine=abs(float(rot[2,0]));cosine=np.sqrt(max(1-sine*sine,1e-6))
            def flux(v):
                if v<=0: return 0.
                index=int(np.clip(round(v*1e-6/cell),1,len(wz)-1))
                head=np.maximum(wz[index]-rz,0)*cosine
                return 1e6*1260*9.81*sine*np.trapezoid(head**3,rim[:,1])/(3*eta)
            q0=flux(source);qm=flux(max(source-.5*dt*q0,0));source=max(0,source-dt*qm)
            depletion.append(300-source);tilts.append(arm.tilt_degrees(quat))
        arm.close()
        name=f'{episode}_dt{dt:.6f}'
        np.savez_compressed(OUT/f'{name}.npz',t=times-ep['t_pour'],source_depletion_ml=depletion,
                            tilt_deg=tilts)
        result=dict(episode=episode,dt_s=dt,eta_pa_s=eta,source_depletion_ml=300-source,
                    settled_receiver_ml_assuming_no_spill=300-source,max_physical_tilt_deg=max(tilts),
                    tail_source_depletion_ml=float(depletion[-1]-np.interp(times[-1]-.5,times,depletion)),
                    motion_hashes={k:hashlib.sha256((path/k).read_bytes()).hexdigest() for k in ['states.jsonl','actions.jsonl','meta.json']})
        results.append(result);print(json.dumps(result),flush=True)
        (OUT/'results.json').write_text(json.dumps(dict(protocol=protocol,cases=results),indent=2)+'\n')


if __name__=='__main__':
    main()
