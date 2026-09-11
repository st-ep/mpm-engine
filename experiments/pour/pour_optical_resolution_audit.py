"""Numerical refinement of the frozen 60-degree receiver optical measurement.

Only ray quadrature and the level-search spacing change. Images, camera geometry,
ROI, colour threshold and weak fitting window remain fixed. No endpoint is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv
import numpy as np

from examples.pour_recorded_twin import SPEC, Q_RCV, R_CUP_REF
from experiments.pour import pour_perception as perception
from experiments.pour.pour_transport_identify import geometry_cache, solve

ROOT=Path(__file__).resolve().parents[2]
EP=ROOT/'pouring_real_data/09-04-60-2s'
BASE=ROOT/'out/pour_weakform_recovery/identified'
OUT=ROOT/'out/pour_physics_audit/optical_resolution'


def main(fine=False,dense=False):
    global OUT
    if fine:
        OUT=OUT.with_name('optical_resolution_fine')
    if dense:
        OUT=OUT.with_name('optical_full_rate')
        fine=True
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError('Retain previous audit outputs')
    OUT.mkdir(parents=True,exist_ok=True)
    obs=dict(np.load(BASE/'observations.npz'))
    geom=json.loads((BASE/'geometry.json').read_text())
    ident=json.loads((BASE/'identify.json').read_text())
    camera=perception.Camera(json.loads((EP/'meta.json').read_text()),'side')
    pos=np.r_[geom['receiver_xy'],geom['table_z']]
    roi=list(map(int,perception.roi_from_pose(camera,pos,R_CUP_REF)))
    roi[2:]=[238,260]  # Exactly the previous fixed right-side strip.
    u0,u1,v0,v1=roi
    rows=[json.loads(s) for s in (EP/'frames_side.jsonl').read_text().splitlines()]
    stride=1 if dense else 3
    selected=[r for r in rows if 4.5<=r['t_host']-float(obs['t_send'])<=14.3][::stride]
    reader=imageio.get_reader(EP/'side_rgb.mp4')
    masks=[];times=[]
    for r in selected:
        rgb=np.asarray(reader.get_data(r['frame_idx']))[v0:v1+1,u0:u1+1]
        hsv=rgb_to_hsv(rgb/255.)
        masks.append(((hsv[:,:,0]<.115)|(hsv[:,:,0]>.965)) & (hsv[:,:,1]>.55) & (hsv[:,:,2]>.12))
        times.append(r['t_host']-float(obs['t_send']))
    reader.close();times=np.array(times)
    protocol=dict(episode=EP.name,calibration_pour_count=1,receiver_roi_native=roi,
                  saturation_threshold=.55,
                  video_frame_stride=stride,
                  ray_level_resolutions=([[4096,.0000078125]] if dense else
                                        [[1024,.00003125],[4096,.0000078125]] if fine else
                                         [[64,.0005],[256,.000125],[1024,.00003125]]),
                  purpose='Convergence of the existing optical calculation, not calibration against volumes',
                  measured_endpoints_used=[],other_recordings_used=[])
    paths=[Path(__file__),Path(perception.__file__),BASE/'geometry.json',BASE/'observations.npz',
           BASE/'identify.json',EP/'side_rgb.mp4',EP/'meta.json',EP/'frames_side.jsonl']
    protocol['input_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    summaries=[];series={'t':times}
    depths=np.linspace(0,SPEC.rim_z-SPEC.floor_z,2001)
    volumes=np.array([SPEC.cavity_volume(d)*1e6 for d in depths])
    for steps,level_step in protocol['ray_level_resolutions']:
        perception.RAY_STEPS=steps;perception.LEVEL_STEP=level_step
        z_on,_,chord=perception.z_on_map(camera,pos,Q_RCV,roi,.001)
        estimates=[perception.fit_level(z_on,chord,m)[:2] for m in masks]
        levels=np.array([a for a,b in estimates])
        values=np.array([SPEC.cavity_volume(z-pos[2]-SPEC.floor_z)*1e6 if np.isfinite(z) else np.nan for z in levels])
        candidate={k:v.copy() for k,v in obs.items()}
        good=np.isfinite(values)
        candidate['rcv_vol']=np.interp(obs['t'],times[good],values[good],left=np.nan,right=np.nan)*1e-6
        candidate['rcv_level']=pos[2]+SPEC.floor_z+np.interp(candidate['rcv_vol']*1e6,volumes,depths)
        fit,arrays=solve(geometry_cache(candidate,ident['t_fit'],1.))
        row=dict(ray_steps=steps,level_step_m=level_step,eta_pa_s=fit['eta_pa_s'],
                 rms_ml=fit['rms_ml'],fit_gain_ml=fit['receiver_gain_ml'],n_kept=fit['n_kept'])
        summaries.append(row);series[f'volume_ml_{steps}']=values
        np.savez_compressed(OUT/f'observations_{steps}.npz',**candidate)
        np.savez_compressed(OUT/f'fit_{steps}.npz',**arrays)
        print(json.dumps(row),flush=True)
    baseline=np.load(ROOT/'out/pour_weakform_recovery/side_strip_probe.npz')
    if (not fine and (not np.allclose(times,baseline['t']) or
                     not np.allclose(series['volume_ml_64'],baseline['right_0.55'],equal_nan=True))):
        raise RuntimeError('The coarse case did not reproduce the frozen optical readout')
    np.savez_compressed(OUT/'series.npz',**series)
    (OUT/'results.json').write_text(json.dumps(dict(protocol=protocol,cases=summaries,status='Numerical audit; no model selected'),indent=2)+'\n')
    fig,axs=plt.subplots(2,1,figsize=(10,7))
    for steps,_ in protocol['ray_level_resolutions']:
        for ax in axs:
            ax.plot(times,series[f'volume_ml_{steps}'],'.-',ms=3,label=f'{steps} ray samples')
    axs[0].set(xlim=(5,11.8),ylabel='Receiver volume (mL)')
    axs[1].set(xlim=ident['t_fit'],ylim=(110,165),xlabel='Seconds after tilt command',ylabel='Receiver volume (mL)')
    for ax in axs: ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'refinement.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fine',action='store_true')
    parser.add_argument('--dense',action='store_true')
    args=parser.parse_args()
    main(args.fine,args.dense)
