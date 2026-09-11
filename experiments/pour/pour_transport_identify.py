"""Candidate weak-form fit with conservative source-to-receiver flight transport.

Uses only the existing 60-degree optical observations and its measured geometry.
The baseline artifacts are never overwritten. No simulator or measured endpoint
appears in the estimation objective. The original hold-window rule is retained.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.pour.pour_perception import build_cavity_lattice, rim_curve_local
from experiments.pour.pour_weakform_identify import fit_eta_integrated, smooth_series
from experiments.pour.pour_weakform_transport import transported_forcing
from warpmpm.colliders.glass import quat_to_mat

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'out/pour_weakform_recovery/identified'
OUT = ROOT/'out/pour_physics_audit/transport'


def geometry_cache(obs, window, prehistory):
    smoothed, _ = smooth_series(obs['t'],obs['rcv_vol']*1e6,s_win=.30)
    take = (np.isfinite(smoothed) & (obs['t'] >= window[0]-prehistory)
            & (obs['t'] <= window[1]+.2))
    indices = np.flatnonzero(take)
    t, receiver = obs['t'][indices], smoothed[indices]
    if len(t) < 12:
        raise ValueError('Insufficient optical history')
    lattice, cell = build_cavity_lattice()
    rim = rim_curve_local(); rim = rim[abs(rim[:,1]) <= .019]
    columns, rim_heights, cosines, sines = [], [], [], []
    for i in indices:
        rot = quat_to_mat(obs['cup_quat'][i])
        columns.append(np.sort(lattice@rot[2]+obs['cup_pos'][i,2]))
        rim_heights.append((rim@rot.T+obs['cup_pos'][i])[:,2])
        sine = abs(float(rot[2,0])); sines.append(sine)
        cosines.append(np.sqrt(max(1-sine*sine,1e-6)))
    fall = np.maximum(obs['lip'][indices,2]-obs['rcv_level'][indices],0)
    transit = np.sqrt(2*fall/9.81)
    fitmask = (t >= window[0]) & (t <= window[1])
    if t[fitmask][0]-transit.max() < t[0]+.3:
        raise ValueError('Not enough prehistory for transported forcing')
    return dict(t=t,receiver=receiver,columns=np.array(columns),rim_z=np.array(rim_heights),
                cosine=np.array(cosines),sine=np.array(sines),transit=transit,
                cell=cell,ys=rim[:,1],fitmask=fitmask)


def solve(cache, initial_air_ml=0.):
    t, receiver = cache['t'], cache['receiver']
    air = np.full(len(t),initial_air_ml)
    history=[]
    for iteration in range(80):
        source_volume = 300.-receiver-air
        if np.any(source_volume <= 0):
            raise ValueError('Nonpositive inferred source volume')
        index = np.clip(np.rint(source_volume*1e-6/cache['cell']).astype(int),
                        1,cache['columns'].shape[1]-1)
        level = cache['columns'][np.arange(len(t)),index]
        head = np.maximum(level[:,None]-cache['rim_z'],0)*cache['cosine'][:,None]
        forcing = 1e6*1260*9.81*cache['sine']*np.trapezoid(head**3,cache['ys'],axis=1)/3
        released, arrived = transported_forcing(t,forcing,cache['transit'],t)
        mask = cache['fitmask']
        eta,se,keep,dv,cf = fit_eta_integrated(dict(t=t[mask],V=receiver[mask],cumF=arrived[mask]))
        if not np.isfinite(eta) or eta <= 0:
            raise ValueError('Nonpositive weak-form viscosity')
        next_air = (released-arrived)/eta
        change = float(np.max(abs(next_air-air)))
        history.append(dict(iteration=iteration,eta_pa_s=eta,max_air_change_ml=change))
        if change < .0005:
            break
        air = .5*air+.5*next_air
    else:
        raise RuntimeError('Transport/head iteration did not converge')
    return dict(eta_pa_s=eta,statistical_se_pa_s=se,n_frames=int(mask.sum()),n_kept=int(keep.sum()),
                rms_ml=float(np.sqrt(np.mean((dv[keep]-cf[keep]/eta)**2))),
                receiver_gain_ml=float(dv[-1]),air_start_end_ml=next_air[mask][[0,-1]].tolist(),
                fit_times_s=t[mask][[0,-1]].tolist(),iteration_history=history), dict(
                    t=t,receiver_ml=receiver,forcing_ml_pa_s_per_s=forcing,
                    cumulative_source_forcing=released,cumulative_receiver_forcing=arrived,
                    air_ml=next_air,transit_s=cache['transit'],head_m=head,fitmask=mask,
                    fit_keep=keep,fit_delta_receiver_ml=dv,fit_cumulative_forcing=cf)


def main():
    obs=dict(np.load(BASE/'observations.npz'))
    baseline=json.loads((BASE/'identify.json').read_text())
    window=baseline['t_fit']
    cache=geometry_cache(obs,window,1.)
    result,arrays=solve(cache)
    alternate,_=solve(cache,initial_air_ml=5.)
    if abs(result['eta_pa_s']-alternate['eta_pa_s']) > .001:
        raise RuntimeError('Transport iteration depends on its starting inventory')
    midpoint=.5*(window[0]+window[1])
    sensitivity=[]
    for lo,hi in [(window[0],midpoint),(midpoint,window[1])]:
        row,_=solve(geometry_cache(obs,[lo,hi],1.))
        sensitivity.append(dict(window=[lo,hi],**row))
    paths=[Path(__file__),ROOT/'experiments/pour/pour_weakform_transport.py',
           ROOT/'experiments/pour/pour_weakform_identify.py',
           ROOT/'experiments/pour/pour_perception.py',
           ROOT/'src/warpmpm/geometry/measuring_cup.py',BASE/'observations.npz',BASE/'identify.json']
    result.update(method='Time-weak brink law with causal source-to-receiver transport and conserved flight inventory',
                  calibration_pour_count=1,identification_episode='09-04-60-2s',
                  measured_endpoints_used=[],other_recordings_used=[],viscosity_fitted_by_simulator=False,
                  baseline_eta_pa_s=baseline['eta'],fit_window_rule='Unchanged: tilt ACK +0.15s to return send -0.15s',
                  transport_model='Ballistic transit sqrt(2*fall/g); zero initial downward velocity assumption',
                  geometry_and_optical_readout_changed=False,
                  independent_half_window_fits=sensitivity,
                  alternate_initial_inventory_eta_pa_s=alternate['eta_pa_s'],
                  status='Candidate consistency correction; not validated or released for robot control',
                  input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'fit.npz',**arrays)
    (OUT/'identify.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['input_sha256','iteration_history']},indent=2))


if __name__ == '__main__':
    main()
