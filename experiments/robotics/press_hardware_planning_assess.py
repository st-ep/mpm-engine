"""Audit prediction phases against fixed observed height/diameter references."""
import argparse
from pathlib import Path
import numpy as np
from experiments.robotics.press_hardware_observe import read, save


def assess(run, observations):
    c=read(run/'completion.json');d=np.load(run/'trajectory.npz')
    o=np.load(observations/c['episode']/'motion.npz')
    s=d['shape_log'];f=d['force_log'];t=s[:,0]
    h=np.interp(t,f[:,0],f[:,3]);ho=np.interp(t,o['time'],o['raw_height'])
    diameter=s[:,1];do=np.interp(t,o['time'],o['observed_max_width'])
    result=dict(run=str(run.resolve()),episode=c['episode'],material=c['material'],grid=c['grid'],dt_s=c['dt'],
        duration_s=float(t[-1]),loading_and_partial_unloading_complete=bool(t[-1]>=13.39),parameters=c['parameters'],
        height_role='Prescribed boundary input, not an independent validation metric' if c.get('recorded_displacement_used_as_control',False) else 'Prediction',
        min_elastic_J=c['min_elastic_J'],pad_normal=c.get('pad_normal',[0,0,1]),
        parameter_scope=c['parameters'].get('parameter_scope','Weak-form candidate; consult fit provenance'),phases={})
    for name,lo,hi in [('loading',.4,10.4),('initial',.4,1.5),('hold',3.,10.4),('unloading',11.,13.4)]:
        mask=(t>=lo)&(t<=hi)
        if not mask.any():continue
        fm=(f[:,0]>=lo)&(f[:,0]<=hi)
        result['phases'][name]=dict(gap_rmse_mm=float(np.sqrt(np.mean((h[mask]-ho[mask])**2))*1000),
            diameter_rmse_mm=float(np.sqrt(np.mean((diameter[mask]-do[mask])**2))*1000),
            gap_bias_mm=float(np.mean(h[mask]-ho[mask])*1000),diameter_bias_mm=float(np.mean(diameter[mask]-do[mask])*1000),
            normal_force_rmse_N=float(np.sqrt(np.mean((f[fm,2]-f[fm,1])**2))),
            normal_force_relative_l2=float(np.linalg.norm(f[fm,2]-f[fm,1])/max(np.linalg.norm(f[fm,1]),1e-12)))
    result['samples']={}
    for tt in [1.,3.,10.,13.]:
        if tt>t[-1]:continue
        result['samples'][str(tt)]=dict(predicted_gap_mm=float(np.interp(tt,t,h)*1000),reference_gap_mm=float(np.interp(tt,t,ho)*1000),
            predicted_diameter_mm=float(np.interp(tt,t,diameter)*1000),reference_diameter_mm=float(np.interp(tt,t,do)*1000))
    result['scope']='Numerical/experimental diagnostic; no automatic planning-readiness approval. Diameter compares full-particle radial 99th percentile with camera-derived visible profile width. Plate-gap reference is nominal initial gap plus recorded vertical EEF displacement. The unloading phase here still has residual contact load; free release at 15 s is assessed separately in image space.'
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--observations',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();results=[]
    for file in sorted(a.root.glob('**/completion.json')):
        completion=read(file)
        if (file.parent/'trajectory.npz').exists() and (a.observations/completion['episode']/'motion.npz').exists():results.append(assess(file.parent,a.observations))
    save(a.out,dict(results=results,selection_scope='Includes rejected and incomplete-duration diagnostic candidates; source paths distinguish studies.'))
    print(len(results),'runs assessed')
