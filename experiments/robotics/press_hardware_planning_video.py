"""Compare a force-driven prediction with RGB using the initial calibrated view."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from experiments.robotics.press_hardware_geometry import configure_camera
from experiments.robotics.press_hardware_observe import read, save, jsonl
from experiments.robotics.press_hardware_report import Video, mesh, text, COLORS, LABELS


def render(run, observations, out, snapshots_only=False, raw_root=None):
    import pyvista as pv
    c=read(run/'completion.json');ep=c['episode'];material=c['material']
    camera=read(observations/ep/'camera.json')
    raw=(raw_root or Path(__file__).resolve().parents[2]/'press_real_data')/ep
    assessment=observations/ep/'assessment.json'
    if not assessment.exists():assessment=Path(__file__).resolve().parents[2]/'out/press_observation_assessment_20260913'/ep/'assessment.json'
    a=read(assessment)
    d=np.load(run/'trajectory.npz')
    if len(d['x'])!=len(d['time']):raise ValueError('Saved particle frames required')
    hosts=np.array([f['t_host'] for f in jsonl(raw/'frames_hand.jsonl')])
    cap=cv2.VideoCapture(str(raw/'hand_rgb.mp4'));out.mkdir(exist_ok=True,parents=True)
    pl=pv.Plotter(off_screen=True,window_size=(848,480));pl.set_background('#d8dee0')
    pl.add_mesh(pv.Plane(center=(.09,.09,.05),direction=(0,0,1),i_size=.3,j_size=.3),color='#8a9899')
    configure_camera(pl,camera)
    size=(600,360);full=(600,420);crop=(320,130,620,310)
    writers=[] if snapshots_only else [Video(out/name,(1200,420) if name=='comparison.mp4' else full) for name in ['reality.mp4','simulation.mp4','comparison.mp4']]
    snapshot_times=[t for t in [0,3,10,13.4,15] if t<=d['time'][-1]+.01]
    indices=[int(np.argmin(abs(d['time']-t))) for t in snapshot_times] if snapshots_only else range(len(d['time']))
    for j in indices:
        t=float(d['time'][j]);idx=int(np.argmin(abs(hosts-a['t0_host']-t)))
        cap.set(1,idx);ok,rgb=cap.read()
        if not ok:raise RuntimeError((ep,idx))
        gap=float(np.interp(t,d['force_log'][:,0],d['force_log'][:,3]))
        pl.add_mesh(mesh(d['x'][j]),name='specimen',color=COLORS[material],smooth_shading=True,ambient=.3,reset_camera=False)
        normal=np.array(c.get('pad_normal',[0.,0.,1.]))
        pl.add_mesh(pv.Cylinder(center=(.09,.09,.05+gap+.003/normal[2]),direction=normal,radius=c['plate_radius_m'],height=.006,resolution=120),name='plate',color='#eeeeee',reset_camera=False)
        pl.render();im=cv2.cvtColor(pl.screenshot(return_img=True),cv2.COLOR_RGB2BGR)
        panels=[]
        simulation_label='Prescribed plate motion' if c.get('recorded_displacement_used_as_control',False) else 'Force-driven simulation'
        if c.get('withdrawal_start_s') is not None and t>c['withdrawal_start_s']:
            simulation_label='Withdrawal / recovery'
        for frame,label in [(rgb,'Recording'),(im,simulation_label)]:
            panel=np.full((420,600,3),245,np.uint8)
            panel[38:398]=cv2.resize(frame[crop[1]:crop[3],crop[0]:crop[2]],size)
            text(panel,f'{LABELS[material]} | {label} | {t:.2f} s',(10,26),.57,thickness=1)
            text(panel,'45 mm initial ball | 100 mm pressing pad',(10,414),.45)
            panels.append(panel)
        combined=np.concatenate(panels,axis=1)
        if writers:
            for writer,frame in zip(writers,[*panels,combined]):writer.write(frame)
        if any(abs(t-target)<.01 for target in snapshot_times):cv2.imwrite(str(out/f'comparison_{t:05.2f}s.jpg'),combined)
        if j%50==0:print('rendered',j,len(d['time']),flush=True)
    for writer in writers:writer.close()
    cap.release();pl.close()
    save(out/'provenance.json',dict(run=str(run.resolve()),episode=ep,camera=camera,crop_xyxy=crop,geometric_warp=False,per_frame_alignment=False,
        rendering='Convex hull of saved actual particle centers; finite particle support is not added. Same fixed initial-camera projection and image crop for real and simulation.',
        timing='Recorded host timestamps relative to force baseline; no time warping',plate_thickness_m_assumed=.006,
        parameter_scope=c['parameters'].get('parameter_scope','Weak-form fitted candidate; see source fit and numerical audit'),snapshots_only=snapshots_only))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['run','observations','out']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--raw-root',type=Path)
    parser.add_argument('--snapshots-only',action='store_true');render(**vars(parser.parse_args()))
