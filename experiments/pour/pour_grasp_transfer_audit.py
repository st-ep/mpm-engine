"""External-outline check of a fixed grasp across the original recordings.

Only the 60-degree calibration sets geometry. The other images receive overlays;
no liquid level, endpoint, viscosity or geometry parameter is fitted here.
"""
from pathlib import Path
import json
import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from examples.pour_recorded_twin import RecordedPanda,load_episode,recorded_pour_actions,SPEC,quat_to_mat
from experiments.pour.pour_perception import Camera

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'out/pour_physics_audit'


def ring(z):
    theta=np.linspace(-np.pi,np.pi,801)
    a,b=SPEC.outer_semi_axes(z);x=a*np.cos(theta);y=b*np.sin(theta)
    x=x+np.where(x>0,SPEC.spout_dx(y,z),0.)
    return np.c_[x,y,np.full(len(x),z)]


def main():
    geom=json.loads((ROOT/'out/pour_weakform_recovery/identified/geometry.json').read_text())
    mesh=ROOT/'out/pour_wf/09-04-60-2s/cup_render.obj'
    ep=load_episode(ROOT/'pouring_real_data/09-04-60-2s',1.,2.5)
    arm=RecordedPanda(ep,mesh,height=64,width=64,max_geom=4000,
                      cup_reference_pos=geom['cup_reference_pos'],cup_reference_quat=geom['cup_reference_quat'])
    hc,g=arm._hand_cup.copy(),arm._grasp.copy();arm.close()
    fig,axes=plt.subplots(2,3,figsize=(13,9))
    for j,angle in enumerate([60,45,50]):
        path=ROOT/f'pouring_real_data/09-04-{angle}-2s'
        ep=load_episode(path,1.,2.5);pour,ret,_=recorded_pour_actions(path)
        camera=Camera(ep['meta'],'side')
        arm=RecordedPanda(ep,mesh,height=64,width=64,max_geom=4000)
        arm._hand_cup=hc.copy();arm._grasp=g.copy()
        rows=[json.loads(s) for s in (path/'frames_side.jsonl').read_text().splitlines()]
        reader=imageio.get_reader(path/'side_rgb.mp4')
        for i,t in enumerate([-.1, .5*(pour['t_ack']+ret['t_send'])-pour['t_send']]):
            r=min(rows,key=lambda r:abs(r['t_host']-pour['t_send']-t))
            actual_t=r['t_host']-pour['t_send'];rgb=np.asarray(reader.get_data(r['frame_idx']))
            pos,q=arm.cup_pose_at(actual_t+ep['t_pour']);rotation=quat_to_mat(q)
            ax=axes[i,j];ax.imshow(np.rot90(rgb));points=[]
            for z in [0.,SPEC.rim_z]:
                uv,_=camera.project(ring(z)@rotation.T+pos)
                xy=np.c_[uv[:,1],camera.w-1-uv[:,0]];points.append(xy)
                ax.plot(xy[:,0],xy[:,1],color='cyan',lw=.8,alpha=.8)
            p=np.concatenate(points)
            ax.set(xlim=(p[:,0].min()-12,p[:,0].max()+12),
                   ylim=(p[:,1].max()+12,p[:,1].min()-12),
                   title=f'{angle}° recording, {actual_t:.2f}s')
        arm.close();reader.close()
    fig.suptitle('External contours from one fixed 60° grasp calibration; no fitting to the other recordings')
    fig.tight_layout();fig.savefig(OUT/'grasp_transfer_overlays.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    main()
