"""Evaluate frozen hardware predictions and render labeled reality/MPM comparisons."""
import argparse
from pathlib import Path
import subprocess
import cv2
import imageio_ffmpeg
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from experiments.robotics.press_hardware_observe import read,save,jsonl
from experiments.robotics.press_hardware_model import stress_history

LABELS={'play_doh':'Play-Doh','butter_slime':'Butter slime','plasticine':'Plasticine'}
COLORS={'play_doh':'#bc2429','butter_slime':'#e5c833','plasticine':'#454442'}


def evaluate(out):
    p=read(out/'protocol.json');results={}
    fig,axes=plt.subplots(3,3,figsize=(14,10),constrained_layout=True)
    for i,m in enumerate(p['materials']):
        name=p['heldout'][m];obs=np.load(out/name/'motion.npz');geo=read(out/name/'geometry.json')
        folder=out/m/p['primary_prediction_tag'];sim=np.load(folder/'trajectory.npz');c=read(folder/'completion.json')
        f=sim['force_log'];shape=sim['shape_log'];t=shape[:,0];h=np.interp(t,f[:,0],f[:,3])
        measured_h=np.interp(t,obs['time'],obs['raw_height'])
        raw=obs['raw_shape_samples'];measured_width=2*np.interp(t,raw[:,0],raw[:,2])
        gap_error=(h-measured_h)*1000;width_error=(shape[:,1]-measured_width)*1000
        loading=(t>=.4)&(t<=10.4);unloading=(t>=11)&(t<=13.4)
        hold=(t>=3)&(t<=10.4)
        force_loading=(f[:,0]>=.4)&(f[:,0]<=10.4);force_hold=(f[:,0]>=3)&(f[:,0]<=10.4)
        selected=read(out/m/'fit.json')['selected'];search=read(out/m/'fit_search.json')
        near=[v for v in search if (v['tau_s']>0)==(selected['tau_s']>0) and v['relative_weak_residual']<=1.02*selected['relative_weak_residual']]
        spread={k:[float(min(v[k] for v in near)),float(max(v[k] for v in near))] for k in ['E_Pa','Y_Pa','tau_s']}
        perturbed=[]
        for factor in [.99,1.01,2.]:
            vectors=[]
            for train in p['train'][m]:
                d=np.load(out/train/'motion.npz');tg=read(out/train/'geometry.json')
                stress=stress_history(d['F'],selected['ratio']*factor,selected['tau_s'],p['nu'],p['export_dt_s'])
                response=tg['volume_m3']*np.einsum('tnij,tmnij,n->tm',stress,d['test_gradient'],d['weights'])
                response=d['windows']@response;keep=(d['window_begin']<10)|(d['window_begin']>=10.8)
                vectors.append((response[keep]*selected['E_Pa']/np.linalg.norm(d['target'][keep])).ravel())
            perturbed.append(np.concatenate(vectors))
        yield_sensitivity=float(np.linalg.norm((perturbed[1]-perturbed[0])/.02)/np.sqrt(3))
        yield_doubling=float(np.linalg.norm(perturbed[2]-(perturbed[0]+perturbed[1])/2)/np.sqrt(3))
        yield_resolved=spread['Y_Pa'][1]/max(spread['Y_Pa'][0],1e-12)<3 and selected['ratio']>1.1e-5 and yield_sensitivity>.1
        entry=dict(material=m,episode=name,law=selected['law'],G_Pa=c['shear_modulus_Pa'],
            E_equivalent_Pa=c['equivalent_E_Pa'],Y_numeric_Pa=selected['Y_Pa'],tau_s=selected['tau_s'],
            eta_Pa_s=selected['eta_Pa_s'],bulk_assumed_Pa=c['bulk_modulus_assumed_Pa'],nu_equivalent=c['equivalent_nu'],
            yield_resolved_in_sampled_search=yield_resolved,sampled_near_fit_spread=spread,
            yield_weak_log_sensitivity=yield_sensitivity,yield_doubling_relative_response=yield_doubling,
            yield_status='unresolved: locally inactive in weak balance' if yield_sensitivity<1e-6 else ('weakly constrained' if not yield_resolved else 'conditional fitted candidate'),
            training_weak_relative_residual=selected['relative_weak_residual'],
            loading_gap_rmse_mm=float(np.sqrt(np.mean(gap_error[loading]**2))),
            loading_width_rmse_mm=float(np.sqrt(np.mean(width_error[loading]**2))),
            hold_gap_rmse_mm=float(np.sqrt(np.mean(gap_error[hold]**2))),
            hold_width_rmse_mm=float(np.sqrt(np.mean(width_error[hold]**2))),
            unloading_gap_rmse_mm=float(np.sqrt(np.mean(gap_error[unloading]**2))),
            final_loading_gap_error_mm=float(np.interp(10,t,gap_error)),
            force_loading_relative_l2=float(np.linalg.norm((f[:,2]-f[:,1])[force_loading])/np.linalg.norm(f[force_loading,1])),
            force_hold_relative_l2=float(np.linalg.norm((f[:,2]-f[:,1])[force_hold])/np.linalg.norm(f[force_hold,1])),
            min_mean_volume_ratio=float(shape[:,3].min()),max_mean_volume_ratio=float(shape[:,3].max()),
            min_particle_elastic_J=c['min_elastic_J'],
            initial_height_m=geo['h0_m'],initial_volume_m3=geo['volume_m3'],
            geometry_scope='Gap comparison uses recorded EEF motion with an assumed initial-height offset. Width is inferred from visible silhouette pixels and supplied depth scale.',
            force_scope='Force tracking verifies prescribed loading, not an independent material prediction.',
            uncertainty_scope='Near-fit spread is sampled objective sensitivity, not a confidence interval.')
        screen=p['diagnostic_screen'];entry['passes_primary_screen']=bool(entry['loading_gap_rmse_mm']<=screen['gap_rmse_mm'] and entry['loading_width_rmse_mm']<=screen['width_rmse_mm'] and entry['force_hold_relative_l2']<=screen['late_hold_force_relative_l2'])
        variations=[]
        for directory in sorted((out/m).glob('check_*')):
            if not (directory/'completion.json').exists():continue
            v=np.load(directory/'trajectory.npz');vf=v['force_log'];vc=read(directory/'completion.json')
            same=np.interp(t,vf[:,0],vf[:,3]);diff=(same-h)*1000
            variations.append(dict(name=directory.name,grid=vc['grid'],friction=vc['friction_assumed'],bulk=vc['bulk_modulus_assumed_Pa'],tick=vc['tick'],
                gap_difference_rms_mm=float(np.sqrt(np.mean(diff[loading]**2))),gap_difference_at_10s_mm=float(np.interp(10,t,diff))))
        entry['forward_sensitivity']=variations;results[m]=entry
        axes[i,0].plot(obs['time'],obs['raw_height']*1000,label='Recorded motion + initial height')
        axes[i,0].plot(f[:,0],f[:,3]*1000,label='Forward MPM')
        axes[i,1].plot(raw[:,0],raw[:,2]*2000,label='RGB silhouette estimate')
        axes[i,1].plot(t,shape[:,1]*1000,label='Forward MPM')
        axes[i,2].plot(f[:,0],f[:,1],label='Recorded force input')
        axes[i,2].plot(f[:,0],f[:,2],label='Simulated reaction',alpha=.8)
        axes[i,0].set(title=f'{LABELS[m]}: plate gap',ylabel='Gap (mm)')
        axes[i,1].set(title='Middle width: geometry estimate',ylabel='Width (mm)')
        axes[i,2].set(title='Input-force tracking, not model validation',ylabel='Force (N)')
        for ax in axes[i]:
            ax.set_xlim(0,13.4);ax.grid(alpha=.2);ax.set_xlabel('Time from baseline (s)');ax.legend(fontsize=7)
            ax.axvspan(10.8,13.4,color='gray',alpha=.08)
        print(m,{k:entry[k] for k in ['loading_gap_rmse_mm','loading_width_rmse_mm','unloading_gap_rmse_mm','yield_resolved_in_sampled_search','passes_primary_screen']},flush=True)
    fig.savefig(out/'validation_curves.png',dpi=150);plt.close(fig)
    save(out/'validation_summary.json',dict(materials=results,scope='Held-out 21 N presses; frozen fitted parameters; conditional geometry and contact',
        contact='Coulomb friction varied, adhesion unmeasured and omitted. Low-load comparison stops before physical retraction.',
        primary_screen=p['diagnostic_screen']))


class Video:
    def __init__(self,path,size,fps=20):
        self.path=path
        self.proc=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo',
            '-pix_fmt','bgr24','-s',f'{size[0]}x{size[1]}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast',
            '-crf','20','-pix_fmt','yuv420p','-threads','1','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    def write(self,frame):self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    def close(self):
        self.proc.stdin.close();code=self.proc.wait();assert code==0,(self.path,code)


def text(im,message,xy,size=.55,color=(35,35,35),thickness=1):
    cv2.putText(im,message,xy,cv2.FONT_HERSHEY_SIMPLEX,size,color,thickness,cv2.LINE_AA)


def mesh(points):
    import pyvista as pv
    hull=ConvexHull(points)
    return pv.PolyData(points,np.c_[np.full(len(hull.simplices),3),hull.simplices].ravel()).clean().compute_normals(auto_orient_normals=True,consistent_normals=True)


def render(out):
    import pyvista as pv
    p=read(out/'protocol.json');media=out/'media';media.mkdir(exist_ok=True)
    fps=20;width,height=512,320;header=36;panel_h=height+header
    times=np.arange(0,p['comparison_duration_s']+.0001,1/fps)
    plots=[];records=[];writers=[]
    for m in p['materials']:
        name=p['heldout'][m];geo=read(out/name/'geometry.json');a=read(Path(p['observations'])/name/'assessment.json')
        sim=np.load(out/m/p['primary_prediction_tag']/'trajectory.npz')
        logs=jsonl(Path(p['source'])/name/'frames_hand.jsonl');hosts=np.array([f['t_host'] for f in logs])
        capture=cv2.VideoCapture(str(Path(p['source'])/name/'hand_rgb.mp4'))
        pl=pv.Plotter(off_screen=True,window_size=(width,height));pl.set_background('#f2f3f4')
        pl.add_mesh(pv.Box(bounds=(.02,.16,.02,.16,.045,.05)),color='#9aa5ab')
        pl.add_mesh(mesh(sim['x'][0]),name='specimen',color=COLORS[m],smooth_shading=True,specular=.15,ambient=.3)
        pl.add_mesh(pv.Cylinder(center=(.09,.09,.05+geo['h0_m']+.003),direction=(0,0,1),radius=.055,height=.006,resolution=80),name='plate',color='#ece9df',smooth_shading=True)
        distance=geo['camera_depth_m'];pl.camera.position=(.09,.09-distance,.056)
        pl.camera.focal_point=(.09,.09,.077);pl.camera.up=(0,0,1)
        # Same nominal metric scale as the RGB crop; no per-frame alignment or scaling.
        scale=width/224;focal=distance/geo['pixel_scale_m']*scale
        pl.camera.view_angle=float(np.degrees(2*np.arctan(height/(2*focal))))
        pl.reset_camera_clipping_range();plots.append(pl)
        records.append((m,sim,hosts,capture,a['t0_host']))
        writers.append((Video(media/f'{m}_reality.mp4',(width,panel_h)),Video(media/f'{m}_simulation.mp4',(width,panel_h))))
    combined=Video(media/'pressing_reality_vs_simulation.mp4',(width*2,panel_h*3+84))
    for i,t in enumerate(times):
        canvas=np.full((panel_h*3+84,width*2,3),245,np.uint8)
        text(canvas,p.get('video_header','WITHHELD 21 N PRESSES | REALITY AND FORCE-DRIVEN 3D MPM'),(16,25),.65,thickness=2)
        text(canvas,p.get('video_subtitle','Conditional models: geometry, bulk stiffness and contact remain assumptions.'),(16,49),.5)
        for row,((m,sim,hosts,cap,t0),pl,pair) in enumerate(zip(records,plots,writers)):
            frame=int(np.argmin(abs(hosts-(t0+t))));cap.set(1,frame);ok,raw=cap.read();assert ok
            real=cv2.resize(raw[140:280,348:572],(width,height),interpolation=cv2.INTER_AREA)
            j=int(np.argmin(abs(sim['time']-t)));points=sim['x'][j];gap=float(np.interp(t,sim['force_log'][:,0],sim['force_log'][:,3]))
            pl.add_mesh(mesh(points),name='specimen',color=COLORS[m],smooth_shading=True,specular=.15,ambient=.3,reset_camera=False)
            pl.add_mesh(pv.Cylinder(center=(.09,.09,.05+gap+.003),direction=(0,0,1),radius=.055,height=.006,resolution=80),
                name='plate',color='#ece9df',smooth_shading=True,reset_camera=False)
            pl.render();generated=cv2.cvtColor(pl.screenshot(return_img=True),cv2.COLOR_RGB2BGR)
            left=np.full((panel_h,width,3),245,np.uint8);right=left.copy();left[header:]=real;right[header:]=generated
            text(left,f'{LABELS[m]} | recording | {t:4.2f} s',(10,24),.55)
            text(right,f'{LABELS[m]} | MPM prediction | {t:4.2f} s',(10,24),.55)
            pair[0].write(left);pair[1].write(right)
            y=60+row*panel_h;canvas[y:y+panel_h,:width]=left;canvas[y:y+panel_h,width:]=right
        text(canvas,'Force is prescribed; displacement and shape are predictions. Mismatches are retained.',(16,canvas.shape[0]-8),.46)
        combined.write(canvas)
        if i in [0,int(5*fps),int(10*fps),len(times)-1]:cv2.imwrite(str(media/f'comparison_{t:05.2f}s.jpg'),canvas)
        if i%40==0:print('rendered',i,'/',len(times),flush=True)
    combined.close()
    for pair,pl,record in zip(writers,plots,records):
        pair[0].close();pair[1].close();pl.close();record[3].release()
    save(media/'render_provenance.json',dict(fps=fps,frames=len(times),real_camera='hand',real_crop_xyxy=[348,140,572,280],
        simulated_geometry='Convex hull of sampled actual MPM particle positions for display; quantitative metrics use full particle ensemble',
        camera='Fixed perspective view using nominal depth/pixel scale; illustrative viewpoint, not validated photometric camera registration',
        geometric_warp_of_recordings=False,per_frame_alignment=False,force_input='Held-out measured normal load',
        videos=[str(v) for v in sorted(media.glob('*.mp4'))]))


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('stage',choices=['evaluate','render']);a.add_argument('--out',type=Path,required=True)
    args=a.parse_args();evaluate(args.out) if args.stage=='evaluate' else render(args.out)
