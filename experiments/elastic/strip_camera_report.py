"""Prediction and visual report for identification from stereo images and force."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import cv2
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image,ImageDraw

from experiments.elastic.block_drop_report import COLORS,INK,font
from experiments.elastic.block_drop_study import save,sha
from experiments.elastic.strip_camera_identify import reconstruct_tracks
from experiments.elastic.strip_camera_reconstruction import fit_balance


def read(path): return json.loads(Path(path).read_text())


def signals(case): return np.genfromtxt(case/'signals.csv',delimiter=',',names=True)


def predict(source,root):
    from experiments.elastic.strip_bending_study import record
    dest=root/'predictions'; dest.mkdir(exist_ok=False)
    shutil.copy2(source/'protocol.json',dest/'protocol.json')
    for i,k in enumerate(COLORS):
        E=read(root/f'fit_{k}/identification.json')['E_pa']
        record(dest,f'prediction_{k}',E,96,f'cuda:{i}',states=False)
        record(dest,f'fine_prediction_{k}',E,128,f'cuda:{i}',states=False)


def summarize(source,root):
    protocol=read(source/'protocol.json'); result=dict(materials={},reference=str(source.resolve()))
    for k in COLORS:
        fit=read(root/f'fit_{k}/identification.json'); p=read(root/f'inputs_{k}/known.json')
        tracks=np.load(root/f'fit_{k}/tracks.npz'); true=np.load(root/f'evaluation_{k}/true_marker_positions.npy')
        valid=tracks['valid']; error=tracks['world']-true
        motion=(tracks['world']-tracks['world'][0])-(true-true[0])
        row=dict(true_E_pa=protocol['E_pa'][k],identified_E_pa=fit['E_pa'],
                 signed_E_error_percent=100*(fit['E_pa']/protocol['E_pa'][k]-1),
                 tracking=read(root/f'fit_{k}/tracking.json'),fit=fit,
                 marker_absolute_rmse_mm=float(np.sqrt(np.mean(np.sum(error[valid]**2,axis=1)))*1000),
                 marker_motion_rmse_mm=float(np.sqrt(np.mean(np.sum(motion[valid]**2,axis=1)))*1000))
        # Evaluation only, after the camera fit is frozen. This identifies the
        # portion of error remaining with ideal surface-marker observations.
        ideal=dict(reference=true[0],world=true,valid=valid,time=tracks['time'])
        X,V,F,vol,_=reconstruct_tracks(ideal,p)
        sensor=np.genfromtxt(root/f'inputs_{k}/force.csv',delimiter=',',names=True)
        row['ideal_marker_diagnostic'],_,_=fit_balance(X,V,F,vol,tracks['time'],sensor['time'],sensor['material_on_pusher_Fx'],p)
        reconstructed=np.load(root/f'fit_{k}/reconstructed.npz'); biases={}
        for offset in [-.001,.001]:
            altered,_,_=fit_balance(reconstructed['x'],reconstructed['v'],reconstructed['F'],
                reconstructed['vol0'],reconstructed['time'],sensor['time'],sensor['material_on_pusher_Fx']+offset,p)
            biases[str(offset)]=altered['E_pa']
        row['force_bias_E_pa']=biases
        if (root/f'fit_50hz_{k}/identification.json').exists():
            row['fit_50hz']=read(root/f'fit_50hz_{k}/identification.json')
        for name,true_name,pred_name in [('main','truth','prediction'),('fine','fine','fine_prediction')]:
            ref=source/f'{true_name}_{k}'; pred=root/f'predictions/{pred_name}_{k}'
            comp=read(pred/'completion.json')
            assert comp['all_frames_completed'] and comp['inverted_count']==0
            assert read(pred/'config.json')['E_pa']==fit['E_pa']
            x=np.load(ref/'x.npy',mmap_mode='r'); xp=np.load(pred/'x.npy',mmap_mode='r')
            np.testing.assert_array_equal(x[0],xp[0])
            t=np.load(ref/'time.npy'); use=t>=protocol['withdraw_end']
            rms=np.array([np.sqrt(np.mean(np.sum((a.astype(float)-b)**2,axis=1)))*1000
                          for a,b in zip(x,xp,strict=True)])
            ref_s=signals(ref); pred_s=signals(pred)
            tip=np.column_stack([ref_s[n]-pred_s[n] for n in ['tip_x','tip_y','tip_z']])*1000
            force=np.column_stack([pred_s[n] for n in ['reaction_x','reaction_y','reaction_z']])
            max_force=float(np.linalg.norm(force[use],axis=1).max())
            assert max_force==0
            row[name]=dict(recovery_particle_rmse_mm=float(np.sqrt(np.mean(rms[use]**2))),
                           recovery_max_frame_rmse_mm=float(rms[use].max()),
                           recovery_tip_rmse_mm=float(np.sqrt(np.mean(np.sum(tip[use]**2,axis=1)))),
                           max_recovery_tool_force_n=max_force,min_J=comp['min_J'])
            np.savetxt(root/f'recovery_{name}_{k}.csv',np.column_stack([t,rms]),delimiter=',',
                       header='time,particle_rmse_mm',comments='')
        result['materials'][k]=row
    save(root/'report.json',result)
    return result


def tracked_frame(root,k,ci,index):
    im=np.load(root/f'inputs_{k}/camera_{ci}.npy',mmap_mode='r')[index]
    rgb=cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
    t=np.load(root/f'fit_{k}/tracks.npz'); color=tuple(int(COLORS[k][i:i+2],16) for i in [1,3,5])
    for x,y in t['pixels'][ci,index,t['valid'][index]]:
        cv2.circle(rgb,(round(x),round(y)),6,color,2,cv2.LINE_AA)
    # One fixed crop for every material, time and camera; no object rescaling.
    return Image.fromarray(rgb[335:1225,315:975])


def render(source,root,result):
    media=root/'media'; media.mkdir(exist_ok=True)
    times=np.load(root/'inputs_A/time.npy')
    force={k:np.genfromtxt(root/f'inputs_{k}/force.csv',delimiter=',',names=True) for k in COLORS}
    def video_frame(i):
        canvas=Image.new('RGB',(1280,1060),'white'); draw=ImageDraw.Draw(canvas)
        draw.text((28,16),'Identification from image tracks and force',font=font(29,True),fill=INK)
        draw.text((28,58),'Stereo experiment · left camera shown · 5× slow motion',font=font(18),fill='#5a6871')
        draw.text((1110,22),f'{times[i]:.2f} s',font=font(24),fill=INK)
        for j,k in enumerate(COLORS):
            im=tracked_frame(root,k,0,i).resize((610,823),Image.Resampling.LANCZOS)
            canvas.paste(im,(15+j*640,139))
            draw.text((29+j*640,102),f'Material {k}',font=font(24,True),fill=COLORS[k])
            f=float(np.interp(times[i],force[k]['time'],force[k]['material_on_pusher_Fx']))
            draw.text((29+j*640,975),f'Measured pusher force: {f:.3f} N',font=font(21),fill=INK)
        draw.text((28,1024),'Rendered images with applied dots; measured image coordinates feed the weak-form fit.',
                  font=font(16),fill='#5a6871')
        return np.asarray(canvas)
    with imageio.get_writer(media/'camera_tracking.mp4',fps=20,codec='libx264',quality=8,
                            macro_block_size=2,pixelformat='yuv420p') as writer:
        first=video_frame(0)
        for _ in range(10): writer.append_data(first)
        for i in range(len(times)): writer.append_data(video_frame(i))
        last=video_frame(len(times)-1)
        for _ in range(18): writer.append_data(last)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':INK,
                         'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':INK})
    fig=plt.figure(figsize=(14,8),facecolor='white')
    fig.text(.035,.948,'Identify stiffness from camera tracks and force',fontsize=20,weight='bold')
    fig.text(.035,.903,'Same bending motion · image-based reconstruction · recovery excluded from fitting',fontsize=12,color='#576772')
    fig.text(.035,.85,'(a) Stereo observations of material A',fontsize=13,weight='bold')
    for ci in range(2):
        ax=fig.add_axes([.027+ci*.228,.18,.223,.635])
        ax.imshow(tracked_frame(root,'A',ci,90)); ax.axis('off')
        ax.set_title(f'Camera {ci+1}',fontsize=11)
    fig.text(.035,.117,'145 of 147 markers remain tracked in each material.',fontsize=11)
    fig.text(.035,.084,'Interior deformation uses a plane-stress approximation.',fontsize=11,color='#576772')
    fig.text(.555,.85,'(b) Identified Young’s modulus',fontsize=13,weight='bold')
    for x,label in [(.625,'True'),(.755,'Camera + force'),(.91,'Error')]:
        fig.text(x,.786,label,fontsize=11,ha='center',color='#576772')
    for j,k in enumerate(COLORS):
        r=result['materials'][k]; y=.729-j*.060
        fig.text(.556,y,k,fontsize=14,weight='bold',color=COLORS[k])
        fig.text(.625,y,f"{r['true_E_pa']/1000:.0f} kPa",fontsize=13,ha='center')
        fig.text(.755,y,f"{r['identified_E_pa']/1000:.2f} kPa",fontsize=13,ha='center',color=COLORS[k],weight='bold')
        fig.text(.91,y,f"{r['signed_E_error_percent']:+.2f}%",fontsize=13,ha='center')
    fig.text(.555,.56,'(c) Recovery with identified stiffness',fontsize=13,weight='bold')
    ax=fig.add_axes([.60,.235,.345,.26])
    for k in COLORS:
        for parent,name,style in [(source,'truth','-'),(root/'predictions','prediction','--')]:
            d=signals(parent/f'{name}_{k}'); keep=d['time']>=1.17
            ax.plot(d['time'][keep]-1.17,(d['tip_x'][keep]-.12)*1000,style,color=COLORS[k],lw=1.7)
    ax.set(xlabel='Time after withdrawal (s)',ylabel='Tip displacement (mm)')
    ax.legend(handles=[Line2D([],[],color=INK,label='True simulation'),
                       Line2D([],[],color=INK,ls='--',label='Identified model')],frameon=False,
              ncol=2,fontsize=9,loc='lower left',bbox_to_anchor=(0,1.02),borderaxespad=0)
    fig.text(.60,.143,'Recovery 3D particle RMSE',fontsize=10,color='#576772')
    fig.text(.60,.102,'  ·  '.join(f"{k}: {result['materials'][k]['main']['recovery_particle_rmse_mm']:.2f} mm" for k in COLORS),
             fontsize=12,weight='bold')
    fig.text(.035,.029,'Synthetic cameras and force. Known geometry, density and ν = 0.45. No simulator F or velocities enter identification.',
             fontsize=10,color='#576772')
    fig.savefig(media/'figure_preview.png',dpi=160); fig.savefig(media/'figure_preview.pdf'); plt.close(fig)
    save(media/'provenance.json',dict(renderer_sha256=sha(Path(__file__)),report_sha256=sha(root/'report.json'),
         camera_crop_xyxy=[315,335,975,1225],same_crop_all_views=True,video_slowdown=5,
         observations_from_saved_pixels=True,figure_camera_time_s=.9))


def write_report(root,r):
    lines=['# Elastic identification from stereo images and force','',
           'The identifier now receives camera pixels and pusher force, not MPM particle states.',
           'The previous supplied-state experiment remains unchanged. No manuscript figure was replaced.','',
           '| Material | True E (kPa) | Camera E (kPa) | Signed error | Recovery particle RMSE (mm) |',
           '|---|---:|---:|---:|---:|']
    for k,v in r['materials'].items():
        lines.append(f"| {k} | {v['true_E_pa']/1000:.0f} | {v['identified_E_pa']/1000:.3f} | {v['signed_E_error_percent']:+.3f}% | {v['main']['recovery_particle_rmse_mm']:.3f} |")
    lines+=['','![Figure preview](media/figure_preview.png)','',
            '[Tracking video, 5× slow motion](media/camera_tracking.mp4)','',
            '## What is observed','',
            'The original 12 × 20 × 100 mm strips and material parameters are unchanged. A 20 mm diameter',
            'pusher performs the same 24 mm stroke on A and B. The upper 15 mm remain gripped.',
            'Two calibrated pinhole cameras render an applied grid of 147 dots on one side at',
            '1280 × 1280 and 100 Hz. Image noise is independent Gaussian grayscale noise, standard',
            'deviation 0.5 on an 8-bit scale. Calibration is exact, lighting controlled, and motion',
            'blur and lens distortion absent. This is a synthetic camera experiment, not hardware footage.','',
            'Pyramidal Lucas–Kanade tracking is checked in both directions and refined to detected',
            'dot centroids. Stereo triangulation yields surface coordinates. Lost tracks are',
            'excluded; no ground-truth coordinates replace them. CoTracker 3 is not used. The',
            'known initial dot pattern establishes identities; the reference positions themselves',
            'come from the initial stereo images.','',
            'The force channel is the actual simulated pusher reaction averaged over 2 ms.',
            'It is integrated at its own 500 Hz rate, rather than subsampled to camera times.',
            'The main result adds no force noise or calibration error. Force is measured, not controlled.','',
            '## Reconstruction and identification','',
            'A spatial displacement spline is fitted to measured surface-marker motion and known',
            'fixed-grip constraints. Its derivatives give in-plane deformation; temporal differentiation',
            'of a seven-frame cubic Savitzky–Golay fit gives velocity. Reconstruction sees loading',
            'images only, from 0 to 0.90 s. The weak balance uses 0.32–0.88 s.','',
            'Motion is assumed planar and uniform through specimen width. A plane-stress closure',
            'normal to the observed face supplies transverse stretch. For the fixed-corotated law,',
            'if d is the determinant of the observed in-plane deformation gradient and r = ν/(1−2ν),',
            'the transverse stretch is (1+r d)/(1+r d²). E cancels from this relation. This is an',
            'approximation of interior deformation, not a camera measurement of the full 3D state.',
            'Geometry, density, Poisson ratio 0.45 and constitutive family are known; only E is estimated.','',
            'The same nodal bending virtual fields as the reference experiment enter the weak',
            'balance. Their unit horizontal trace around the pusher uses its net measured force;',
            'their zero grip trace removes the unknown grip reaction. A scalar least-squares solve',
            'fits E. There is no stiffness lookup, forward-rollout fitting, or empirical correction factor.','',
            'The displacement basis has degree two across thickness and cubic splines with 8 mm',
            'knots along length. This shared setting was chosen from ten candidates using four-fold',
            'held-marker displacement prediction across loading frames of both materials. The criterion',
            'does not use E, simulator states, or recovery. Its slightly lower marker error produced',
            'worse parameter errors than the initial 10 mm setting (85.512/257.098 kPa); both results',
            'are retained. We did not choose the setting that happens to match true E best.','',
            '## What the accuracy does and does not show','',
            'The observation-only rerun blocks access to other experiment files and reproduces',
            'the exact tracks and fitted E for both materials. Simulator states are used only by',
            'the image generator and the subsequent evaluation. This closes the direct-state input',
            'gap for this controlled experiment; it does not establish general 3D observability.','',
            'The frozen model starts from the original undeformed specimen, replays the entire',
            'pusher trajectory, and predicts recovery from 1.17 to 2.00 s. It is never initialized',
            'from the true release state. Recovery is a later interval of the same episode, not an',
            'independent action or geometry-transfer experiment. Error uses all corresponding 3D',
            'particles, with no alignment or rescaling. All four prediction runs finish without',
            'inversions and with zero pusher contact after withdrawal.','']
    for k,v in r['materials'].items():
        lines.append(f"- {k}: marker position RMSE {v['marker_absolute_rmse_mm']:.4f} mm; marker-motion RMSE {v['marker_motion_rmse_mm']:.4f} mm. "
                     f"The same reconstruction with ideal surface markers gives E = {v['ideal_marker_diagnostic']['E_pa']/1000:.3f} kPa. "
                     f"Recovery particle RMSE at grid 128 is {v['fine']['recovery_particle_rmse_mm']:.3f} mm.")
        if 'fit_50hz' in v:
            lines.append(f"- {k}, rerunning tracking and fitting at 50 Hz: E = {v['fit_50hz']['E_pa']/1000:.3f} kPa. "
                         'The seven-frame temporal filter spans twice as much time in this check.')
        lines.append(f"- {k}, illustrative force-offset sensitivity (−1/+1 mN): E = "
                     f"{v['force_bias_E_pa']['-0.001']/1000:.3f}/{v['force_bias_E_pa']['0.001']/1000:.3f} kPa.")
    lines+=['','These diagnostics indicate a remaining reconstruction/modeling limitation, even',
            'before real calibration errors, occlusions, or sensor bias. The near-exact stiffness',
            'accuracy of the supplied-state reference does not carry over. Further improvements',
            'should address the interior closure and its strain sensitivity; this pilot should not',
            'be described as hardware validation.','',
            '## Development record','',
            'The first surface extrapolation gave 77.660/194.354 kPa after correcting force integration.',
            'Plane stress reduced its weak residuals. The diagnostic files preserve those outcomes.',
            'The first stereo rendering attempt reused a cached camera image; the initial marker',
            'projection check exposed the error. Those images are kept in development_render_cache',
            'and excluded. The corrected renderer explicitly renders every camera before capture.',
            'The initial camera reconstruction is retained in development_initial_reconstruction.','',
            '## Reproduce','',
            'Use the unchanged source archive out/strip_bending_20260912_clearance, or reproduce it',
            'using its own report. Install opencv-python-headless==4.11.0.86 into the project environment.',
            'From the repository root, use a fresh output directory and run these commands for A and B:','',
            '```bash',
            '.venv/bin/python -m experiments.elastic.strip_camera_observe --source out/strip_bending_20260912_clearance --out out/new_strip_camera --material A',
            '.venv/bin/python -m experiments.elastic.strip_camera_identify --inputs out/new_strip_camera/inputs_A --out out/new_strip_camera/fit_A',
            '```','',
            'After both materials:','',
            '```bash',
            '.venv/bin/python -m experiments.elastic.strip_camera_checks select --out out/new_strip_camera',
            '.venv/bin/python -m experiments.elastic.strip_camera_checks isolate --out out/new_strip_camera',
            '.venv/bin/python -m experiments.elastic.strip_camera_report predict --source out/strip_bending_20260912_clearance --out out/new_strip_camera',
            '.venv/bin/python -m experiments.elastic.strip_camera_report report --source out/strip_bending_20260912_clearance --out out/new_strip_camera',
            '```','',
            'Set OMP_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1 for the recorded CPU setup. Rendering',
            'or GPU versions may affect floating-point and rasterization details; versions and source',
            'snapshots are retained with this run. The figure uses a single fixed crop for every view.','',
            '## Pending work','',
            'The plastic-squeeze experiment still supplies elastic F_e and needs a separate correction',
            'based on observed loading/unloading and force. This is recorded in project memory.',
            'Insertion planning and manuscript edits are not part of this run.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['predict','report']); ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.stage=='predict': predict(args.source,args.out)
    else:
        result=summarize(args.source,args.out); write_report(args.out,result); render(args.source,args.out,result)
