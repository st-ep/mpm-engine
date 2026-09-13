"""Presentation of the frozen plate pilot, with failures retained in the report."""
import argparse
import json
from pathlib import Path
import subprocess
import hashlib
import platform
import sys
from importlib.metadata import distributions
import numpy as np
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.elastic.block_drop_report import font

INK='#24313c';MUTED='#596773';BLUE='#237eae';ORANGE='#ce733b'


def panel(seq,i,width=460):
    # Identical camera crop and scale for both materials and every time.
    return Image.fromarray(seq[i]).convert('RGB').crop((0,300,1280,1210)).resize((width,round(width*910/1280)),Image.Resampling.LANCZOS)


def media(root):
    out=root/'media';out.mkdir(exist_ok=True)
    seq={k:np.load(root/f'inputs_{k}/camera_0.npy',mmap_mode='r') for k in 'AB'}
    ev=json.loads((root/'evaluation.json').read_text());truth={k:np.genfromtxt(root/f'truth_{k}/force.csv',delimiter=',',names=True) for k in 'AB'}
    im=Image.new('RGB',(1560,1170),'white');d=ImageDraw.Draw(im)
    d.text((45,26),'Plastic identification from stereo images + force',font=font(43,True),fill=INK)
    d.text((45,86),'One common plate motion. Same material parameters as the shaping study.',font=font(27),fill=MUTED)
    for j,label in enumerate(['Initial','Maximum compression','After unloading']):
        d.text((110+480*j,145),label,font=font(28,True),fill=INK)
    for row,k in enumerate('AB'):
        y=187+350*row;d.text((42,y+125),k,font=font(40,True),fill=BLUE if k=='A' else ORANGE)
        for j,i in enumerate([0,100,140]):im.paste(panel(seq[k],i),(100+480*j,y))
    d.line((45,895,1515,895),fill='#dce2e5',width=2)
    d.text((45,919),'Parameter estimates',font=font(30,True),fill=INK)
    d.text((625,921),'Young’s modulus',font=font(27,True),fill=INK);d.text((1105,921),'Yield parameter',font=font(27,True),fill=INK)
    for row,k in enumerate('AB'):
        e=ev[k];y=970+54*row;color=BLUE if k=='A' else ORANGE
        d.text((48,y),f'Material {k}',font=font(28,True),fill=color)
        d.text((625,y),f"{e['E_pa']/1000:.2f} kPa  ({e['E_error_percent']:+.2f}%)",font=font(28),fill=INK)
        d.text((1105,y),f"{e['yield_pa']/1000:.3f} kPa  ({e['yield_error_percent']:+.2f}%)",font=font(28),fill=INK)
    d.text((45,1100),'A’s stiffness and both yield parameters recover well; B’s stiffness remains underestimated.',font=font(27),fill=MUTED)
    im.save(out/'identification_preview.png')
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axs=plt.subplots(1,2,figsize=(12,3.6),layout='constrained')
    for ax,k,color in zip(axs,'AB',[BLUE,ORANGE],strict=True):
        f=truth[k];p=np.genfromtxt(root/f'prediction_{k}/force.csv',delimiter=',',names=True)
        ax.axvspan(0,1.4,color='#e8edf0',alpha=.8);ax.plot(f['time'],f['Fz'],color=color,label='True material',lw=2)
        ax.plot(p['time'],p['Fz'],color=INK,ls='--',lw=1.6,label='Identified model')
        ax.set(xlabel='Time (s)',ylabel='Plate force (N)',xlim=(0,2.9),ylim=(-.02*f['Fz'].max(),1.18*f['Fz'].max()),title=f'Material {k}')
        ax.text(.04,.91,'Identification',transform=ax.transAxes,color=MUTED,fontsize=10)
        ax.text(.60,.91,'Withheld reload',transform=ax.transAxes,color=MUTED,fontsize=10)
    fig.legend(*axs[0].get_legend_handles_labels(),loc='outside upper center',ncol=2,fontsize=10,frameon=False)
    fig.savefig(out/'force_prediction.png',dpi=180);plt.close(fig)
    # Slowed first press from the actual camera inputs, without inferred points
    # or simulator surface overlays. Labels and curves are presentation only.
    width,height=1440,900
    proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r','30','-i','-','-an','-c:v','libx264','-crf','19','-pix_fmt','yuv420p',str(out/'plate_identification.mp4')],stdin=subprocess.PIPE)
    indices=[0]*24+[min(140,round(i*.25/30/.01)) for i in range(169)]+[140]*36
    for i in indices:
        canvas=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(canvas);t=i*.01
        draw.text((35,20),'One press, two plastic responses',font=font(38,True),fill=INK)
        phase='Approach' if t<.2 else 'Compression' if t<=1 else 'Unload and recover'
        draw.text((35,72),f'{phase}   |   t = {t:.2f} s   |   0.25× speed',font=font(25),fill=MUTED)
        for j,k in enumerate('AB'):
            left=35+j*710;color=BLUE if k=='A' else ORANGE
            draw.text((left,120),f'Material {k}',font=font(29,True),fill=color)
            canvas.paste(panel(seq[k],i,670),(left,162))
            f=truth[k];F=float(np.interp(t,f['time'],f['Fz']))
            draw.text((left,652),f'Plate force  {F:.2f} N',font=font(27),fill=color)
            # Shared 0–18 N scale makes the physical force difference explicit.
            x0=left+30;y0=808;ww=600;hh=105
            draw.line((x0,y0,x0+ww,y0),fill='#b4bec6',width=2)
            use=f['time']<=t;points=[(x0+tt/1.4*ww,y0-ff/18*hh) for tt,ff in zip(f['time'][use],f['Fz'][use])]
            if len(points)>1:draw.line(points,fill=color,width=3)
            draw.text((x0,y0+8),'0 s',font=font(19),fill=MUTED);draw.text((x0+ww-50,y0+8),'1.4 s',font=font(19),fill=MUTED)
        draw.text((35,860),'Synthetic calibrated stereo + applied texture. No simulator motion or elastic state supplied to identification.',font=font(22),fill=MUTED)
        proc.stdin.write(np.asarray(canvas).tobytes())
    proc.stdin.close()
    if proc.wait():raise RuntimeError('Video encoding failed')


def write_report(root):
    e=json.loads((root/'evaluation.json').read_text());rows=[];prediction=[]
    for k in 'AB':
        a=e[k];rows.append(f"| {k} | 80.00 | {a['E_pa']/1000:.5f} | {a['E_error_percent']:+.5f}% | {1 if k=='A' else 10:.3f} | {a['yield_pa']/1000:.6f} | {a['yield_error_percent']:+.5f}% |")
        prediction.append(f"| {k} | {a['held_out_force_relative_l2']*100:.3f}% | {a['held_out_particle_rmse_mm']:.4f} | {a['fine_held_out_force_relative_l2']*100:.3f}% | {a['fine_held_out_particle_rmse_mm']:.4f} |")
    text='''# Observable plastic identification: plate pilot

The experiment works as an image-and-force pipeline, but is not yet an accurate
replacement for the central paper example. A's stiffness and both yield parameters
recover well. B's stiffness is underestimated by 23.79%. No paper figure or shaping
result has been replaced.

![Identification](media/identification_preview.png)

[Watch the first press](media/plate_identification.mp4). The video shows actual
synthetic camera inputs at quarter speed. Both materials have the same camera,
image scale, initial geometry, texture and prescribed plate motion.

## Result

| Material | True E (kPa) | Estimated E (kPa) | E error | True yield (kPa) | Estimated yield (kPa) | Yield error |
|---|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

Signed parameter errors are relative to the known simulation values. Yield uses
the engine's convention: the Frobenius norm of deviatoric Kirchhoff stress is
capped at Y. These are not directly standard equivalent Cauchy yield stresses.
Material properties were not changed to improve identification.

## What was measured and fitted

We press a 30 x 30 x 25 mm block against a flat support, then unload it. The
plate opening falls from 26 to 16 mm during the first loading and rises to 27 mm.
Both A and B undergo permanent deformation. The first 1.4 seconds identify E and
Y. A later press to 13 mm and final release are withheld.

The identifier reads two calibrated grayscale image sequences at 100 Hz, normal
plate force and plate opening at 500 Hz, timestamps and disclosed metadata.
OpenCV image corners, Lucas-Kanade initialization and affine reference-image
digital image correlation track the applied texture. Stereo triangulation gives
surface displacement. This is not CoTracker and not hardware video. Exact
camera calibration, low image noise (0.5 gray level), noiseless contact-force
readout and controlled lighting make this an idealized observation test.

Surface displacement is interpolated into a low-order square-symmetric 3D field.
Its spatial and temporal derivatives supply total deformation and velocity.
For each candidate Y/E, a local constitutive update infers elastic and plastic
history along that fixed measured motion. Divergence-free virtual fields eliminate
the isotropic stress contribution from the weak momentum balance. E is then
solved linearly; a scalar search selects Y/E. The final fit used 43 candidate
evaluations per material and zero forward MPM rollouts. Its roughly 11 seconds
of constitutive fitting exclude image rendering, tracking and reconstruction.
The complete parameter fit is nonlinear, not one convex solve.

An independent repeat was restricted to image/force bundles. Access to other
experiment outputs was blocked, and the guard was tested. It reproduced exactly
the tracks and both final estimates. Simulator positions and elastic states are
stored separately for rendering and evaluation, never supplied to this fit.

## Assumptions that remain

| Assumption | Why it is needed / what was checked |
|---|---|
| Known homogeneous isotropic Hencky elasticity with perfect von Mises plasticity, no hardening or viscosity | We estimate E and Y within this family. We do not discover an unknown constitutive family. |
| Known Poisson ratio 0.30, density 1000 kg/m3, initial geometry and stress-free reference | These are not jointly identified. |
| Frictionless plates, centered loading, initially square specimen | They justify zero tangential boundary work and reflection/quarter-turn symmetry. The finite-friction case has not been validated. |
| Low-order interior deformation inferred from one observed side and square symmetry | This replaces uniform compression but does not remove the surface-to-volume ambiguity. The basis resolves variation along height and quadratic/cubic cross-sectional variation; unobserved interior modes are omitted. |
| Accurate calibrated texture tracking | Motion was recovered from images. Evaluation-only surface-motion RMS errors are 0.01116 mm (A) and 0.01073 mm (B). |

No uniform compression, plane stress, plane strain or incompressibility assumption
is used in the final fit. In particular, divergence-free **test fields** do not
require divergence-free **material motion**. Relative to the elastic strip pilot,
the central added issue is reconstructing a three-dimensional plastic history
inside a thick specimen. This is a substantive approximation, not a detail we
can omit from the paper.

The shared spline spacing (6.25 mm) was selected by held-feature displacement
prediction, with positive reconstructed volume required. No true material
parameters entered this selection. The polynomial virtual fields of degrees
1, 2 and 3 use one common construction for both materials.

## Attempts that did not solve the problem

Uniform compression plus pressure-sensitive balance estimated E at 5.86 kPa for
A and 60.06 kPa for B. With ideal surface positions it still failed. The visible
deformation is not uniform, even with frictionless contacts.

The richer reconstruction reduced surface displacement residuals substantially,
but retaining pressure-sensitive equations remained poor (A reached the 5 kPa
lower search bound; B was 48.93 kPa). Volume changes inside the specimen are not
well constrained by one visible surface.

A single linear divergence-free test improved the richer-field estimates to
78.11/63.42 kPa. The final three-field construction gives the values above;
it was not selected as the variant with the smallest ground-truth parameter
error. Uniform reconstruction with that single pressure-free test gave
66.35/67.21 kPa. These diagnostics are retained in the output directory.

Replacing image tracks with ideal surface positions in the final reconstruction,
strictly as an evaluation diagnostic, gives E = 77.34/60.63 kPa. Thus better
tracking alone is unlikely to fix B. The remaining bias involves interior
reconstruction, weak discretization and simulation/contact discretization;
this test does not fully separate their contributions.

## Prediction on the withheld second press

The identified models are simulated from the initial state through both presses,
without resetting to a true deformed state. No data after 1.4 s are fitted.
Errors below cover 1.5–2.9 s, including unloading and noncontact intervals.

| Material | Force relative L2, grid 128 | Particle RMS (mm), grid 128 | Force relative L2, grid 160 | Particle RMS (mm), grid 160 |
|---|---:|---:|---:|---:|
'''+ '\n'.join(prediction)+'''

![Force prediction](media/force_prediction.png)

The geometry predictions are useful despite B's parameter bias. Particle RMS is
a full-volume matched-particle evaluation metric available in simulation, not a
camera measurement or the paper's surface shape error. The finer-grid comparisons
use the frozen estimates, not refitted parameters. All truth and prediction runs
completed without inversions. This is a withheld continuation with a deeper press,
not validation of transfer to finger shaping or a new geometry.

The true-force difference between grids is 17.17% for A and 7.09% for B on this
interval, relative to the finer-grid force norm. Therefore the simulation has
not established force convergence, especially for A. Small identified-vs-true
errors at a common grid should not hide that numerical limitation.

## Judgment and next decision

The observable-input route is feasible, and pressure elimination helps. Uniform
compression is not an adequate closure for this particular test. The remaining
interior approximation and B's stiffness bias prevent calling this a fully
validated central identification experiment yet. Keep the current paper intact.
The next useful improvement would address interior observability or specimen
geometry, with the same image/force inputs, rather than assuming uniform motion
or supplying hidden elastic states. Friction also needs a dedicated check before
claiming practical transfer to the hardware press.

## Reproduction and evidence

From the repository root:

```bash
.venv/bin/python -m experiments.robotics.plate_observable_reproduce --out out/plate_observable_FRESH --device cuda:0
.venv/bin/pytest -q tests/test_plate_observable.py
```

The output directory must be new. The reproduction runs truth, camera export,
tracking, initial failed closures, final identification, prediction, a finer-grid
comparison, isolated repetition and ideal-surface diagnostics. Ad hoc exploratory
single-field diagnostic JSONs from the original pilot are retained separately;
the replay reproduces the final estimator and the main failed closures.

`protocol.json` contains the complete motion, solver and material settings.
`inputs_A/B` are the allowed observations; `fit_A/B` contain camera tracks and
the failed uniform fit. **Final estimates are in `divfree_A/B`.** `field_A/B`
retain the failed pressure-sensitive nonuniform fit. `field_selection.json`,
`observation_isolation.json`, `tracking_accuracy.json`, `evaluation.json` and
`ideal_surface_check` contain selection and validation evidence. Sources and
environment are recorded in `provenance.json` and `environment.txt`.
'''
    (root/'REPORT.md').write_text(text)


def provenance(root):
    repo=Path(__file__).resolve().parents[2]
    paths=list((repo/'experiments/robotics').glob('plate_observable_*.py'))+[repo/'tests/test_plate_observable.py']
    paths+=list((repo/'experiments/elastic').glob('strip*.py'))
    paths+=list((repo/'src/warpmpm').rglob('*.py'))
    info=dict(python=sys.version,platform=platform.platform(),sources={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip())
    (root/'provenance.json').write_text(json.dumps(info,indent=2))
    (root/'environment.txt').write_text('\n'.join(sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions()))+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();media(a.out);write_report(a.out);provenance(a.out)
