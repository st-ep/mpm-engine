"""Report frozen observation-only estimates without hard-coded result claims."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw
from experiments.elastic.block_drop_report import font

INK = '#25343e'
MUTED = '#5b6973'
COLORS = {'A': '#227dad', 'B': '#c76c37'}


def validation_plot(root, p, evaluation):
    """Plot frozen forward predictions; never feeds back into identification."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for ax, k in zip(axes, 'AB'):
        for kind, label, color, style in [
            ('truth', 'Reference simulation', '#293943', '-'),
            ('prediction', 'Recovered parameters', COLORS[k], '--'),
        ]:
            f = np.genfromtxt(root/f'{kind}_{k}/force.csv', names=True, delimiter=',')
            ax.plot(f['time'], f['Fz'], style, color=color, lw=1.5, label=label)
        ax.axvspan(p['validation_start'], p['knots'][-1][0], color='#e8eef1', zorder=-2)
        ax.axvline(p['validation_start'], color='#8b969f', lw=.8)
        ax.set(xlabel='Time (s)', ylabel='Plate force (N)', xlim=(0, p['knots'][-1][0]))
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom, top*1.15)
        ax.set_title(f'Material {k}', loc='left', fontweight='bold', color=COLORS[k])
        ax.text(.02, .93, 'Identification probe', transform=ax.transAxes, fontsize=9)
        ax.text(p['validation_start']/p['knots'][-1][0]+.01, .93,
                'Withheld press', transform=ax.transAxes, fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
        ax.legend(loc='upper left', bbox_to_anchor=(0, .85), fontsize=8, frameon=False)
        e = evaluation[k]
        ax.text(.02, -.29, f"Withheld force error: {100*e['held_out_force_relative_l2']:.2f}%"
                f"  |  3D position RMS: {e['held_out_particle_rmse_mm']:.3f} mm",
                transform=ax.transAxes, fontsize=9)
    fig.savefig(root/'media/withheld_validation.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def crop(sequence, frame, width):
    # One fixed crop and physical image scale for both materials and all times.
    return Image.fromarray(sequence[frame]).convert('RGB').crop(
        (100, 180, 1180, 1080)).resize((width, round(width*900/1080)), Image.Resampling.LANCZOS)


def report(root):
    p = json.loads((root/'protocol.json').read_text())
    config=json.loads((root/'identification_protocol.json').read_text())
    cycles=config.get('balanced_identification_intervals')
    fits = {k: json.loads((root/f'divfree_{k}/identification.json').read_text()) for k in 'AB'}
    errors = {k: {'E_percent': 100*(fits[k]['E_pa']/p['E_pa'][k]-1),
                  'yield_percent': 100*(fits[k]['yield_pa']/p['yield_pa'][k]-1)} for k in 'AB'}
    passed = all(abs(e) <= 5 for row in errors.values() for e in row.values())
    robustness = json.loads((root/'numerical_robustness.json').read_text()) if (root/'numerical_robustness.json').exists() else None
    robust_pass = None if robustness is None else robustness['all_four_within_five_percent']
    (root/'parameter_evaluation.json').write_text(json.dumps(dict(
        signed_errors=errors, all_four_within_five_percent=passed,
        scope='Evaluation after estimates were frozen; this alone does not establish validation.'), indent=2)+'\n')
    seq = {k: np.load(root/f'inputs_{k}/camera_0.npy', mmap_mode='r') for k in 'AB'}
    times = np.load(root/'inputs_A/time.npy')
    press_time = min((pair for pair in p['knots'] if pair[0] <= p['fit_end']), key=lambda a:a[1])[0]
    frames = [0, int(np.argmin(abs(times-press_time))), len(times)-1]
    out = root/'media'; out.mkdir(exist_ok=True)
    canvas = Image.new('RGB', (1530, 1160), 'white'); draw = ImageDraw.Draw(canvas)
    draw.text((35, 22), 'Plastic identification from images and plate force', font=font(39, True), fill=INK)
    subtitle='Same light preload and deeper press for both materials' if cycles else 'Same geometry, texture and loading–unloading motion for both materials'
    draw.text((35, 78), subtitle, font=font(25), fill=MUTED)
    for col, title in enumerate(['Initial', 'Compressed', 'Released']):
        draw.text((100+col*475, 129), title, font=font(27, True), fill=INK)
    for row, k in enumerate('AB'):
        top = 175+row*390
        draw.text((34, top+155), k, font=font(35, True), fill=COLORS[k])
        for col, frame in enumerate(frames):canvas.paste(crop(seq[k], frame, 450), (85+col*475, top))
    draw.line((35, 966, 1495, 966), fill='#d9e1e6', width=2)
    draw.text((35, 984), 'Estimated parameters (signed error)', font=font(27, True), fill=INK)
    for row, k in enumerate('AB'):
        f = fits[k]; e = errors[k]
        draw.text((35, 1030+row*42), f"{k}    E = {f['E_pa']/1000:.2f} kPa ({e['E_percent']:+.2f}%)"
                  f"       Y = {f['yield_pa']/1000:.3f} kPa ({e['yield_percent']:+.2f}%)",
                  font=font(27), fill=COLORS[k])
    gate_text = 'All four parameter errors ≤5% at this resolution.' if passed else 'The ≤5% target has not been met for all four parameters.'
    if passed and robust_pass is False:
        gate_text = 'This resolution passes; the finer-image repeat does not. Robust 5% target not met.'
    elif passed and robust_pass is True:
        gate_text = 'All four parameter errors ≤5% at both tested resolutions.'
    draw.text((35, 1120), gate_text, font=font(23), fill=MUTED)
    canvas.save(out/'identification_preview.png')

    forces = {k: np.genfromtxt(root/f'inputs_{k}/force.csv', names=True, delimiter=',') for k in 'AB'}
    max_force = max(float(f['Fz'].max()) for f in forces.values())*1.08
    width, height = 1280, 960
    process = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
        '-s',f'{width}x{height}','-r','30','-i','-','-an','-c:v','libx264','-crf','19',
        '-pix_fmt','yuv420p',str(out/'plate_identification.mp4')], stdin=subprocess.PIPE)
    indices = [0]*18+[int(np.argmin(abs(times-t))) for t in np.arange(0, times[-1]+1e-9, .25/30)]+[len(times)-1]*24
    for frame in indices:
        t = times[frame]; im = Image.new('RGB', (width,height), 'white'); d = ImageDraw.Draw(im)
        d.text((30,20), 'One plate motion, two plastic responses', font=font(36,True), fill=INK)
        d.text((30,72), f'Synthetic stereo camera input  |  t = {t:.2f} s  |  0.25× speed', font=font(24), fill=MUTED)
        phase='Initial settling'
        if t>=times[-1]-1e-9:phase='Released'
        else:
            for start,end in cycles or [[.2,p['fit_end']]]:
                if start<=t<end:
                    prefix='Light probe' if cycles and end==cycles[0][1] else 'Main probe'
                    segment=next(((a,x,b,y) for (a,x),(b,y) in zip(p['knots'][:-1],p['knots'][1:]) if a<=t<b),None)
                    phase=prefix+' / '+('closing' if segment and segment[3]<segment[1] else 'unloading')
        d.text((30,108),phase,font=font(22),fill=MUTED)
        for col,k in enumerate('AB'):
            left=30+col*635; color=COLORS[k]
            d.text((left,145), f'Material {k}', font=font(27,True), fill=color)
            im.paste(crop(seq[k],frame,600),(left,185))
            f=forces[k]; force=float(np.interp(t,f['time'],f['Fz']))
            d.text((left,703), f'Plate force: {force:.2f} N', font=font(25), fill=color)
            x0=left+25; y0=860; ww=545; hh=105
            d.line((x0,y0,x0+ww,y0),fill='#bcc7cf',width=2)
            points=[(x0+tt/times[-1]*ww,y0-ff/max_force*hh) for tt,ff in zip(f['time'],f['Fz']) if tt<=t]
            if len(points)>1:d.line(points, fill=color,width=3)
            d.text((x0,y0+5),'0 s',font=font(18),fill=MUTED)
            d.text((x0+ww-60,y0+5),f'{times[-1]:.1f} s',font=font(18),fill=MUTED)
        d.text((30,925),'Same image scale and force scale. No simulator motion or elastic state is supplied to identification.',font=font(19),fill=MUTED)
        process.stdin.write(np.asarray(im).tobytes())
    process.stdin.close()
    if process.wait():raise RuntimeError('Video encoding failed')

    rows=[]
    for k in 'AB':
        f=fits[k];e=errors[k]
        rows.append(f"| {k} | {f['E_pa']/1000:.3f} | {e['E_percent']:+.2f}% | {f['yield_pa']/1000:.4f} | {e['yield_percent']:+.2f}% |")
    status='All four parameter errors meet the 5% target.' if passed else 'The 5% parameter target is not met.'
    if passed and robust_pass is False:
        status='The primary fit meets 5%, but the finer-image identification repeat fails. Do not claim robust 5% recovery.'
    elif passed and robust_pass is True:
        status='All four parameter errors meet 5% at both tested resolutions.'
    validation='Complete; see `evaluation.json`.' if (root/'evaluation.json').exists() else 'Not complete; do not claim validated recovery.'
    isolation='Passed; see `observation_isolation.json`.' if (root/'observation_isolation.json').exists() else 'Not yet repeated for this frozen run.'
    probe_description=(f'The probe contains a light loading–unloading cycle followed by a deeper press to {min(gap for time,gap in p["knots"] if time<=p["fit_end"])*1000:g} mm. '
                       'The two cycles receive equal relative weak-balance weight; plastic history is evaluated throughout both.' if cycles else 'Both materials receive the same loading–unloading probe.')
    text=f'''# Observable plastic identification

{status} The manuscript and shaping results are unchanged.

{probe_description}

![Press](media/identification_preview.png)

[Video: actual synthetic camera inputs, quarter speed](media/plate_identification.mp4)

| Material | Estimated E (kPa) | E error | Estimated Y (kPa) | Y error |
|---|---:|---:|---:|---:|
'''+ '\n'.join(rows)+f'''

True values are E=80 kPa for both, Y=1/10 kPa for A/B. Y is the engine's
deviatoric Kirchhoff-stress norm threshold, not conventional equivalent Cauchy
yield stress. Errors are evaluated only after fitting.

- **Inputs:** calibrated synthetic stereo images with applied texture, measured
  normal plate force/motion, timestamps and disclosed specimen metadata.
- **Inference:** image correlation and stereo recover surface motion. A
  square-symmetric spatial field extends it into the volume. Local plastic
  history and weak momentum balance estimate E and Y. This is a nonlinear
  two-parameter fit within a known family, not constitutive-family discovery.
- **Fit to the paper:** E is solved linearly conditional on a scalar search
  over Y/E, which determines the inferred plastic history. No forward dynamics
  rollouts enter fitting, but this is not the manuscript's single convex linear
  solve. It needs an explicit methodological distinction before replacing the
  current supplied-state plastic experiment.
- **Assumptions:** known homogeneous isotropic Hencky/von-Mises family, no
  hardening, known nu=0.30, density, initial geometry and stress-free reference;
  frictionless plates; square symmetry and approximate surface-to-volume
  reconstruction. No uniform compression, plane stress, plane strain or
  incompressibility assumption was added.
- **Measurement limits:** exact calibration, controlled lighting, applied
  texture, 0.5 gray-level image noise and noiseless simulated force. The videos
  are rendered; this is not hardware validation.
- **Independent observation-only repeat:** {isolation}
- **Withheld deeper press and finer-grid validation:** {validation}

Shared reconstruction settings are selected using held-out tracked points and
positive reconstructed volume. Integration is refined using changes in estimates,
not true parameter errors. Simulator states are reserved for image generation
and explicitly labelled numerical diagnostics. See `design.md` and source hashes
for the numerical changes underlying this run.
'''
    (root/'REPORT.md').write_text(text)
    if robustness is not None:
        extra = '\n## Identification repeated from finer-grid observations\n\n'
        extra += '| Material | E error | Y error |\n|---|---:|---:|\n'
        for k, e in robustness['signed_errors'].items():
            extra += f"| {k} | {e['E_percent']:+.2f}% | {e['yield_percent']:+.2f}% |\n"
        extra += ('\nThe same physical probe and observation-only procedure were repeated, including '
                  'reconstruction selection. These results are reported alongside the primary fit; '
                  'the mesh is not selected by its true-parameter errors. See `numerical_robustness.json`.\n')
        with (root/'REPORT.md').open('a') as f:
            f.write(extra)
    if (root/'evaluation.json').exists():
        evaluation = json.loads((root/'evaluation.json').read_text())
        validation_plot(root, p, evaluation)
        extra = '\n## Frozen forward prediction\n\n![Validation](media/withheld_validation.png)\n\n'
        has_fine_predictions = any('fine_held_out_force_relative_l2' in e for e in evaluation.values())
        extra += ('The deeper press was excluded from identification. Predictions start from '
                  'the original undeformed specimen and run the complete history, without '
                  'resetting to the reference state before validation. Position errors below '
                  'use simulated volume particles for evaluation only.\n\n')
        extra += '| Material | Withheld force L2 error | 3D position RMS (mm) |'
        extra += ' Finer-grid force error | Finer-grid position RMS (mm) |\n|---|---:|---:|---:|---:|\n' if has_fine_predictions else '\n|---|---:|---:|\n'
        for k, e in evaluation.items():
            ff = f"{100*e['fine_held_out_force_relative_l2']:.2f}%" if 'fine_held_out_force_relative_l2' in e else 'Not run'
            fx = f"{e['fine_held_out_particle_rmse_mm']:.4f}" if 'fine_held_out_particle_rmse_mm' in e else 'Not run'
            extra += (f"| {k} | {100*e['held_out_force_relative_l2']:.2f}% | "
                      f"{e['held_out_particle_rmse_mm']:.4f} |")
            extra += f' {ff} | {fx} |\n' if has_fine_predictions else '\n'
        extra += '\nReference-force changes under mesh refinement (relative L2 on the withheld interval): '
        extra += ', '.join(f"{k}: {100*e['grid_force_relative_l2']:.2f}%" for k, e in evaluation.items())
        extra += '. This is a mesh-sensitivity check, not proof of a continuum-converged solution.\n'
        if (root/'grid_comparison.json').exists():
            grid = json.loads((root/'grid_comparison.json').read_text())['relative_force_l2']
            extra += '\nReference-force changes during identification, normalized by the finer reference: '
            extra += ', '.join(f"{k}: {100*e['identification']:.2f}%" for k, e in grid.items())
            extra += '. Finite contact resolution remains a numerical limitation.\n'
        with (root/'REPORT.md').open('a') as f:
            f.write(extra)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    report(ap.parse_args().out)
