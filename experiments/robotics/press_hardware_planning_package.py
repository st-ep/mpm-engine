"""Freeze the selected compact hardware models and their reviewable evidence."""
import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np
from experiments.robotics.press_hardware_observe import read, save
from experiments.robotics.press_hardware_planning_assess import assess
from experiments.robotics.press_hardware_release import compare

ROOT=Path(__file__).resolve().parents[2]
SELECTED={
    'play_doh':'planning_release_v1/play_doh/release',
    'butter_slime':'soft_recovery_diagnostic/butter_slime/release',
    'plasticine':'gray_elastic_recovery_diagnostic/plasticine/release',
}


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(1<<20):h.update(block)
    return h.hexdigest()


def prepare(scratch,out):
    out.mkdir(exist_ok=True,parents=True)
    observations=scratch/'filtered_measured';release=scratch/'release_observations_v2'
    materials={};metrics={};runs={}
    for material,relative in SELECTED.items():
        run=scratch/relative;c=read(run/'completion.json')
        if not c['complete']:raise ValueError(relative)
        a=assess(run,observations);save(run/'assessment.json',a)
        compare(run,observations,release)
        dest=out/material;dest.mkdir(exist_ok=True)
        shutil.copytree(run,dest/'selected',dirs_exist_ok=True)
        save(dest/'fit.json',dict(selected=c['parameters']))
        G=c['shear_modulus_Pa'];K=c['bulk_modulus_assumed_Pa']
        materials[material]=dict(parameters=c['parameters'],equivalent_E_Pa=9*K*G/(3*K+G),
            equivalent_nu=(3*K-2*G)/(2*(3*K+G)),G_Pa=G,K_Pa=K,
            bulk_scope='Derived from E and fixed nu=0.45' if material=='plasticine' else 'Shared effective bulk calibration K=15 kPa',
            representative_episode=c['episode'],selected_run=str(run.resolve()))
        metrics[material]=dict(pressing=a,full_release=read(run/'release_assessment.json'))
        runs[material]=str(run.resolve())
    save(out/'models.json',dict(materials=materials,
        method='Exploratory forward calibration after weak-form diagnostics; not weak-form-only identification. All recordings inspected during development.',
        shared_soft_flow=dict(power=2,stress_squared_time_scale_Pa2_s=60e6,
            equation='plastic logarithmic strain rate norm = max(||dev Kirchhoff stress|| - Y, 0)^2 / (60e6 Pa^2 s)',
            scope='G and Y vary by soft material. Bulk stiffness and stress-normalized flow coefficient are additional shared calibration choices, not independent measurements.'),
        contact=dict(friction_assumed=.15,adhesion='Unmeasured and omitted; some effects may be absorbed in effective material calibration, with no guarantee of transfer to fingers.'),
        interpretation='Nominal models for exploratory planning; parameter uniqueness and transfer to X shaping have not been established.'))
    save(out/'metrics.json',metrics)
    protocol=dict(materials=list(SELECTED),heldout={m:materials[m]['representative_episode'] for m in SELECTED},
        heldout_key_scope='Compatibility key used by the existing forward runner; these are representative development recordings, not an untouched test set.',
        density_kg_m3=1000.,nu=.45,plate_radius_m=.05,ball_diameter_m=.045,
        contact_band_cells=1.,max_substep_s=5e-6,use_measured_initial_tilt=True,withdrawal_start_s=13.4,
        comparison_duration_s=15.,servo_gain_m_per_N_s=.012,
        force_input_scope='Recorded incremental normal force target through 13.4 s. Unknown initial physical preload is omitted. Thereafter recorded vertical tool displacement increments are prescribed, anchored to the predicted gap.',
        geometry_scope='User-supplied nominal 45 mm sphere and 100 mm pad. Camera fixed from initial contour plus recorded orientation; initial stress-free tangency is a conditional approximation.',
        calibration_method='Exploratory forward calibration; all recordings inspected; one selected parameter set per material across comparisons.',
        selection_scope='Favor compact models that separate spreading and retained shape. User explicitly accepts approximate recovery because support motion and sticking are not separated.')
    save(out/'protocol.json',protocol)
    for i in range(12):
        ep=f'ep{i:04d}';dest=out/ep;dest.mkdir(exist_ok=True)
        for name in ['geometry.json','camera.json','load_only.npz','motion.npz']:
            shutil.copy2(observations/ep/name,dest/name)
        shutil.copy2(ROOT/'out/press_observation_assessment_20260913'/ep/'assessment.json',dest/'assessment.json')
        shutil.copy2(scratch/'planning_release_v1'/ep/'withdrawal.npz',dest/'withdrawal.npz')
    shutil.copytree(release,out/'release_observations_v2',dirs_exist_ok=True)
    source=out/'source/experiments/robotics';source.mkdir(parents=True,exist_ok=True)
    for f in (ROOT/'experiments/robotics').glob('press_hardware_*.py'):shutil.copy2(f,source/f.name)
    # Frozen hardware modules take precedence; other experiment dependencies
    # can still be imported from the matching repository checkout.
    for parent in [source,source.parent]:
        (parent/'__init__.py').write_text('from pkgutil import extend_path\n__path__ = extend_path(__path__, __name__)\n')
    manifest={str(f.relative_to(out)):digest(f) for f in (out/'source').rglob('*.py')}
    engine={str(f.relative_to(ROOT)):digest(f) for f in (ROOT/'src/warpmpm').rglob('*.py')}
    raw={}
    for ep in protocol['heldout'].values():
        for name in ['meta.json','states.jsonl','frames_hand.jsonl','hand_rgb.mp4']:
            f=ROOT/'press_real_data'/ep/name;raw[str(f.relative_to(ROOT))]=digest(f)
    save(out/'provenance.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),selected_runs=runs,
        source_sha256=manifest,engine_sha256=engine,raw_representative_sha256=raw,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        selection='Parameters frozen for the current checks; no claim of blind validation or constitutive-family discovery.',
        manuscript_changed=False,committed_or_pushed=False))
    plot(out)


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names={'play_doh':'Play-Doh','butter_slime':'Butter slime','plasticine':'Plasticine'}
    colors={'play_doh':'#bd3438','butter_slime':'#ad9200','plasticine':'#545754'}
    fig,axes=plt.subplots(2,3,figsize=(12,6),sharex=True,constrained_layout=True)
    for col,m in enumerate(SELECTED):
        c=read(out/m/'selected/completion.json');d=np.load(out/m/'selected/trajectory.npz');o=np.load(out/c['episode']/'motion.npz')
        f=d['force_log'];s=d['shape_log'];keep=s[:,0]<=13.4
        axes[0,col].plot(o['time'],o['raw_height']*1000,color=colors[m],label='Recorded gap proxy')
        axes[0,col].plot(f[f[:,0]<=13.4,0],f[f[:,0]<=13.4,3]*1000,'k--',label='Predicted plate gap')
        axes[1,col].plot(o['time'],o['observed_max_width']*1000,color=colors[m],label='Camera-derived diameter')
        axes[1,col].plot(s[keep,0],s[keep,1]*1000,'k--',label='Predicted diameter')
        axes[0,col].set_title(names[m]);axes[1,col].set_xlabel('Time (s)')
        for row in range(2):
            ax=axes[row,col];ax.axvspan(10.4,13.4,color='#d9e4e9',alpha=.6);ax.set_xlim(0,13.4);ax.grid(alpha=.2);ax.legend(fontsize=7)
    axes[0,0].set_ylabel('Plate gap (mm)');axes[1,0].set_ylabel('Lateral diameter (mm)')
    fig.suptitle('21 N development recordings | shaded interval: partial unloading, still in contact',fontsize=12)
    fig.savefig(out/'comparison_curves.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scratch',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();prepare(a.scratch,a.out)
