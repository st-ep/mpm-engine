"""Collect completed diagnostic results without fitting or issuing commands."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'out/pour_physics_audit'
BASE=ROOT/'out/pour_weakform_recovery'


def read_curve(path,count):
    rows=list(csv.DictReader(path.open()))
    return (np.array([float(r['t'])-1 for r in rows]),
            np.array([float(r['n_rcv'])*300/count for r in rows]))


def main():
    baseline_path=BASE/'mpm/n384_shift0/angle_060.00/result.json'
    baseline=json.loads(baseline_path.read_text())
    candidates={}
    for name in ['baseline_source_sticky_n384','transport_source_separable_n384']:
        candidates[name]=json.loads((OUT/'replay'/name/'result.json').read_text())
    report=dict(status='No precision-pouring table released',calibration_pour_count=1,
                identification_episode='09-04-60-2s',
                six_validation_outcomes_used_for_fitting=False,
                simulator_viscosity_inverse_fit=False,
                same_pour_observed_endpoint_ml=159.,
                baseline=dict(eta_pa_s=baseline['provenance']['eta_pa_s'],receiver_ml=baseline['receiver_ml']),
                isolated_replays={name:{k:r[k] for k in ['eta_pa_s','receiver_ml','source_depletion_ml',
                                                      'outside_ml','tail_variation_ml','elapsed_s']}
                                  for name,r in candidates.items()},
                optical_audits={name:json.loads((OUT/name/'results.json').read_text())['cases']
                                for name in ['optical_resolution','optical_resolution_fine','optical_full_rate']},
                reduced_law_diagnostics=json.loads((OUT/'frozen_brink_diagnostic/results.json').read_text()),
                export_check=json.loads((OUT/'release_check.json').read_text()),
                conclusion='The weak estimator and MPM wall describe different flow mechanisms. Neither tested '
                           'wall variant establishes accurate transfer; image and closure uncertainties remain. '
                           'Do not fit an endpoint correction or issue an accuracy-qualified robot table.')
    fig,ax=plt.subplots(figsize=(10,6))
    obs=np.load(OUT/'optical_full_rate/observations_4096.npz')
    ax.plot(obs['t'],obs['rcv_vol']*1e6,'.',color='.45',ms=3,label='60° video: refined optical readout')
    t,v=read_curve(baseline_path.with_name('metrics.csv'),baseline['particle_count'])
    ax.plot(t,v,lw=2,label=f"Original MPM, separable wall: {v[-1]:.1f} mL")
    for name,r in candidates.items():
        t,v=read_curve(OUT/'replay'/name/'09-04-60-2s/metrics.csv',r['particle_count'])
        label='No-slip source wall' if r['source_wall']=='sticky' else 'Conservative weak transport, separable wall'
        ax.plot(t,v,lw=2,label=f'{label}: {v[-1]:.1f} mL')
    ax.axhline(159,color='k',ls='--',lw=1,label='Reported settled amount: 159 mL (check only)')
    ax.axvspan(6.32865834236145,8.33254909515381,color='k',alpha=.05)
    ax.set(xlim=(0,12),ylim=(0,190),xlabel='Seconds after tilt command',ylabel='Receiver volume (mL)',
           title='Exact 60° reference: weak identification retained, no endpoint fitting')
    ax.grid(alpha=.2);ax.legend(loc='upper left',fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'reference_comparison.png',dpi=160);plt.close(fig)
    film=[]
    for path in sorted((OUT/'film').glob('*/result.json')):
        r=json.loads(path.read_text())
        film.append({k:r[k] for k in ['name','collider','surface','grid','phase','band_cells',
                                     'flux_ratio_to_noslip','profile_relative_l2','tail_change_relative']})
    report['analytical_film_cases']=film
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for phase in [0.,.5]:
        cases=[r for r in film if r['collider']=='sdf' and r['surface']=='sticky'
               and r['band_cells']==.5 and r['phase']==phase]
        cases.sort(key=lambda r:r['grid'])
        axes[0].plot([r['grid']/4 for r in cases],[r['flux_ratio_to_noslip'] for r in cases],
                     'o-',label=f'Sticky SDF, phase {phase:g} cell')
    axes[0].axhline(1,color='k',ls='--');axes[0].axhspan(.9,1.1,color='green',alpha=.07)
    axes[0].set(xlabel='Grid cells across film',ylabel='MPM / analytical mean velocity',
                title='No-slip contact still depends on grid placement');axes[0].legend(fontsize=8)
    for name in ['sdf_separable_n64_phase0_band0.5','sdf_sticky_n64_phase0_band0.5']:
        path=OUT/'film'/name
        r=json.loads((path/'result.json').read_text());h=np.load(path/'state.npz')['history']
        axes[1].plot(h[:,0],h[:,1]/r['exact_noslip_mean_velocity_m_s'],label=r['surface'])
    axes[1].axhline(1,color='k',ls='--',label='Analytical no-slip steady state')
    axes[1].set(xlabel='Time (s)',ylabel='MPM / analytical mean velocity',
                title='Separable wall permits continued sliding');axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'film_verification.png',dpi=160);plt.close(fig)
    paths=[Path(__file__),baseline_path,*[OUT/'replay'/n/'result.json' for n in candidates],
           ROOT/'docs/pour_20260906_physics_audit.md']
    report['input_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (OUT/'audit_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','baseline','isolated_replays']},indent=2))


if __name__=='__main__':
    main()
