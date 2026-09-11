"""Independently check and plot a completed arithmetic-correction reference pair."""
from pathlib import Path
import argparse
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('variant', choices=['stable_volume', 'compensated_position'])
    args = parser.parse_args()
    work = ROOT / 'out/pour_physics_audit' / (args.variant + '_20260907')
    protocol_path = work / 'reference_protocol.json'
    assert digest(protocol_path) == (work / 'reference_protocol.sha256').read_text().strip()
    protocol = json.loads(protocol_path.read_text())
    for relative, expected in protocol['input_sha256'].items():
        assert digest(ROOT / relative) == expected, relative
    comparison = json.loads((work / 'reference_comparison.json').read_text())
    evidence, histories = [], []
    for scale in [1., .5]:
        case = work / 'reference_replays/replays/original-separable' / f'recorded60_bNA_n160_phase0_dt{scale:g}_mu0.117000'
        result_path = case / 'result.json'
        result = json.loads(result_path.read_text())
        assert digest(result_path) == next(r for r in comparison['cases'] if r['dt_scale'] == scale)['result_sha256']
        assert result['eta_pa_s'] == protocol['eta_pa_s']
        assert result['source_coulomb_friction'] == protocol['source_contact']
        assert result['initial_volume_ml'] == 300 and result['particle_count'] == 229280
        assert result['dt_scale'] == scale and result['n_grid'] == 160
        assert not result['validation_outcomes_used'] and not result['measured_endpoints_read_by_forward_runner']
        assert Path(result['actual_kernel_module']).resolve().is_relative_to(work / 'isolated_src')
        metrics_path = case / '09-04-60-2s/metrics.csv'
        with metrics_path.open() as f:
            records = list(csv.DictReader(f))
        t = np.array([float(r['t']) for r in records])
        counts = np.array([[int(r[k]) for k in ['n_src','n_rcv','n_air_spill']] for r in records])
        assert len(records) == 773 and np.all(np.diff(t)>0)
        assert np.all(counts>=0) and np.all(counts.sum(axis=1)==229280)
        volume = counts[:,1] * 300/229280
        assert abs(volume[-1]-result['receiver_ml'])<1e-10
        tail = volume[t >= t[-1]-.5]
        assert abs(np.ptp(tail)-result['tail_variation_ml'])<1e-9
        assert result['outside_ml'] <= protocol['outside_limit_ml']
        assert result['tail_variation_ml'] <= protocol['tail_limit_ml']
        histories.append((t,volume))
        evidence.append(dict(dt_scale=scale, receiver_ml=result['receiver_ml'],
            outside_ml=result['outside_ml'], tail_variation_ml=result['tail_variation_ml'],
            maximum_projected_count=int(records[-1]['projected']),
            result_sha256=digest(result_path), metrics_sha256=digest(metrics_path)))
    difference = evidence[1]['receiver_ml']-evidence[0]['receiver_ml']
    common = np.linspace(max(h[0][0] for h in histories), min(h[0][-1] for h in histories), 1000)
    gap = np.interp(common,*histories[1])-np.interp(common,*histories[0])
    fig, axes = plt.subplots(1,2,figsize=(10,3.8))
    for (t,v), scale in zip(histories,[1.,.5]):
        axes[0].plot(t,v,label=f'Timestep factor {scale:g}: {v[-1]:.2f} mL')
    axes[0].set(xlabel='Replay time (s)',ylabel='Receiver volume (mL)')
    axes[0].legend(fontsize=8)
    axes[1].plot(common,gap,color='tab:purple')
    axes[1].axhspan(-1,1,alpha=.12,color='tab:green',label='±1 mL comparison threshold')
    axes[1].axhline(0,color='gray',lw=.6)
    axes[1].set(xlabel='Replay time (s)',ylabel='Half minus standard timestep (mL)')
    axes[1].legend(fontsize=8)
    for ax in axes: ax.grid(alpha=.2)
    title = 'Stable volume update' if args.variant == 'stable_volume' else 'Stable volume + compensated position'
    fig.suptitle(f'{title} — original 60° replay, fixed viscosity and contact')
    fig.tight_layout()
    fig.savefig(work / 'reference_comparison.png',dpi=160)
    plt.close(fig)
    review = dict(cases=evidence, endpoint_half_minus_full_ml=difference,
        maximum_absolute_curve_gap_ml=float(np.max(abs(gap))),
        threshold_ml=protocol['full_pour_timestep_difference_limit_ml'],
        timestep_pair_passed=abs(difference)<=protocol['full_pour_timestep_difference_limit_ml'],
        finite_count_ledgers_and_capture_passed=True,
        frozen_input_hashes_passed=True, identification_changed=False,
        physical_parameters_refitted=[], validation_outcomes_used=[],
        robot_accuracy_established=False, temporal_convergence_established=False,
        reviewer_sha256=digest(Path(__file__)))
    (work / 'independent_review.json').write_text(json.dumps(review,indent=2)+'\n')
    print(json.dumps(review,indent=2))


if __name__=='__main__': main()
