"""Independent ledger, provenance and timestep review of an isolated pair."""
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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work', type=Path)
    args = parser.parse_args()
    work = args.work.resolve()
    path = work / 'reference_protocol.json'
    protocol = json.loads(path.read_text())
    assert digest(path) == (work / 'reference_protocol.sha256').read_text().strip()
    for rel, sha in protocol['input_sha256'].items():
        assert digest(ROOT / rel) == sha, rel
    comparison = json.loads((work / 'reference_comparison.json').read_text())
    evidence, histories = [], []
    for scale in [1., .5]:
        case = work / 'reference_replays/replays/original-separable' / f'recorded60_bNA_n160_phase0_dt{scale:g}_mu0.117000'
        result = json.loads((case / 'result.json').read_text())
        expected_hash = next(r['result_sha256'] for r in comparison['cases'] if r['dt_scale'] == scale)
        assert digest(case / 'result.json') == expected_hash
        assert result['eta_pa_s'] == protocol['eta_pa_s'] and result['source_coulomb_friction'] == protocol['source_contact']
        assert result['particle_count'] == 229280 and result['initial_volume_ml'] == 300.
        assert result['dt_scale'] == scale and result['n_grid'] == 160
        assert not result['validation_outcomes_used'] and not result['measured_endpoints_read_by_forward_runner']
        assert result['initial_settling_gate']['passed']
        assert Path(result['actual_kernel_module']).resolve().is_relative_to(work / 'isolated_src')
        metrics = case / '09-04-60-2s/metrics.csv'
        with metrics.open() as stream:
            rows = list(csv.DictReader(stream))
        counts = np.array([[int(row[k]) for k in ['n_src', 'n_rcv', 'n_air_spill']] for row in rows])
        t = np.array([float(row['t']) for row in rows])
        assert len(rows) == 773 and np.all(np.diff(t) > 0)
        assert np.all(counts >= 0) and np.all(counts.sum(1) == 229280)
        volume = counts[:, 1] * 300 / 229280
        assert abs(volume[-1] - result['receiver_ml']) < 1e-10
        assert abs(np.ptp(volume[t >= t[-1] - .5]) - result['tail_variation_ml']) < 1e-9
        assert result['outside_ml'] <= protocol['outside_limit_ml']
        assert result['tail_variation_ml'] <= protocol['tail_limit_ml']
        state = np.load(case / '09-04-60-2s/final_n160.npz')
        assert np.isfinite(state['x']).all() and np.isfinite(state['v']).all()
        histories.append((t, volume))
        evidence.append(dict(dt_scale=scale, receiver_ml=float(volume[-1]),
            outside_ml=result['outside_ml'], tail_variation_ml=result['tail_variation_ml'],
            initial_settling_mean_speed_m_s=result['initial_settling_gate']['mean_speed_m_s'],
            result_sha256=digest(case / 'result.json'), metrics_sha256=digest(metrics)))
    difference = evidence[1]['receiver_ml'] - evidence[0]['receiver_ml']
    np.testing.assert_array_equal(histories[0][0], histories[1][0])
    common = histories[0][0]
    gap = histories[1][1] - histories[0][1]
    limit = protocol['full_pour_timestep_difference_limit_ml']
    review = dict(cases=evidence, endpoint_half_minus_full_ml=difference,
        maximum_absolute_curve_gap_ml=float(np.max(abs(gap))), threshold_ml=limit,
        curve_gap_evaluated_at='Every saved frame, with identical replay timestamps; no resampling',
        timestep_pair_passed=abs(difference) <= limit,
        frozen_input_hashes_passed=True, finite_states_and_count_ledgers_passed=True,
        initial_settling_and_capture_passed=True, physical_parameters_refitted=[],
        validation_outcomes_used=[], identification_changed=False,
        robot_accuracy_established=False, temporal_convergence_established=False,
        reviewer_sha256=digest(Path(__file__)))
    with (work / 'independent_review.json').open('x') as stream:
        json.dump(review, stream, indent=2)
        stream.write('\n')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for (t, volume), scale in zip(histories, [1., .5]):
        axes[0].plot(t, volume, label=f'Timestep factor {scale:g}: {volume[-1]:.2f} mL')
    axes[0].set(xlabel='Replay time (s)', ylabel='Receiver volume (mL)')
    axes[1].plot(common, gap, color='tab:purple')
    axes[1].axhspan(-limit, limit, color='tab:green', alpha=.15, label=f'±{limit:g} mL threshold')
    axes[1].axhline(0, color='gray', lw=.6)
    axes[1].set(xlabel='Replay time (s)', ylabel='Half minus standard timestep (mL)')
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle('Time-scaled AFLIP blend — original 60° replay, fixed viscosity and contact')
    fig.tight_layout()
    fig.savefig(work / 'reference_comparison.png', dpi=160)
    plt.close(fig)
    print(json.dumps(review, indent=2))


if __name__ == '__main__':
    main()
