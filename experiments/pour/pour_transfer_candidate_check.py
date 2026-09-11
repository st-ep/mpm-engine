"""Verify a same-pour contact candidate before any new-angle trial.

Checks original60 endpoint, grid and timestep consistency. Passing these limited
screens permits two angle probes, not a claim about real-robot target accuracy.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'out/pour_physics_audit/aflip_blend_time_20260907'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def case_path(contact, grid, scale):
    return WORK / f'recalibration_mu{contact:.6f}_n{grid}_dt{scale:g}'


def read_case(contact, grid, scale):
    case = case_path(contact, grid, scale)
    protocol_path = case / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    for rel, sha in protocol['input_sha256'].items():
        assert digest(ROOT / rel) == sha, rel
    paths = list((case / 'forward').rglob('result.json'))
    assert len(paths) == 1
    path = paths[0]
    result = json.loads(path.read_text())
    completion = json.loads((case / 'completion.json').read_text())
    assert completion['result_sha256'] == digest(path)
    assert result['numerical_correction_protocol_sha256'] == digest(protocol_path)
    assert result['eta_pa_s'] == 3.4392377844275503 and result['source_coulomb_friction'] == contact
    assert result['n_grid'] == grid and result['dt_scale'] == scale
    assert result['initial_volume_ml'] == 300. and result['particle_count'] == {160:229280, 192:396197}[grid]
    assert result['actual_kernel_module'] == protocol['actual_kernel_module']
    assert Path(result['actual_kernel_module']).resolve().is_relative_to(WORK / 'isolated_src')
    assert result['initial_settling_gate']['passed']
    assert not result['validation_outcomes_used'] and not result['measured_endpoints_read_by_forward_runner']
    assert result['source_wall'] == 'original-separable' and result['receiver_wall'] == 'separable, friction=0.05'
    assert result['identification_episode'] == '09-04-60-2s' and result['planned_angle_deg'] is None
    assert result['outside_ml'] <= 3. and result['tail_variation_ml'] <= .5
    metrics = path.parent / '09-04-60-2s/metrics.csv'
    with metrics.open() as stream:
        rows = list(csv.DictReader(stream))
    t = np.array([float(r['t']) for r in rows])
    counts = np.array([[int(r[k]) for k in ['n_src', 'n_rcv', 'n_air_spill']] for r in rows])
    assert len(rows) == 773 and np.all(np.diff(t)>0)
    assert np.all(counts>=0) and np.all(counts.sum(1)==result['particle_count'])
    volume = 300.*counts[:,1]/result['particle_count']
    assert abs(volume[-1]-result['receiver_ml']) < 1e-9
    assert abs(np.ptp(volume[t >= t[-1]-.5])-result['tail_variation_ml']) < 1e-9
    data = np.load(path.parent / f'09-04-60-2s/final_n{grid}.npz')
    assert np.isfinite(data['x']).all() and np.isfinite(data['v']).all()
    row = dict(n_grid=grid, dt_scale=scale, source_contact=contact,
        receiver_ml=result['receiver_ml'], outside_ml=result['outside_ml'],
        tail_variation_ml=result['tail_variation_ml'], initial_settling_gate=result['initial_settling_gate'],
        result_path=str(path.relative_to(ROOT)), result_sha256=digest(path),
        protocol_path=str(protocol_path.relative_to(ROOT)), protocol_sha256=digest(protocol_path),
        metrics_sha256=digest(metrics))
    return row, t, volume


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contact', type=float, required=True)
    parser.add_argument('--stage', choices=['grid', 'time', 'all'], required=True)
    args = parser.parse_args()
    revision_path = WORK / 'calibration_range_revision.json'
    revision = json.loads(revision_path.read_text())
    assert revision['new_contact_interval'][0] <= args.contact <= revision['new_contact_interval'][1]
    out = WORK / f'candidate_mu{args.contact:.6f}'
    out.mkdir(exist_ok=True)
    output = out / f'{args.stage}_review.json'
    if output.exists():
        raise FileExistsError('Preserve earlier review')
    trials = [(160, 1.)] + ([(192, 1.)] if args.stage=='grid' else [(160, .5)] if args.stage=='time' else [(192, 1.), (160, .5)])
    checked = [read_case(args.contact, g, s) for g, s in trials]
    rows = [r[0] for r in checked]
    baseline = rows[0]['receiver_ml']
    if args.stage != 'all':
        same_video = out / f'{args.stage}_same_video_review.json'
        subprocess.run([sys.executable, '-m', 'experiments.pour.pour_transfer_calibration_review',
            '--output', str(same_video), *[str(case_path(args.contact,g,s)) for g,s in trials]],
            cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        limit = revision['numerical_gates']['same_parameter_160_192_limit_ml' if args.stage=='grid' else 'same_parameter_timestep_pair_limit_ml']
        np.testing.assert_array_equal(checked[0][1], checked[1][1])
        gap = checked[1][2]-checked[0][2]
        result = dict(stage=args.stage, cases=rows, eta_pa_s=revision['eta_pa_s'],
            source_contact=args.contact, endpoint_calibration_passed=abs(baseline-159.)<=1.,
            endpoint_comparison_difference_ml=rows[1]['receiver_ml']-baseline,
            comparison_limit_ml=limit, comparison_passed=abs(rows[1]['receiver_ml']-baseline)<=limit,
            maximum_raw_frame_curve_difference_ml=float(np.max(abs(gap))),
            same_video_review_sha256=digest(same_video),
            numerical_acceptance_complete=False, physical_target_accuracy_established=False,
            validation_outcomes_used=[], calibration_range_revision_sha256=digest(revision_path),
            reviewer_sha256=digest(Path(__file__)))
        fig, axes = plt.subplots(1,2,figsize=(10,3.8))
        for row,t,v in checked:
            axes[0].plot(t,v,label=f'{row["n_grid"]}³, dt ×{row["dt_scale"]:g}: {v[-1]:.2f} mL')
        axes[0].axhline(159.,color='gray',ls=':',label='Same-pour calibration: 159 mL')
        axes[0].set(xlabel='Replay time (s)',ylabel='Receiver volume (mL)')
        axes[1].plot(checked[0][1],gap,color='tab:purple')
        axes[1].axhspan(-limit,limit,color='tab:green',alpha=.15,label=f'±{limit:g} mL endpoint criterion')
        axes[1].set(xlabel='Replay time (s)',ylabel='Check minus 160³ standard timestep (mL)')
        for ax in axes:
            ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.suptitle(f'Original 60° calibration — contact {args.contact:g}; viscosity fixed at 3.439 Pa·s')
        fig.tight_layout();fig.savefig(out/f'{args.stage}_comparison.png',dpi=160);plt.close(fig)
    else:
        reviews = {stage:json.loads((out/f'{stage}_review.json').read_text()) for stage in ['grid','time']}
        for stage, review in reviews.items():
            for row in review['cases']:
                assert digest(ROOT/row['result_path']) == row['result_sha256']
            assert review['source_contact'] == args.contact
        passed = abs(baseline-159.) <= 1. and all(r['comparison_passed'] for r in reviews.values())
        result = dict(stage='all', cases=rows, eta_pa_s=revision['eta_pa_s'],
            source_contact=args.contact, limited_numerical_checks_passed=passed,
            one_pour_calibration_endpoint_ml=159., calibration_pour_count=1,
            boundary_parameter_calibrated=passed, forward_volume_adjustment_applied=False,
            identification_changed=False, geometry_changed=False,
            allowed_next_action='Two new-angle probes with measured timing; chat forecast before any bulk campaign' if passed else 'Review failed criteria; no target-angle release',
            bulk_target_search_authorized=False, physical_target_accuracy_established=False,
            full_numerical_convergence_established=False, validation_outcomes_used=[],
            interpretation='One-pour weak viscosity plus same-pour effective numerical contact calibration',
            remaining_limitations=['Coarse analytical shear amplitude exceeds 5% error',
                'Coulomb wall is an effective calibrated approximation to the no-slip weak closure',
                'Cup reinsertion and future robot execution variability are not measured by these numerical checks'],
            input_sha256={str((out/f'{s}_review.json').relative_to(ROOT)):digest(out/f'{s}_review.json') for s in ['grid','time']},
            calibration_range_revision_sha256=digest(revision_path), reviewer_sha256=digest(Path(__file__)))
    with output.open('x') as stream:
        json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
