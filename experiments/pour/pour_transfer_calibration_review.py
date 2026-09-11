"""Audit original60 calibration probes and compare their same-video dynamics.

The same original159 mL endpoint is a calibration datum. Relative receiver gain
is compared at the unchanged weak-fit samples, with timestamp alignment from
the logged pre-roll. Neither an offset nor a time shift is fitted. This script
does not select parameters, consume other pours, or forecast real target errors.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np
from experiments.pour import pour_navier_reference as reference
from experiments.pour.pour_weakform_recovery import fit

ROOT = Path(__file__).resolve().parents[2]


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('cases', type=Path, nargs='+')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier review; select a new output file')
    pour, ret, _ = reference.twin.recorded_pour_actions(reference.REFERENCE)
    window = [pour['t_ack']-pour['t_send']+.15, ret['t_send']-pour['t_send']-.15]
    eta, se, keep, dv, forcing, obs, rms = fit(dict(np.load(reference.OBSERVATIONS)), window)
    assert abs(eta-reference.ETA) < 1e-10
    episode = reference.twin.load_episode(reference.REFERENCE, reference.twin.PRE_ROLL,
                                         reference.twin.HOLD_SECONDS)
    queries = obs['t'] + episode['t_pour']
    evidence = []
    for case in args.cases:
        case = case.resolve()
        protocol_path = case / 'protocol.json'
        protocol = json.loads(protocol_path.read_text())
        for rel, sha in protocol['input_sha256'].items():
            assert digest(ROOT / rel) == sha, rel
        paths = list((case / 'forward').rglob('result.json'))
        assert len(paths) == 1
        result = json.loads(paths[0].read_text())
        assert result['numerical_correction_protocol_sha256'] == digest(protocol_path)
        assert result['eta_pa_s'] == eta and result['source_coulomb_friction'] == protocol['source_coulomb_friction']
        assert not result['validation_outcomes_used'] and result['initial_settling_gate']['passed']
        assert result['identification_episode'] == reference.REFERENCE.name and result['planned_angle_deg'] is None
        assert result['particle_count'] == {160:229280, 192:396197}[result['n_grid']]
        completion = json.loads((case / 'completion.json').read_text())
        assert completion['result_sha256'] == digest(paths[0])
        metrics = paths[0].parent / '09-04-60-2s/metrics.csv'
        with metrics.open() as stream:
            rows = list(csv.DictReader(stream))
        t = np.array([float(r['t']) for r in rows])
        counts = np.array([[int(r[k]) for k in ['n_src', 'n_rcv', 'n_air_spill']] for r in rows])
        n = result['particle_count']
        assert len(rows) == 773 and np.all(np.diff(t)>0)
        assert np.all(counts>=0) and np.all(counts.sum(1)==n)
        v = 300.*counts[:,1]/n
        assert abs(v[-1]-result['receiver_ml']) < 1e-9
        assert result['outside_ml'] <= 3 and result['tail_variation_ml'] <= .5
        assert queries.min() >= t.min() and queries.max() <= t.max()
        fitted_times = np.interp(queries, t, v)
        delta = fitted_times-fitted_times[0]
        state = np.load(paths[0].parent / f'09-04-60-2s/final_n{result["n_grid"]}.npz')
        assert np.isfinite(state['x']).all() and np.isfinite(state['v']).all()
        evidence.append(dict(source_contact=result['source_coulomb_friction'],
            n_grid=result['n_grid'], dt_scale=result['dt_scale'], receiver_ml=float(v[-1]),
            in_sample_endpoint_residual_ml=float(v[-1]-159.),
            one_ml_endpoint_tolerance_passed=bool(abs(v[-1]-159.) <= 1.),
            outside_ml=result['outside_ml'], tail_variation_ml=result['tail_variation_ml'],
            same_video_hold_gain_ml=float(delta[-1]),
            same_video_relative_volume_rmse_ml=float(np.sqrt(np.mean((delta-dv)**2))),
            protocol_sha256=digest(protocol_path), result_sha256=digest(paths[0]),
            metrics_sha256=digest(metrics)))
    output = dict(purpose=__doc__, cases=evidence, eta_pa_s=eta,
        weak_fit_reproduced=True, weak_fit_rmse_ml=rms, observed_hold_gain_ml=float(dv[-1]),
        fitted_weak_law_hold_gain_ml=float(forcing[-1]/eta), kept_samples=int(keep.sum()),
        window_after_pour_command_s=window, replay_time_offset_s=episode['t_pour'],
        one_pour_calibration_endpoint_ml=159., relative_curve_offset='Subtract each curve at its first fixed-window sample, as in identification; no offset parameter fit',
        time_alignment_fitted=False, calibration_pour_count=1,
        validation_outcomes_used=[], parameters_selected_by_reviewer=[],
        future_robot_accuracy_established=False,
        observation_sha256=digest(reference.OBSERVATIONS), reviewer_sha256=digest(Path(__file__)))
    with args.output.open('x') as stream:
        json.dump(output, stream, indent=2)
        stream.write('\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
