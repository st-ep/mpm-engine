"""Two full-MPM target probes after same-pour numerical calibration checks.

No bulk search: angles were selected from the previous MPM table before the
new calibration. Robot timing comes only from joint/action histories. Liquid
validation outcomes are neither loaded nor used to select commands.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'out/pour_physics_audit/aflip_blend_time_20260907'
SOURCE = WORK / 'isolated_src'
TIMING = ROOT / 'out/pour_navier_calibration/measured_timing_20260907/motion_characterization.json'
ANGLES = (48.37, 53.04)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contact', type=float, required=True)
    parser.add_argument('--angle', type=float, choices=ANGLES)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if not args.preflight and args.angle is None:
        parser.error('--angle is required for a forward replay')
    sys.path.insert(0, str(SOURCE))
    import numpy as np
    from warpmpm.kernels import mpm_utils
    from experiments.pour import pour_transfer_recalibration_reference as reference
    from experiments.pour.pour_measured_timing import MeasuredTimingMotion, characterize, LOG_ROOT, TIMING_ANGLES

    assert Path(mpm_utils.__file__).resolve().is_relative_to(SOURCE)
    characterization = json.loads(TIMING.read_text())
    base, ep = reference.planned_motion(70.)
    # Reproduce the frozen clock correction without reading any liquid outcome.
    assert characterize(base) == characterization
    motion = MeasuredTimingMotion(base, characterization)
    for angle in np.linspace(45., 68., 93):
        old, new = motion.clock_knots(float(angle))
        assert np.all(np.diff(old) > 0) and np.all(np.diff(new) > 0)
        t = np.linspace(new[0], new[-1], 1001)
        joints = motion.joints(t, float(angle))
        assert np.isfinite(joints).all()
        # Corresponding clock knots must represent identical spatial poses.
        np.testing.assert_allclose(motion.joints(new, float(angle)),
            base.joints(old, float(angle)), rtol=0., atol=1e-12)
    if args.preflight:
        print(json.dumps(dict(clock_preflight_passed=True, allowed_probe_angles_deg=ANGLES,
            actual_kernel_module=mpm_utils.__file__, calibration_acceptance_checked=False,
            simulations_launched=0, liquid_validation_outcomes_used=[]), indent=2))
        return

    accepted_path = WORK / f'candidate_mu{args.contact:.6f}/all_review.json'
    accepted = json.loads(accepted_path.read_text())
    assert accepted['limited_numerical_checks_passed']
    assert accepted['source_contact'] == args.contact and accepted['eta_pa_s'] == reference.ETA
    assert not accepted['identification_changed'] and not accepted['geometry_changed']
    assert accepted['validation_outcomes_used'] == []
    paths = [Path(__file__), Path(reference.__file__), TIMING, accepted_path,
        ROOT / 'experiments/pour/pour_measured_timing.py',
        ROOT / 'experiments/pour/pour_navier_reference.py']
    hashes = {}
    for row in accepted['cases']:
        result_path = ROOT / row['result_path']
        assert digest(result_path) == row['result_sha256']
        protocol_path = ROOT / row['protocol_path']
        assert digest(protocol_path) == row['protocol_sha256']
        protocol = json.loads(protocol_path.read_text())
        hashes.update(protocol['input_sha256'])
        paths.extend([result_path, protocol_path])
    hashes.update(accepted['input_sha256'])
    for angle in TIMING_ANGLES:
        paths.extend(LOG_ROOT / f'recorded_{angle:g}' / name
                     for name in ['actions.jsonl', 'states.jsonl', 'meta.json'])
    hashes.update({str(path.relative_to(ROOT)): digest(path) for path in paths})

    def validate():
        for relative, sha in hashes.items():
            assert digest(ROOT / relative) == sha, relative

    validate()
    destination = WORK / f'candidate_mu{args.contact:.6f}/timing_probe_{args.angle:.2f}'
    destination.mkdir(exist_ok=False)
    protocol_path = destination / 'protocol.json'
    write_new(protocol_path, dict(purpose=__doc__, input_sha256=hashes,
        eta_pa_s=reference.ETA, source_contact=args.contact, grid=160, dt_scale=1.,
        planned_angle_deg=args.angle, allowed_probe_angles_deg=ANGLES,
        initial_volume_ml=300., particle_count=229280,
        viscosity_identification_pours=1, contact_calibration_pours=1,
        contact_calibration_episode='09-04-60-2s',
        contact_calibration='Original same-pour 159 mL endpoint; viscosity fixed',
        identification_changed=False, geometry_changed=False,
        spatial_path_changed=False, timing_model=characterization['method'],
        timing_characterization_angles_deg=list(TIMING_ANGLES),
        fluid_validation_outcomes_used=[], maximum_angle_probes=2,
        actual_kernel_module=str(Path(mpm_utils.__file__).resolve()),
        real_error_forecast_saved=False, bulk_search_queued=False))
    reference.OUT = destination / 'forward'
    old_factory = reference.planned_motion
    old_writer = reference.write_planned_episode
    old_project = reference.twin.project_out_of_solid
    gate = {}

    def timing_factory(max_angle):
        original, episode = old_factory(max_angle)
        return MeasuredTimingMotion(original, characterization), episode

    def timing_writer(*positional, **keywords):
        result = old_writer(*positional, **keywords)
        meta_path = positional[1] / 'meta.json'
        meta = json.loads(meta_path.read_text())
        meta['planning'].update(timing_method=characterization['method'],
            timing_protocol_sha256=digest(protocol_path))
        meta_path.write_text(json.dumps(meta, indent=2)+'\n')
        return result

    def checked_project(x, v, *positional, **keywords):
        if not gate:
            speed = np.linalg.norm(v, axis=1)
            gate.update(mean_speed_m_s=float(speed.mean()),
                passed=bool(np.isfinite(speed).all() and speed.mean()<reference.twin.SETTLE_SPEED))
            write_new(destination / 'settling_gate.json', gate)
            if not gate['passed']:
                raise RuntimeError('Initial settling failed; no pour will be run')
        return old_project(x, v, *positional, **keywords)

    reference.planned_motion = timing_factory
    reference.write_planned_episode = timing_writer
    reference.twin.project_out_of_solid = checked_project
    try:
        reference.run(SimpleNamespace(source_friction=args.contact, grid=160, phase=0.,
            wall='original-separable', slip_mm=None, angle=args.angle, max_angle=70.,
            dt_scale=1., device=args.device))
    finally:
        reference.planned_motion = old_factory
        reference.write_planned_episode = old_writer
        reference.twin.project_out_of_solid = old_project
    results = list((destination / 'forward').rglob('result.json'))
    assert len(results)==1 and gate['passed']
    result = json.loads(results[0].read_text())
    assert result['eta_pa_s']==reference.ETA and result['source_coulomb_friction']==args.contact
    assert result['planned_angle_deg']==args.angle and result['particle_count']==229280
    assert result['outside_ml']<=3. and result['tail_variation_ml']<=.5
    validate()
    # Wrapper metadata stays separate from the raw forward output.
    write_new(destination / 'completion.json', dict(receiver_ml=result['receiver_ml'],
        result_path=str(results[0].relative_to(ROOT)), result_sha256=digest(results[0]),
        protocol_sha256=digest(protocol_path), initial_settling_gate=gate,
        actual_kernel_module=str(Path(mpm_utils.__file__).resolve()),
        fluid_validation_outcomes_used=[], identification_changed=False,
        physical_target_accuracy_established=False, bulk_search_queued=False))


if __name__ == '__main__':
    main()
