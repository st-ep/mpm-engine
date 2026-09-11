"""Check planned held robot poses against command definitions, without liquid data.

This is a forward-kinematics diagnostic. It neither measures cup reinsertion nor
changes any model, trajectory or command. No acceptance threshold is inferred
from the results. The sampled range includes the high-volume extrapolation.
"""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np
from scipy.spatial.transform import Rotation

from experiments.pour import pour_transfer_recalibration_reference as reference

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT/'out/pour_physics_audit/aflip_blend_time_20260907/candidate_mu0.272000'
OUT = CANDIDATE/'table_search/command_pose_check'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    model_path = CANDIDATE/'target_model_protocol.json'
    assert digest(model_path) == model_path.with_suffix('.sha256').read_text().strip()
    model = json.loads(model_path.read_text())
    for rel, expected in model['input_sha256'].items():
        assert digest(ROOT/rel) == expected, rel
    motion, episode = reference.planned_motion(70.)
    arm = reference.twin.RecordedPanda(episode, reference.MESH,
        height=64, width=64, max_geom=4000)
    angles = np.unique(np.r_[np.linspace(45.,68.,93),
        [47.18,49.77,52.23,54.77,57.35,62.88,65.84]])
    rows = []
    try:
        for angle in angles:
            ack, ret, _ = motion.timing(float(angle))
            half = np.radians(angle/2.)
            c, s = np.cos(half), np.sin(half)
            target_r = Rotation.from_quat([.5*(c+s),.5*(c-s),-.5*(c-s),.5*(c+s)])
            target_p = np.array([.5-angle/1200.,0.,.1+angle/1200.])
            orientation_errors, position_errors = [], []
            for time in np.linspace(ack+.2,ret-.2,25):
                arm.data.qpos[:7] = motion.joints([time],float(angle))[0]
                arm.mj.mj_forward(arm.model,arm.data)
                actual_r = Rotation.from_quat(np.roll(arm.data.xquat[arm.ee],-1))
                tcp = arm.data.xpos[arm.ee]+actual_r.apply(arm.TCP_LOCAL)
                orientation_errors.append(float(np.degrees((actual_r*target_r.inv()).magnitude())))
                position_errors.append(float(1000*np.linalg.norm(tcp-target_p)))
            assert np.isfinite(orientation_errors).all() and np.isfinite(position_errors).all()
            rows.append(dict(angle_deg=float(angle),
                maximum_held_orientation_error_deg=max(orientation_errors),
                maximum_held_tcp_position_error_mm=max(position_errors)))
    finally:
        arm.close()
    csv_path = OUT/'sampled_robot_pose_errors.csv'
    with csv_path.open('x',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    paths = [Path(__file__),Path(reference.__file__),Path(reference.twin.__file__),model_path,
        ROOT/'experiments/pour/pour_angle_sweep.py',reference.MESH,
        *[reference.REFERENCE/name for name in ['actions.jsonl','states.jsonl','meta.json']]]
    report = dict(purpose=__doc__,angles_sampled=len(rows),hold_samples_per_angle=25,
        angle_range_deg=[45.,68.],
        maximum_sampled_orientation_error_deg=max(r['maximum_held_orientation_error_deg'] for r in rows),
        maximum_sampled_tcp_position_error_mm=max(r['maximum_held_tcp_position_error_mm'] for r in rows),
        current_200ml_proposal=next(r for r in rows if r['angle_deg']==65.84),
        parameters_fitted=[],commands_changed=False,liquid_measurements_used=[],
        actual_cup_reinsertion_measured=False,real_volume_errors_estimated=False,
        interpretation='Differences from nominal command pose; not a measured robot or cup-pose error bound.',
        timing_note='The measured timing wrapper changes only the clock, leaving these spatial poses unchanged.',
        input_sha256={str(path.relative_to(ROOT)):digest(path) for path in paths},
        sampled_csv_sha256=digest(csv_path))
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='input_sha256'},indent=2))


if __name__ == '__main__':
    main()
