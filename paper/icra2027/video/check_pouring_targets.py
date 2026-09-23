"""Validate six rendered replays against the exact frozen target plans."""
from pathlib import Path
import csv,hashlib,json,subprocess
import cv2
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
WORK=HERE/'assets/pouring_targets'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def check():
    commands=list(csv.DictReader((ROOT/'out/pour_hardware_receiver_remap_review_20260911/verified_commands.csv').open()))[:6]
    report=[]
    for r in commands:
        folder=WORK/f'target_{int(r["target_ml"])}'
        frozen_path=ROOT/r['result_path'];replay_path=next((folder/'forward').rglob('result.json'))
        frozen=json.loads(frozen_path.read_text());replay=json.loads(replay_path.read_text())
        verified=json.loads((folder/'verification.json').read_text());assert verified['accepted']
        for key in ['eta_pa_s','source_coulomb_friction','n_grid','dt_scale','planned_angle_deg','particle_count']:
            assert replay[key]==frozen[key],(r['target_ml'],key)
        source=frozen_path.parent/'planned_episode';dest=replay_path.parent/'planned_episode'
        for name in ['meta.json','actions.jsonl','states.jsonl','joint_trajectory.csv']:
            assert sha(source/name)==sha(dest/name),(r['target_ml'],name)
        a=np.genfromtxt(source/'metrics.csv',names=True,delimiter=',')
        b=np.genfromtxt(dest/'metrics.csv',names=True,delimiter=',')
        assert len(a)==len(b)
        np.testing.assert_array_equal(a['t'],b['t'])
        np.testing.assert_array_equal(a['tilt_deg'],b['tilt_deg'])
        expected=np.linspace(0,len(a)-1,50).round().astype(int)
        for i,n in enumerate(expected):
            with np.load(folder/f'capture/states/state_{i:04d}.npz') as state:
                assert int(state['replay_frame'])==n
                assert np.isfinite(state['x_world']).all()
        path=folder/'target.mp4';cap=cv2.VideoCapture(str(path))
        assert cap.isOpened() and cap.get(cv2.CAP_PROP_FPS)==25
        assert cap.get(cv2.CAP_PROP_FRAME_COUNT)==50
        assert (cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT))==(480,394)
        cap.release()
        decoded=subprocess.run(['ffmpeg','-v','error','-i',str(path),'-f','null','-'],capture_output=True,text=True)
        assert decoded.returncode==0 and not decoded.stderr
        report.append({'target_ml':int(r['target_ml']),'angle_deg':float(r['command_angle_deg']),
            'frozen_robot_trajectory_identical':True,'frozen_parameters_identical':True,
            'physics_frames':len(a),'video_frames':50,'video_seconds':2,
            'maximum_volume_history_difference_ml':max(verified['max_history_difference_ml'].values()),
            'endpoint_difference_ml':verified['endpoint_difference_ml'],
            'clip':str(path.relative_to(ROOT)),'clip_sha256':sha(path),
            'verification':str((folder/'verification.json').relative_to(ROOT)),
            'verification_sha256':sha(folder/'verification.json')})
    (WORK/'validated_clips.json').write_text(json.dumps({'passed':True,'cases':report},indent=2)+'\n')
    return report


if __name__=='__main__':
    for r in check():print(r['target_ml'],'mL: PASS; endpoint deviation',round(r['endpoint_difference_ml'],4),'mL')
