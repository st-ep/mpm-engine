"""Check camera extrinsics against the recorded tool AprilTag and robot poses.

Uses the supplied 30 mm tag size and tag-to-EEF transform, conditionally. Fits
camera extrinsics on alternating tag detections and evaluates unused detections.
No material surface or desired material response enters calibration.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from experiments.robotics.press_hardware_observe import read, jsonl, save, to_base, deproject


def collect(ep, cam):
    meta = read(ep / 'meta.json')
    states = jsonl(ep / 'states.jsonl')
    t = np.array([s['t'] for s in states])
    xyz = np.array([s['state']['ee_pos'] for s in states])
    rotation = Slerp(t, Rotation.from_quat([s['state']['ee_quat_xyzw'] for s in states]))
    frames = jsonl(ep / f'frames_{cam}.jsonl')
    camera = meta['cameras'][cam]
    intr = camera['intrinsics']
    K = np.array([[intr['fx'], 0, intr['ppx']], [0, intr['fy'], intr['ppy']], [0, 0, 1.]])
    T_ee_tag = np.array(meta['extrinsics']['T_ee_tag']['matrix'])
    size = meta['extrinsics']['tag']['size_m']
    corners_tag = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]]) * size / 2
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11))
    cap = cv2.VideoCapture(str(ep / f'{cam}_rgb.mp4'))
    observations = []
    for index in range(0, len(frames), 10):
        host = frames[index]['t_host']
        if not t[0] <= host <= t[-1]:
            continue
        cap.set(1, index)
        ok, image = cap.read()
        assert ok
        corners, ids, _ = detector.detectMarkers(image)
        if ids is None:
            continue
        for pixels, tag_id in zip(corners, ids.ravel()):
            T_base_ee = np.eye(4)
            T_base_ee[:3, :3] = rotation(host).as_matrix()
            T_base_ee[:3, 3] = [np.interp(host, t, xyz[:, d]) for d in range(3)]
            T_base_tag = T_base_ee @ T_ee_tag
            world = to_base(corners_tag, T_base_tag)
            observations.append(dict(index=index, time_host=host, id=int(tag_id),
                                     pixels=pixels.reshape(4, 2).tolist(), world=world.tolist(),
                                     T_base_tag=T_base_tag.tolist()))
    cap.release()
    return observations, K, meta


def project(world, transform, K):
    camera = (world - transform[:3, 3]) @ transform[:3, :3]
    uv = camera @ K.T
    return uv[:, :2] / uv[:, 2, None]


def calibrate(root, out, episodes):
    result = dict(source_episodes=episodes, conditional_inputs='Recorded tag size 30 mm, supplied T_ee_tag and robot EEF poses; tag geometry not independently measured here', cameras={})
    for cam in ['side', 'hand']:
        obs = []
        for name in episodes:
            records, K, meta = collect(root / name, cam)
            obs.extend([dict(episode=name, **r) for r in records])
        assert len(obs) >= 8, (cam, len(obs))
        world = np.array([o['world'] for o in obs])
        pixels = np.array([o['pixels'] for o in obs])
        train = np.arange(len(obs)) % 3 != 0
        # Direct PnP avoids assuming that the existing extrinsic is close.
        ok, rv, tv, inliers = cv2.solvePnPRansac(world[train].reshape(-1, 3), pixels[train].reshape(-1, 2), K, None,
                                               flags=cv2.SOLVEPNP_ITERATIVE, reprojectionError=3., iterationsCount=1000)
        assert ok
        def residual(v, selected):
            points = world[selected].reshape(-1, 3)
            q = points @ Rotation.from_rotvec(v[:3]).as_matrix().T + v[3:]
            uv = q @ K.T
            return (uv[:, :2] / uv[:, 2, None] - pixels[selected].reshape(-1, 2)).ravel()
        fit = least_squares(lambda v: residual(v, train), np.r_[rv.ravel(), tv.ravel()], loss='soft_l1', f_scale=1.)
        T_cam_base = np.eye(4)
        T_cam_base[:3, :3] = Rotation.from_rotvec(fit.x[:3]).as_matrix()
        T_cam_base[:3, 3] = fit.x[3:]
        T = np.linalg.inv(T_cam_base)
        original = np.array(meta['extrinsics']['cameras'][cam]['T_base_cam']['matrix'])
        old_error = np.linalg.norm(project(world.reshape(-1, 3), original, K) - pixels.reshape(-1, 2), axis=1)
        new_error = np.linalg.norm(project(world.reshape(-1, 3), T, K) - pixels.reshape(-1, 2), axis=1).reshape(-1, 4)
        depth_error = []
        for o in obs:
            ep = root / o['episode']
            if (o['episode'], cam) in {('ep0005', 'side'), ('ep0010', 'side')}:
                continue
            d = cv2.imread(str(ep / f'{cam}_depth/{o["index"]:06}.png'), -1)
            mask = np.zeros(d.shape, np.uint8)
            cv2.fillConvexPoly(mask, np.rint(o['pixels']).astype(int), 1)
            mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
            y, x = np.nonzero(mask & (d > 100) & (d < 1000))
            if len(x) < 10:
                continue
            points = to_base(deproject(np.column_stack((x, y)), d[y, x] * meta['cameras'][cam]['depth_scale'], meta['cameras'][cam]), T)
            tag = np.array(o['T_base_tag'])
            distance = (points - tag[:3, 3]) @ tag[:3, 2]
            depth_error.append(dict(episode=o['episode'], index=o['index'], median_plane_distance_mm=float(np.median(distance) * 1000), samples=len(x)))
        result['cameras'][cam] = dict(T_base_cam=T.tolist(), detections=len(obs), tag_ids=sorted(set(o['id'] for o in obs)),
                                     train_rms_px=float(np.sqrt(np.mean(new_error[train] ** 2))),
                                     withheld_rms_px=float(np.sqrt(np.mean(new_error[~train] ** 2))),
                                     withheld_max_px=float(new_error[~train].max()),
                                     original_rms_px=float(np.sqrt(np.mean(old_error ** 2))),
                                     original_translation_change_mm=float(np.linalg.norm(T[:3, 3] - original[:3, 3]) * 1000),
                                     depth_plane_checks=depth_error, observations=obs)
        print(cam, {k: v for k, v in result['cameras'][cam].items() if k not in ['T_base_cam', 'observations', 'depth_plane_checks']}, flush=True)
    save(out / 'tag_calibration.json', result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--episodes', nargs='+', default=['ep0001', 'ep0009'])
    a = p.parse_args()
    cv2.setNumThreads(1)
    calibrate(a.root, a.out, a.episodes)
