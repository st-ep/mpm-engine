"""Conditional stereo calibration from the common tool tag, independent of robot offsets.

Fits ep0001 and checks ep0009. This remains conditional on the recorded 30 mm
tag size and camera intrinsics; it does not calibrate depth scale or robot pose.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from experiments.robotics.press_hardware_observe import read, jsonl, save, to_base


def check(root, out):
    original = read(out / 'tag_calibration.json')
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11))
    pairs = []
    for name in ['ep0001', 'ep0009']:
        ep = root / name
        meta = read(ep / 'meta.json')
        frames = jsonl(ep / 'frames_hand.jsonl')
        times = np.array([f['t_host'] for f in frames])
        cap = cv2.VideoCapture(str(ep / 'hand_rgb.mp4'))
        side = [o for o in original['cameras']['side']['observations'] if o['episode'] == name and o['id'] == 1]
        for o in side:
            index = int(np.argmin(abs(times - o['time_host'])))
            cap.set(1, index)
            ok, image = cap.read()
            assert ok
            corners, ids, _ = detector.detectMarkers(image)
            if ids is None or 1 not in ids:
                continue
            ix = int(np.flatnonzero(ids.ravel() == 1)[0])
            pairs.append(dict(episode=name, side_index=o['index'], hand_index=index,
                time_gap_ms=float((times[index] - o['time_host']) * 1000),
                side_pixels=o['pixels'], hand_pixels=corners[ix].reshape(4, 2).tolist()))
        cap.release()
    K = []
    for cam in ['side', 'hand']:
        k = meta['cameras'][cam]['intrinsics']
        K.append(np.array([[k['fx'], 0, k['ppx']], [0, k['fy'], k['ppy']], [0, 0, 1.]]))
    size = meta['extrinsics']['tag']['size_m']
    obj = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], np.float32) * size / 2
    train = [p for p in pairs if p['episode'] == 'ep0001']
    assert len(train) >= 4, len(train)
    rms, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate([obj] * len(train),
        [np.array(p['side_pixels'], np.float32) for p in train],
        [np.array(p['hand_pixels'], np.float32) for p in train], K[0], None, K[1], None, (848, 480),
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-9))
    # Epipolar error avoids using a test-pair hand image to estimate tag pose.
    tx, ty, tz = T.ravel()
    cross = np.array([[0, -tz, ty], [tz, 0, -tx], [-ty, tx, 0]])
    F = np.linalg.inv(K[1]).T @ cross @ R @ np.linalg.inv(K[0])
    for p in pairs:
        a = np.c_[p['side_pixels'], np.ones(4)]
        b = np.c_[p['hand_pixels'], np.ones(4)]
        line = a @ F.T
        p['epipolar_rms_px'] = float(np.sqrt(np.mean((np.sum(b * line, 1) / np.linalg.norm(line[:, :2], axis=1)) ** 2)))
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = R, T.ravel()
    center_checks = []
    # Diagnose an overall metric-scale discrepancy using initial geometry only.
    # This cannot distinguish tag-size error from depth-scale/shape-fit error.
    camera_centers = {}
    for i in range(12):
        name = f'ep{i:04}'
        a = read(out / name / 'assessment.json')
        if any(a['cameras'][cam]['initial_sphere'] is None for cam in ['side', 'hand']):
            continue
        meta = read(root / name / 'meta.json')
        center = []
        for cam in ['side', 'hand']:
            c = np.array(a['cameras'][cam]['initial_sphere']['center_base_m'])
            old = np.array(meta['extrinsics']['cameras'][cam]['T_base_cam']['matrix'])
            center.append(to_base(c[None], np.linalg.inv(old))[0])
        camera_centers[name] = center
        gap = np.linalg.norm(to_base(center[0][None], transform)[0] - center[1]) * 1000
        center_checks.append(dict(episode=name, sphere_center_gap_mm=float(gap)))
    a, b = camera_centers['ep0001']
    translation = T.ravel()
    factor = float(translation @ (b - R @ a) / (translation @ translation))
    scale_checks = [dict(episode=name,
        initial_center_residual_mm=float(np.linalg.norm(R @ a + factor * translation - b) * 1000))
        for name, (a, b) in camera_centers.items()]
    result = dict(train_episode='ep0001', test_episode='ep0009', train_pairs=len(train),
        test_pairs=len(pairs) - len(train), train_stereo_rms_px=float(rms),
        train_epipolar_rms_px=float(np.sqrt(np.mean([p['epipolar_rms_px'] ** 2 for p in train]))),
        test_epipolar_rms_px=float(np.sqrt(np.mean([p['epipolar_rms_px'] ** 2 for p in pairs if p['episode'] == 'ep0009']))),
        T_hand_side=transform.tolist(), sphere_center_checks=center_checks, pairs=pairs,
        metric_scale_diagnostic=dict(fit_episode='ep0001', translation_scale=factor,
            equivalent_tag_size_mm=float(size * 1000 * factor), initial_geometry_checks=scale_checks,
            scope='Scale inferred from approximate initial sphere centers and supplied depth units. NOT a tag measurement or accepted calibration. Cannot distinguish tag-size, depth-scale, and shape-fit errors.'),
        scope='Conditional trial using recorded 30 mm tag, intrinsics, and nearest host-time RGB pairs. Does not validate absolute robot alignment or depth scale; not applied to exported observations.')
    save(out / 'stereo_tag_check.json', result)
    print({k: v for k, v in result.items() if k not in ['pairs', 'T_hand_side']}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    cv2.setNumThreads(1)
    check(a.root, a.out)
