"""Bounded affine-patch tracking check on real RGB images, without a material model.

Uses the repository's affine DIC routine on unrectified camera images. Accepted
correlations remain candidate correspondences, not validated material identities.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from experiments.elastic.strip_texture_dic import correlate, sample
from experiments.robotics.press_hardware_observe import (
    read, save, segment, depth_at, deproject, to_base, FROZEN, CAMERA_ROI,
)


def check(root, out, episode, cam):
    dest = out / episode
    meta = read(root / episode / 'meta.json')
    audit = read(dest / 'assessment.json')
    obs = np.load(dest / f'{cam}_observations.npz')
    choose = np.flatnonzero(obs['time'] <= 10.1)
    times, indices = obs['time'][choose], obs['frame_idx'][choose]
    camera = meta['cameras'][cam]
    transform = np.array(meta['extrinsics']['cameras'][cam]['T_base_cam']['matrix'])
    frozen = (episode, cam) in FROZEN
    cap = cv2.VideoCapture(str(root / episode / f'{cam}_rgb.mp4'))
    radius = 7
    offsets = np.stack(np.meshgrid(np.arange(-radius, radius + 1),
                                   np.arange(-radius, radius + 1)), -1).reshape(-1, 2)
    pixels, positions, tiles = [], [], []
    for k, idx in enumerate(indices):
        cap.set(1, int(idx))
        ok, image = cap.read()
        assert ok
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        depth = cv2.imread(str(root / episode / f'{cam}_depth/{idx:06}.png'), -1)
        mask, _ = segment(image, None if frozen else depth, cam, audit['material'])
        if k == 0:
            reference = gray.copy()
            corners = cv2.goodFeaturesToTrack(gray, 200, .01, 5,
                        mask=cv2.erode(mask, np.ones((17, 17), np.uint8)) * 255, blockSize=7)
            seeds = np.empty((0, 2), np.float32) if corners is None else corners[:, 0]
            current = seeds.copy()
            matrices = np.tile(np.eye(2), (len(seeds), 1, 1))
            alive = np.ones(len(seeds), bool)
            ref_std = sample(reference, seeds[:, None, :] + offsets[None]).std(1)
        elif alive.any():
            ids = np.flatnonzero(alive)
            src = current[ids].astype('float32').reshape(-1, 1, 2)
            nxt, ok, _ = cv2.calcOpticalFlowPyrLK(previous, gray, src, None, winSize=(21, 21), maxLevel=3)
            back, okb, _ = cv2.calcOpticalFlowPyrLK(gray, previous, nxt, None, winSize=(21, 21), maxLevel=3)
            good = ok[:, 0].astype(bool) & okb[:, 0].astype(bool)
            good &= np.linalg.norm(back[:, 0] - src[:, 0], axis=1) < .7
            refined, affine, error = correlate(reference, gray, seeds[ids], nxt[:, 0], matrices[ids], radius=radius)
            sv = np.linalg.svd(affine, compute_uv=False)
            good &= (np.linalg.det(affine) > 0) & (sv.min(1) >= .35) & (sv.max(1) <= 3)
            good &= error / np.maximum(ref_std[ids], 1e-9) <= np.sqrt(1 - .7 ** 2)
            good &= np.linalg.norm(refined - nxt[:, 0], axis=1) <= 3
            coords = np.einsum('nij,pj->npi', affine, offsets) + refined[:, None]
            within = (coords[..., 0] >= 0) & (coords[..., 0] < gray.shape[1] - 1)
            within &= (coords[..., 1] >= 0) & (coords[..., 1] < gray.shape[0] - 1)
            ix = np.clip(np.rint(coords[..., 0]).astype(int), 0, gray.shape[1] - 1)
            iy = np.clip(np.rint(coords[..., 1]).astype(int), 0, gray.shape[0] - 1)
            good &= np.mean(within & (mask[iy, ix] != 0), axis=1) >= .9
            alive[:] = False
            alive[ids[good]] = True
            current[ids] = refined
            matrices[ids] = affine
        xy = np.full((len(seeds), 2), np.nan, np.float32)
        xyz = np.full((len(seeds), 3), np.nan, np.float32)
        xy[alive] = current[alive]
        ids = np.flatnonzero(alive)
        if len(ids) and not frozen:
            d = depth_at(current[ids], depth, camera['depth_scale'])
            valid = np.isfinite(d)
            xyz[ids[valid]] = to_base(deproject(current[ids[valid]], d[valid], camera), transform)
        pixels.append(xy)
        positions.append(xyz)
        if k in [int(np.argmin(abs(times - t))) for t in [0, 1, 5, 10]]:
            cx, _ = CAMERA_ROI[cam]
            for uv in xy[alive]:
                cv2.circle(image, tuple(np.rint(uv).astype(int)), 2, (0, 255, 255), -1)
            tile = image[95:325, cx - 115:cx + 115].copy()
            cv2.putText(tile, f'{times[k]:.2f}s: {alive.sum()} candidates', (3, 18), 0, .43, (0, 255, 255), 1)
            tiles.append(tile)
        previous = gray
    cap.release()
    pixels, positions = np.asarray(pixels), np.asarray(positions)
    np.savez_compressed(dest / f'{cam}_affine_candidates.npz', time=times, pixels=pixels, positions_base=positions)
    cv2.imwrite(str(dest / f'{cam}_affine_candidates.jpg'), np.hstack(tiles))
    result = dict(episode=episode, camera=cam, initial_candidates=len(seeds),
        candidates_2d={str(t): int(np.isfinite(pixels[np.argmin(abs(times - t))]).all(-1).sum()) for t in [0, 1, 3, 5, 10]},
        candidates_3d={str(t): int(np.isfinite(positions[np.argmin(abs(times - t))]).all(-1).sum()) for t in [0, 1, 3, 5, 10]},
        scope='Fixed-reference 15x15 affine DIC; LK forward/backward <0.7px; equivalent NCC >=0.7; positive determinant; singular values 0.35..3; correction <=3px; >=90% patch in specimen mask. No respawning. Candidate identities only.')
    save(dest / f'{cam}_affine_check.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--episodes', nargs='+', default=['ep0001', 'ep0004', 'ep0009'])
    a = p.parse_args()
    cv2.setNumThreads(1)
    for episode in a.episodes:
        for cam in ['side', 'hand']:
            check(a.root, a.out, episode, cam)
