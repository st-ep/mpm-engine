"""Prepare animation meshes using the audited paper surface reconstruction.

The paper's side view is extended upward to include the wrist and forearm.
Particle states, all disconnected droplets, and the numerical model are
preserved. The original paper scene and figure are read-only inputs.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
import hashlib
import json
import time

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.measure import marching_cubes

from experiments.pour import pour_paper_render as still
from warpmpm.geometry.measuring_cup import solid_sdf_local

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out/pour_share_video_20260911"
PAPER = ROOT / "out/pour_figure_final_20260911"
twin = still.reference.twin


def write_json(path, data):
    temp = path.with_suffix(".pending.json")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.rename(path)


def transform(position, rotation):
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, position
    return matrix


def prepare_scene():
    info = json.loads((PAPER / "render_assets/scene.json").read_text())
    geometry = json.loads(still.reference.GEOMETRY.read_text())
    ep = twin.load_episode(still.reference.REFERENCE, twin.PRE_ROLL, twin.HOLD_SECONDS)
    arm = twin.RecordedPanda(ep, still.reference.MESH, height=64, width=64, max_geom=4000,
        cup_reference_pos=geometry["cup_reference_pos"],
        cup_reference_quat=geometry["cup_reference_quat"])
    objects = [o for o in info["objects"] if o["material"] == "robot"]

    def matrices(t):
        p, q = arm.cup_pose_at(t)
        arm.set_time(t)
        result = {o["name"]: transform(arm.data.geom_xpos[int(o["name"].split("_")[1])],
                    arm.data.geom_xmat[int(o["name"].split("_")[1])].reshape(3, 3))
                  for o in objects}
        result["source_cup"] = transform(p, twin.quat_to_mat(q))
        return result

    reference = matrices(info["time_s"])
    inverse = {k: np.linalg.inv(v) for k, v in reference.items()}
    times = [(i+1)/30 for i in range(386)]
    poses = [{k: (v @ inverse[k]).tolist() for k, v in matrices(t).items()} for t in times]
    arm.close()
    scene = dict(paper_scene=str(PAPER / "mpm_scene.blend"),
        paper_scene_sha256=still.digest(PAPER / "mpm_scene.blend"),
        camera=ep["meta"]["cameras"]["side"],
        extrinsics=ep["meta"]["extrinsics"]["cameras"]["side"]["T_base_cam"]["matrix"],
        camera_name="side", camera_mount="static", display_fps=30,
        upright_crop_xyxy=[20, 25, 460, 585], render_scale=1.75,
        frame_count=len(times), host_start_s=ep["t0"],
        appearance="Original paper materials and lighting; blue simulated glycerol",
        poses=poses,
        surface_method="Same 0.65 mm CIC grid, 0.85 mm Gaussian, 0.5 isovalue and sparse-particle union as the paper; cup-boundary clipping has local exceptions around genuinely penetrating particle centers so these are not concealed")
    write_json(OUT / "scene.json", scene)
    return scene


def surface(points, vol, cups):
    spacing, sigma = .00065, .00085
    origin = points.min(0) - 5*sigma
    shape = np.ceil((points.max(0)-origin+5*sigma)/spacing).astype(int)+1
    density = np.zeros(shape, dtype=np.float32)
    u = (points-origin)/spacing
    base = np.floor(u).astype(int)
    fraction = u-base
    for bits in np.ndindex(2, 2, 2):
        index = base+bits
        weights = np.prod(np.where(np.array(bits), fraction, 1-fraction), axis=1)
        np.add.at(density, tuple(index.T), weights*vol/spacing**3)
    density = gaussian_filter(density, sigma=sigma/spacing)
    sparse = map_coordinates(density, u.T, order=1, mode="nearest") < .5
    field = (density-.5)*sigma
    radii = np.cbrt(3*vol[sparse]/(4*np.pi))
    for point, radius in zip(points[sparse], radii, strict=True):
        lo = np.maximum(np.floor((point-radius-origin)/spacing).astype(int), 0)
        hi = np.minimum(np.ceil((point+radius-origin)/spacing).astype(int)+1, shape)
        index = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi, strict=True))
        xyz = np.meshgrid(*[origin[k]+np.arange(lo[k], hi[k])*spacing for k in range(3)], indexing="ij")
        distance = np.sqrt(sum((xyz[k]-point[k])**2 for k in range(3)))
        field[index] = np.maximum(field[index], radius-distance)
    audit = {}
    for name, position, quat in cups:
        rotation = twin.quat_to_mat(quat)
        sdf = solid_sdf_local((points-position) @ rotation, twin.SPEC)
        penetrating = sdf < 0
        audit[name] = dict(raw_particle_minimum_solid_sdf_m=float(sdf.min()),
                           raw_particle_centers_inside_solid=int(penetrating.sum()),
                           handling="Preserve the original density field within volume-equivalent particle-radius neighborhoods of penetrating centers; no particle moves")
        exemptions = None
        if penetrating.any():
            # The full replay occasionally has genuine sub-grid contact overlap.
            # Preserve that evidence rather than clipping those centers away.
            # This relaxes only the display clipping mask: it adds no density.
            exemptions = np.full(shape, -np.inf, dtype=np.float32)
            for point, radius in zip(points[penetrating],
                    np.cbrt(3*vol[penetrating]/(4*np.pi)), strict=True):
                lo = np.maximum(np.floor((point-radius-origin)/spacing).astype(int), 0)
                hi = np.minimum(np.ceil((point+radius-origin)/spacing).astype(int)+1, shape)
                index = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi, strict=True))
                xyz = np.meshgrid(*[origin[k]+np.arange(lo[k], hi[k])*spacing for k in range(3)], indexing="ij")
                distance = np.sqrt(sum((xyz[k]-point[k])**2 for k in range(3)))
                exemptions[index] = np.maximum(exemptions[index], radius-distance)
        for first in range(0, shape[0], 8):
            last = min(first+8, shape[0])
            xyz = np.stack(np.meshgrid(origin[0]+np.arange(first, last)*spacing,
                origin[1]+np.arange(shape[1])*spacing, origin[2]+np.arange(shape[2])*spacing,
                indexing="ij"), axis=-1)
            mask = solid_sdf_local((xyz-position) @ rotation, twin.SPEC)
            if exemptions is not None:
                mask = np.maximum(mask, exemptions[first:last])
            field[first:last] = np.minimum(field[first:last], mask)
        if penetrating.any():
            at_centers = map_coordinates(field, u[penetrating].T, order=1, mode="nearest")
            assert np.all(at_centers >= 0), "Display surface would conceal a penetrating particle"
            audit[name]["minimum_display_field_at_penetrating_center_m"] = float(at_centers.min())
    vertices, faces, _, _ = marching_cubes(field, level=0., spacing=(spacing,)*3)
    vertices += origin
    signed_volume = np.einsum("ij,ij->i", vertices[faces[:, 0]],
        np.cross(vertices[faces[:, 1]], vertices[faces[:, 2]])).sum()/6
    if signed_volume < 0:
        faces = faces[:, ::-1]
    return vertices, faces, dict(particles=len(points), particle_selection="all",
        component_filtering="none", mesh_smoothing="none", sparse_particles=int(sparse.sum()),
        vertices=len(vertices), faces=len(faces), clipping_audit=audit)


def prepare_frame(index, poses):
    started = time.monotonic()
    path = OUT / f"capture/states/state_{index:04d}.npz"
    data = dict(np.load(path))
    common = dict(np.load(OUT / "capture/particles.npz"))
    cups = [("source", data["source_pos"], data["source_quat"]),
            ("receiver", common["receiver_pos"], common["receiver_quat"])]
    vertices, faces, audit = surface(data["x_world"], common["vol"], cups)
    mesh = OUT / f"meshes/liquid_{index:04d}.ply"
    still.write_ply(mesh, vertices, faces)
    write_json(OUT / f"frames/frame_{index:04d}.json", dict(index=index,
        t=float(data["t"]), t_host=float(data["t_host"]),
        replay_frame=int(data["replay_frame"]), mesh=str(mesh),
        state_sha256=still.digest(path), poses=poses, surface=audit,
        preparation_seconds=time.monotonic()-started))
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    for name in ("meshes", "frames", "rendered"):
        (OUT / name).mkdir(exist_ok=True)
    scene = prepare_scene()
    pending = {}
    index = 0
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while completed < scene["frame_count"]:
            while index < scene["frame_count"] and len(pending) < args.workers*2:
                path = OUT / f"capture/states/state_{index:04d}.npz"
                if not path.exists():
                    break
                if (OUT / f"frames/frame_{index:04d}.json").exists():
                    completed += 1
                else:
                    pending[pool.submit(prepare_frame, index, scene["poses"][index])] = index
                index += 1
            for future in list(pending):
                if future.done():
                    value = future.result()
                    del pending[future]
                    completed += 1
                    if completed % 10 == 0 or completed == scene["frame_count"]:
                        print(f"Prepared {completed}/{scene['frame_count']} frames; latest {value}", flush=True)
            if completed < scene["frame_count"]:
                time.sleep(.5)
    write_json(OUT / "meshes_complete.json", dict(frames=completed, status="complete"))


if __name__ == "__main__":
    main()
