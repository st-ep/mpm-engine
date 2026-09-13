"""Prepare calibrated-camera display meshes from an actual MPM snapshot.

The density is reconstructed solely for rendering; recorded particle states and
volume measurements are untouched. All particles contribute, including spill.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out/pour_figure_final_20260911"
sys.path.insert(0, str(ROOT / "out/pour_physics_audit/aflip_blend_time_20260907/isolated_src"))
from experiments.pour import pour_transfer_recalibration_reference as reference


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_ply(path, vertices, faces):
    """Small binary PLY writer; avoids changing the shared Python environment."""
    vertices, faces = np.asarray(vertices, dtype="<f4"), np.asarray(faces, dtype="<i4")
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\n"
              f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
    records = np.empty(len(faces), dtype=[("count", "u1"), ("indices", "<i4", (3,))])
    records["count"], records["indices"] = 3, faces
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        vertices.tofile(f)
        records.tofile(f)


def cup_display_mesh(path):
    """Keep the frozen OBJ's surfaces; repair topology/normals on a display copy.

    Its first two bottom rings coincide and create zero-area triangles, and
    the disconnected handle has inward winding. Neither affects the analytic
    collider, but both are inappropriate for a dielectric path-traced mesh.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    vertices, faces = [], []
    for line in path.read_text().splitlines():
        words = line.split()
        if words and words[0] == "v":
            vertices.append([float(v) for v in words[1:4]])
        elif words and words[0] == "f":
            face = [int(v.split("/")[0]) for v in words[1:]]
            face = [i-1 if i > 0 else len(vertices)+i for i in face]
            faces.extend([[face[0], face[j], face[j+1]] for j in range(1, len(face)-1)])
    original = np.asarray(vertices)
    vertices, inverse = np.unique(original, axis=0, return_inverse=True)
    faces = inverse[np.asarray(faces)]
    area2 = np.linalg.norm(np.cross(vertices[faces[:, 1]]-vertices[faces[:, 0]],
                                   vertices[faces[:, 2]]-vertices[faces[:, 0]]), axis=1)
    degenerate = area2 < 1e-18
    faces = faces[~degenerate]
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    edge, count = np.unique(edges, axis=0, return_counts=True)
    assert np.all(count == 2), "Cup display components must be closed"
    graph = coo_matrix((np.ones(len(edge)), (edge[:, 0], edge[:, 1])),
                       shape=(len(vertices), len(vertices)))
    n, labels = connected_components(graph, directed=False)
    volumes, flipped = [], []
    for component in range(n):
        which = labels[faces[:, 0]] == component
        tri = faces[which]
        volume = np.einsum("ij,ij->i", vertices[tri[:, 0]],
            np.cross(vertices[tri[:, 1]], vertices[tri[:, 2]])).sum()/6
        volumes.append(float(volume*1e6))
        if volume < 0:
            faces[which] = faces[which, ::-1]
            flipped.append(component)
    np.testing.assert_array_equal(vertices.min(0), original.min(0))
    np.testing.assert_array_equal(vertices.max(0), original.max(0))
    return vertices, faces, dict(welded_duplicate_vertices=len(original)-len(vertices),
        zero_area_faces_removed=int(degenerate.sum()), components=n,
        original_component_signed_volumes_ml=volumes, reversed_components=flipped,
        components_closed=True, vertex_positions_modified=False,
        measured_dimensions_modified=False)


def prepare(static_only=False):
    from PIL import Image
    from scipy.ndimage import gaussian_filter, map_coordinates
    from skimage.measure import marching_cubes
    from warpmpm.geometry.measuring_cup import solid_sdf_local

    assets = OUT / "render_assets"
    assets.mkdir(exist_ok=True)
    twin = reference.twin
    geometry = json.loads(reference.GEOMETRY.read_text())
    ep = twin.load_episode(reference.REFERENCE, twin.PRE_ROLL, twin.HOLD_SECONDS)
    time_s = json.loads((OUT / "replay/provenance.json").read_text())["snapshot_time_s"]
    arm = twin.RecordedPanda(ep, reference.MESH, height=64, width=64, max_geom=4000,
        cup_reference_pos=geometry["cup_reference_pos"],
        cup_reference_quat=geometry["cup_reference_quat"])
    p, q = arm.cup_pose_at(time_s)
    receiver_pos = np.r_[geometry["receiver_xy"], geometry["table_z"]]
    arm.set_glass_pose("glass_src", p, q)
    arm.set_glass_pose("glass_rcv", receiver_pos, twin.Q_RCV)
    arm.set_time(time_s)
    m, d = arm.model, arm.data
    objects = []
    for gid in range(m.ngeom):
        # Only visual robot meshes; the collision hulls would obscure the hand.
        if m.geom_group[gid] != 2:
            continue
        assert int(m.geom_type[gid]) == 7
        mid = m.geom_dataid[gid]
        va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        vertices = m.mesh_vert[va:va+vn] @ d.geom_xmat[gid].reshape(3, 3).T + d.geom_xpos[gid]
        name = f"robot_{gid:02d}"
        write_ply(assets / f"{name}.ply", vertices, m.mesh_face[fa:fa+fn])
        material = m.geom_matid[gid]
        rgba = m.mat_rgba[material] if material >= 0 else m.geom_rgba[gid]
        objects.append(dict(name=name, file=f"{name}.ply", material="robot", rgba=rgba.tolist()))
    # Repair only the display copy, preserving every nondegenerate surface and
    # all measured dimensions. The frozen OBJ and analytic collider stay intact.
    cup_vertices, cup_faces, cup_repair = cup_display_mesh(reference.MESH)
    for name, position, quat in [("source_cup", p, q), ("receiver_cup", receiver_pos, twin.Q_RCV)]:
        vertices = cup_vertices @ twin.quat_to_mat(quat).T + position
        write_ply(assets / f"{name}.ply", vertices, cup_faces)
        objects.append(dict(name=name, file=f"{name}.ply", material="cup"))
    arm.close()
    info = dict(time_s=time_s, t_host=ep["t0"]+time_s,
        camera=ep["meta"]["cameras"]["side"],
        extrinsics=ep["meta"]["extrinsics"]["cameras"]["side"]["T_base_cam"]["matrix"],
        table_z=geometry["table_z"], objects=objects,
        geometry_sha256=digest(reference.GEOMETRY), cup_mesh_sha256=digest(reference.MESH),
        cup_display_mesh_repair=cup_repair,
        appearance="Blue simulated liquid for visibility; clear cups; neutral lighting. No geometry fitted to a photograph.",
        upright_crop_xyxy=[45, 150, 445, 575], render_scale=3)
    if not static_only:
        snapshot = OUT / "replay/snapshot.npz"
        data = np.load(snapshot)
        np.testing.assert_allclose(p, data["source_pos"], atol=1e-10)
        np.testing.assert_allclose(q, data["source_quat"], atol=1e-10)
        assert abs(float(data["t"])-time_s) < 1e-8
        points = data["x_world"].astype(float)
        # Cloud-in-cell splat at 0.65 mm, Gaussian sigma 0.85 mm, then the
        # half-density isosurface. Preserve every component, including droplets.
        spacing, sigma_m = .00065, .00085
        origin = points.min(0) - 5*sigma_m
        shape = np.ceil((points.max(0)-origin+5*sigma_m)/spacing).astype(int)+1
        density = np.zeros(shape, dtype=np.float32)
        u = (points-origin)/spacing
        base = np.floor(u).astype(int)
        frac = u-base
        for bits in np.ndindex(2, 2, 2):
            index = base+bits
            weights = np.prod(np.where(np.array(bits), frac, 1-frac), axis=1)
            np.add.at(density, tuple(index.T), weights*data["vol"]/spacing**3)
        density = gaussian_filter(density, sigma=sigma_m/spacing)
        # Form one implicit union before tessellation. Separate overlapping
        # spheres previously introduced artificial seams/particle texture on
        # the pool. Retain exactly the same low-density particles and radii.
        sparse = map_coordinates(density, u.T, order=1, mode="nearest") < .5
        field = (density-.5)*sigma_m  # positive inside the display liquid
        radii = np.cbrt(3*data["vol"][sparse]/(4*np.pi))
        for point, radius in zip(points[sparse], radii, strict=True):
            lo = np.maximum(np.floor((point-radius-origin)/spacing).astype(int), 0)
            hi = np.minimum(np.ceil((point+radius-origin)/spacing).astype(int)+1, shape)
            index = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi, strict=True))
            xyz = np.meshgrid(*[origin[k]+np.arange(lo[k], hi[k])*spacing for k in range(3)], indexing="ij")
            distance = np.sqrt(sum((xyz[k]-point[k])**2 for k in range(3)))
            field[index] = np.maximum(field[index], radius-distance)
        # Kernel support is not a physical liquid extension into the plastic.
        # Intersect the display field with the exterior of the exact frozen
        # analytic cup solids, equally for source and receiver. This does not
        # move particles or modify any measured/predicted volume.
        clipping_audit = {}
        for name, position, quat in [("source", p, q), ("receiver", receiver_pos, twin.Q_RCV)]:
            rotation = twin.quat_to_mat(quat)
            sdf = solid_sdf_local((points-position) @ rotation, twin.SPEC)
            assert np.all(sdf >= 0), "Do not conceal actual particle penetration with display clipping"
            clipping_audit[name] = dict(raw_particle_minimum_solid_sdf_m=float(sdf.min()),
                raw_particle_centers_inside_solid=int((sdf < 0).sum()))
            for first in range(0, shape[0], 8):
                last = min(first+8, shape[0])
                xyz = np.stack(np.meshgrid(origin[0]+np.arange(first, last)*spacing,
                    origin[1]+np.arange(shape[1])*spacing,
                    origin[2]+np.arange(shape[2])*spacing, indexing="ij"), axis=-1)
                sdf = solid_sdf_local((xyz-position) @ rotation, twin.SPEC)
                field[first:last] = np.minimum(field[first:last], sdf)
        verts, faces, _, _ = marching_cubes(field, level=0., spacing=(spacing,)*3)
        verts += origin
        signed_volume = np.einsum("ij,ij->i", verts[faces[:, 0]],
            np.cross(verts[faces[:, 1]], verts[faces[:, 2]])).sum()/6
        if signed_volume < 0:
            faces = faces[:, ::-1]
        write_ply(assets / "liquid.ply", verts, faces)
        objects.append(dict(name="liquid", file="liquid.ply", material="liquid"))
        info["surface_reconstruction"] = dict(snapshot_sha256=digest(snapshot),
            particles=len(points), spacing_m=spacing, gaussian_sigma_m=sigma_m,
            density_isovalue=.5, mesh_vertices=len(verts), mesh_faces=len(faces),
            particle_selection="all", mesh_smoothing="none", component_filtering="none",
            sparse_particle_spheres=int(sparse.sum()),
            sparse_particle_rule="Density below 0.5 at particle center: sphere of volume-equivalent radius",
            sparse_particle_tessellation="Implicit union with density surface before marching cubes; no overlapping shell objects",
            cup_boundary="Implicit intersection with exterior of both frozen true-dimension analytic cup solids",
            clipping_audit=clipping_audit,
            purpose="Display only; numerical volumes come from particle counts, never the render mesh")
    # Synchronize by host timestamps, rather than assuming the MP4 starts at meta.created.
    frame_rows = [json.loads(line) for line in (reference.REFERENCE / "frames_side.jsonl").read_text().splitlines()]
    matched = min(frame_rows, key=lambda r: abs(r["t_host"]-info["t_host"]))
    delta = matched["t_host"]-info["t_host"]
    assert abs(delta) <= 1/60
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(reference.REFERENCE / "side_rgb.mp4"),
        "-vf", f"select=eq(n\\,{matched['frame_idx']})", "-frames:v", "1",
        str(OUT / "hardware_raw.png")], check=True)
    raw = Image.open(OUT / "hardware_raw.png")
    upright = raw.transpose(Image.Transpose.ROTATE_90)
    upright.save(OUT / "hardware_upright.png")
    upright.crop(tuple(info["upright_crop_xyxy"])).save(OUT / "hardware_crop.png")
    info["hardware_frame"] = dict(frame_index=matched["frame_idx"],
        t_host=matched["t_host"], time_difference_s=delta,
        video_sha256=digest(reference.REFERENCE / "side_rgb.mp4"),
        transform="90 degrees counterclockwise, then the same crop used for the rendered view")
    (assets / "scene.json").write_text(json.dumps(info, indent=2)+"\n")
    print(json.dumps({k:v for k,v in info.items() if k not in ("objects", "camera", "extrinsics")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-only", action="store_true")
    prepare(parser.parse_args().static_only)
