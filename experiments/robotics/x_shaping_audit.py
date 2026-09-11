"""Additional reconstruction, replay, contact-geometry, and control checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from experiments.robotics.x_shaping import CONFIG, cylinder_sdf, target_mesh, score
from experiments.robotics.x_shaping_study import OUT, check_sources
from experiments.robotics.hex_shaping_surface import mesh_distance_mm
from experiments.robotics.plastic_shaping_figure import surface
from experiments.robotics.plastic_shaping_study import chamfer_mm, save_json


def audit(folder, planner=None):
    planner = folder if planner is None else planner
    check_sources(planner); rows = []
    target = target_mesh(); fine_target = target.subdivide(1, subfilter="linear")
    sdf = cylinder_sdf(CONFIG["radius"], CONFIG["finger_height"])
    # Independently sample side/end-cap distances at exact lattice sites.
    from scipy.ndimage import map_coordinates
    pts = np.array([[.014, 0, 0], [0, 0, .0225], [.018, 0, 0]])
    values = map_coordinates(sdf.values, ((pts-sdf.origin)/sdf.cell).T, order=1)
    np.testing.assert_allclose(values, [0., 0., .004], atol=1e-7)
    for path in sorted(folder.glob("baseline_*.npz")):
        data = np.load(path); mesh = surface(data["x_after_1s"], data["vol0"], h=CONFIG["voxel"])
        connectivity = mesh.connectivity(extraction_mode="all").compute_cell_sizes(length=False, volume=False)
        ids = connectivity.cell_data["RegionId"]
        areas = np.bincount(ids, weights=connectivity.cell_data["Area"])
        largest_fraction = float(areas.max()/areas.sum())
        assert largest_fraction > .995, (path, largest_fraction)
        record = json.loads(path.with_suffix(".json").read_text())
        points = data["x_after_1s"]
        phases = json.loads(path.with_suffix(".phases.json").read_text())
        rotation_ids = [i for i, p in enumerate(phases) if p["name"].endswith("rotate")]
        rotation_force = np.linalg.norm(data["reaction_force"][np.isin(data["phase_id"], rotation_ids)], axis=-1)
        assert rotation_force.max() == 0., path
        row = dict(file=path.name, material=record["material"], method=record["method"],
                   reconstruction_voxels_mm=[1., 1.25, 1.5],
                   surface_errors_mm=[score(data, voxel=h) for h in [.001, .00125, .0015]],
                   refined_target_quadrature_error_mm=mesh_distance_mm(mesh, fine_target),
                   connected_surface_area_fraction=largest_fraction,
                   peak_force_during_raised_rotation_n=float(rotation_force.max()),
                   centroid_xy_shift_mm=(1000*(points.mean(axis=0)[:2]-CONFIG["domain"]/2)).tolist(),
                   minimum_particle_height_above_floor_mm=float((points[:, 2].min()-CONFIG["floor"])*1000),
                   maximum_pressed_height_mm=max(float((data[f"stage_{i}_pressed"][:, 2].max()-CONFIG["floor"])*1000) for i in range(2)))
        if record["method"] == "oracle":
            selected = json.loads((planner/"plans"/record["model"]/"selected.json").read_text())
            planned = np.load(planner/selected["file"])
            row["independent_replay_difference_mm"] = chamfer_mm(points, planned["x_after_1s"])
            assert row["independent_replay_difference_mm"] < .02
        rows.append(row)
    assert len(rows) == 6
    a = np.load(folder/"baseline_A_nominal.npz"); b = np.load(folder/"baseline_B_nominal.npz")
    for key in ["time", "tool_centers", "phase_id", "gaps_mm"]:
        np.testing.assert_array_equal(a[key], b[key])
    contact_sampling = []
    for path in sorted(folder.glob("*.npz")):
        data = np.load(path)
        depth = np.maximum(float(data["floor"])-data["x_after_1s"][:, 2], 0.)
        contact_sampling.append(dict(file=path.name, grid_spacing_mm=1000*CONFIG["domain"]/int(data["n_grid"]),
              maximum_particle_penetration_mm=float(1000*depth.max()),
              particle_fraction_below_floor=float(np.mean(depth > 0.)),
              volume_weighted_penetration_mm=float(1000*np.average(depth, weights=data["vol0"]))))
    result = dict(results=rows, common_nominal_trajectory="Byte-identical numeric arrays for both true-material executions",
                  target_volume_m3=float(target.volume), target_height_mm=float(1000*(target.bounds.z_max-target.bounds.z_min)),
                  contact_sampling=contact_sampling,
                  contact_note="Plane contact acts on grid velocities; particles can penetrate by a fraction of a cell. No particle clipping or output correction is applied.",
                  note="Surface reconstruction sensitivity and exact physical-volume refinement; not a formal convergence proof")
    save_json(folder/"additional_audit.json", result)
    source = folder/"audit_source"; source.mkdir(exist_ok=True); shutil.copy2(__file__, source/Path(__file__).name)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--planner", type=Path, help="Original planning folder when auditing retimed executions")
    args = p.parse_args(); audit(args.out, args.planner)
