"""Geometry checks and paper figure for the frozen A/B plan cross-executions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pyvista as pv

from experiments.robotics.x_letter_cross import OUT, check
from experiments.robotics.x_letter_study import digest, score
from experiments.robotics.x_letter_study_report import render, report
from experiments.robotics.plastic_shaping_figure import surface
from experiments.robotics.plastic_shaping_study import save_json
from experiments.robotics.hex_shaping_surface import mesh_distance_mm


def audit(folder):
    check(folder)
    summary = json.loads((folder / "summary.json").read_text())
    target = pv.read(folder / "target.vtp")
    assert target.n_open_edges == 0
    np.testing.assert_allclose(target.volume, 0.00009, rtol=2e-6)
    target_mask = render(target, mask=True, top=True)[..., :3].mean(-1) < 128
    rows = []
    for record in summary["results"]:
        path = folder / record["file"]
        data = np.load(path)
        mesh = surface(data["x_after_1s"], data["vol0"], h=0.00125)
        top = render(mesh, mask=True, top=True)[..., :3].mean(-1) < 128
        oblique = render(mesh, mask=True)[..., :3].mean(-1) < 128
        for mask in [top, oblique]:
            assert not mask[[0, -1], :].any() and not mask[:, [0, -1]].any(), path
        connected = mesh.connectivity(extraction_mode="all").compute_cell_sizes(length=False, volume=False)
        areas = np.bincount(connected.cell_data["RegionId"], weights=connected.cell_data["Area"])
        penetration = np.maximum(float(data["floor"]) - data["x_after_1s"][:, 2], 0)
        phases = json.loads(path.with_suffix(".phases.json").read_text())
        turns = [i for i, p in enumerate(phases) if p["name"].endswith("rotate")]
        force = data["reaction_force"][np.isin(data["phase_id"], turns)]
        assert np.max(np.abs(force)) == 0
        assert int(data["inverted_count"]) == 0
        np.testing.assert_allclose(score(data, target), record["surface_mm"], rtol=1e-10)
        row = dict(
            **record,
            footprint_iou=float(np.count_nonzero(top & target_mask) / np.count_nonzero(top | target_mask)),
            reconstruction_voxels_mm=[1.0, 1.25, 1.5],
            reconstruction_errors_mm=[score(data, target, h) for h in [0.001, 0.00125, 0.0015]],
            finer_quadrature_error_mm=mesh_distance_mm(mesh, target.subdivide(1, subfilter="linear")),
            surface_components=len(areas), largest_surface_area_fraction=float(areas.max() / areas.sum()),
            maximum_floor_penetration_mm=float(penetration.max() * 1000),
            fraction_particles_below_floor=float(np.mean(penetration > 0)),
            mean_floor_penetration_mm=float(np.average(penetration, weights=data["vol0"]) * 1000),
            max_pressed_height_mm=max(float(data[f"stage_{i}_pressed"][:, 2].max() - data["floor"]) for i in range(4)) * 1000,
            peak_force_per_finger_n=float(np.linalg.norm(data["reaction_force"], axis=-1).max()),
            peak_rotation_force_n=float(np.max(np.abs(force))),
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    assert len(rows) == 12
    save_json(folder / "additional_audit.json", dict(
        results=rows,
        scope="Fixed commands, common scale; numerical and reconstruction sensitivity, not a convergence proof",
    ))
    shutil.copy2(Path(__file__), folder / "audit_source.py")


def paper_report(folder, dest):
    base = Path(json.loads((folder / "protocol.json").read_text())["base"])
    report(base, dest, folder, cross_folder=folder)
    shutil.copy2(Path(__file__), folder / "report_source" / Path(__file__).name)
    provenance_path = folder / "figure_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["renderer_sources"][Path(__file__).name] = digest(Path(__file__))
    save_json(provenance_path, provenance)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["audit", "report"])
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--dest", type=Path)
    a = p.parse_args()
    if a.stage == "audit":
        audit(a.out.resolve())
    else:
        paper_report(a.out.resolve(), a.dest)
