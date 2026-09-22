"""Replay the frozen four-pinch motions on both true materials, without feedback.

Only saved finger poses drive execution. Forces are recorded, never commanded.
This experiment does not rerun identification or planning and cannot write the
paper. Use a fresh output directory for each reproduction.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import shutil
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from experiments.robotics import x_shaping as geometry
from experiments.robotics.plastic_shaping_study import save_json, save_npz
from experiments.robotics.plastic_shaping_figure import surface
from experiments.robotics.hex_shaping_surface import mesh_distance_mm

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "out/x_consistent_force_study_20260911"
OUT = ROOT / "out/x_motion_shaping_20260911"
COLORS = {"A": "#7aacc7", "B": "#d39b76"}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def prepare(source, folder):
    assert not folder.exists(), "Use a new output directory; preserve earlier runs"
    manifest = read(source / "checksums.json")
    for rel, sha in manifest.items():
        assert digest(source / rel) == sha, rel
    folder.mkdir(parents=True)
    inputs = folder / "inputs"
    inputs.mkdir()
    for name in ["models.json"]:
        shutil.copy2(source / "inputs" / name, inputs / name)
    shutil.copy2(source / "target.vtp", inputs / "target.vtp")
    references = {}
    initial = None
    for planned in "AB":
        selected = read(source / f"plans/identified_{planned}/selected.json")
        data = arrays(source / f"reference_{planned}.npz")
        assert digest(source / f"reference_{planned}.npz") == selected["data_sha256"]
        assert data["reference_generation"] and np.max(abs(data["force_error"])) == 0
        phases = read(source / f"reference_{planned}.phases.json")
        motion = dict(time=data["time"], tool_centers=data["tool_centers"],
                      phase_id=data["phase_id"], pinch_id=data["pulse_id"],
                      start_pose=np.array(phases[0]["start_centers"]),
                      gaps_mm=np.array(selected["gaps_mm"]))
        previous = np.concatenate([motion["start_pose"][None], motion["tool_centers"][:-1]])
        np.testing.assert_allclose((motion["tool_centers"] - previous) / .004,
                                   data["tool_velocity"], atol=1e-12)
        np.testing.assert_allclose(data["min_gaps_mm"], motion["gaps_mm"], atol=1e-7)
        specimen = dict(initial=data["initial"], vol0=data["vol0"])
        if initial is None:
            initial = specimen
            save_npz(inputs / "specimen.npz", initial)
        else:
            for key in specimen:
                np.testing.assert_array_equal(specimen[key], initial[key])
        for phase in phases:
            phase["name"] = phase["name"].replace(":force_profile", ":close")
        save_npz(inputs / f"motion_{planned}.npz", motion)
        save_json(inputs / f"phases_{planned}.json", phases)
        save_json(inputs / f"selected_{planned}.json", selected)
        # Keep the previous results as explicit comparison data, not new runs.
        for actual in "AB":
            shutil.copy2(source / f"baseline_{actual}_plan_{planned}.json",
                         inputs / f"previous_force_{actual}_plan_{planned}.json")
        references[planned] = dict(reference_sha256=selected["data_sha256"],
            selected_file=selected["file"], gaps_mm=selected["gaps_mm"],
            original_evaluations=selected["evaluations"],
            original_optimizer_success=selected["optimizer_success"])
    snapshot = folder / "source_snapshot"
    sources = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "experiments/robotics").glob("*.py"))
    for path in sources:
        dest = snapshot / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    protocol = dict(source_archive=str(source.resolve()),
        source_manifest_sha256=digest(source / "checksums.json"),
        original_archive_files_verified=len(manifest), references=references,
        scene=geometry.CONFIG, n_grid=64, dt=.00005, tick=.004,
        cases=[f"{a}_plan_{b}" for a in "AB" for b in "AB"],
        execution="Replay every saved finger pose at its original tick; no force or shape feedback",
        planning="Import the existing selected plans; no search, retiming, or parameter tuning",
        measurement="1 s after final withdrawal, as in the source planning trajectory",
        metric="Symmetric area-weighted mean 3D surface distance in world coordinates, mm",
        surface_voxel_m=.00125, gaussian_sigma_voxels=1.3, isovalue=.5,
        all_components_included=True, alignment=False, rescaling=False,
        scope="Four baseline simulations only; no new grid study or hardware validation",
        input_sha256={str(p.relative_to(folder)): digest(p) for p in sorted(inputs.iterdir())},
        source_sha256=source_hashes,
        packages={p: importlib.metadata.version(p) for p in
                  ["numpy", "scipy", "warp-lang", "pyvista", "matplotlib", "scikit-image"]})
    save_json(folder / "protocol.json", protocol)
    (folder / "commit.txt").write_text(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True))
    print(json.dumps(dict(prepared=str(folder), references=references)), flush=True)


def check(folder):
    protocol = read(folder / "protocol.json")
    assert protocol["scene"] == json.loads(json.dumps(geometry.CONFIG))
    for rel, sha in protocol["input_sha256"].items():
        assert digest(folder / rel) == sha, rel
    for rel, sha in protocol["source_sha256"].items():
        assert digest(ROOT / rel) == digest(folder / "source_snapshot" / rel) == sha, rel
    return protocol


def run(folder, material, device):
    protocol = check(folder)
    cfg = protocol["scene"]
    initial = arrays(folder / "inputs/specimen.npz")
    law = read(folder / "inputs/models.json")[f"true_{material}"]
    tick, dt = protocol["tick"], protocol["dt"]
    substeps = round(tick / dt)
    assert abs(substeps * dt - tick) < 1e-12
    for planned in "AB":
        stem = folder / f"{material}_plan_{planned}"
        assert not stem.with_suffix(".config.json").exists(), "Preserve every attempted run"
        command = arrays(folder / f"inputs/motion_{planned}.npz")
        phases = read(folder / f"inputs/phases_{planned}.json")
        config = dict(material=material, planned_for=planned, law=law, device=device,
            protocol_sha256=digest(folder / "protocol.json"),
            motion_sha256=digest(folder / f"inputs/motion_{planned}.npz"),
            force_feedback=False, shape_feedback=False)
        save_json(stem.with_suffix(".config.json"), config)
        start = time.monotonic()
        with contextlib.redirect_stdout(io.StringIO()):
            solver = geometry.Solver(geometry.GridConfig(n_grid=protocol["n_grid"], grid_lim=cfg["domain"]),
                device=device, inversion_policy="raise").load_particles(initial["initial"], initial["vol0"])
            solver.set_material(geometry.vonmises(**law, density=cfg["density"]))
        solver.add_plane(point=(0, 0, cfg["floor"]), normal=(0, 0, 1),
                         surface="separable", friction=cfg["floor_friction"])
        sdf = geometry.cylinder_sdf(cfg["radius"], cfg["finger_height"])
        pose = command["start_pose"].copy()
        handles = [solver.add_sdf_collider(sdf, center=cp, band=cfg["band"],
                   surface="separable", friction=cfg["tool_friction"]) for cp in pose]
        forces, velocities, extents = [], [], []
        data = dict(**initial, **command)
        print(json.dumps(dict(starting=stem.name, ticks=len(command["time"]), device=device)), flush=True)
        for j, nxt in enumerate(command["tool_centers"]):
            velocity = (nxt - pose) / tick
            for handle, cp, v in zip(handles, pose, velocity, strict=True):
                solver.set_sdf_pose(handle, center=cp, velocity=v)
                solver.reset_sdf_force(handle)
            solver.step(dt, substeps=substeps)
            forces.append(np.stack([solver.sdf_wrench(h, tick)["force"] for h in handles]))
            velocities.append(velocity)
            pose = nxt
            if (j + 1) % 25 == 0:
                xyz = solver.x()
                assert np.all(np.isfinite(xyz))
                assert xyz[:, :2].min() > .003 and xyz[:, :2].max() < cfg["domain"] - .003
                extents.append(np.r_[command["time"][j], xyz.min(0), xyz.max(0)])
            if j + 1 == len(command["time"]) or command["phase_id"][j + 1] != command["phase_id"][j]:
                name = phases[int(command["phase_id"][j])]["name"]
                if name.endswith(":close") or name.endswith(":withdraw"):
                    data[name.replace(":", "_")] = solver.x().copy()
            if (j + 1) % 1000 == 0:
                print(json.dumps(dict(case=stem.name, completed_ticks=j + 1)), flush=True)
        data.update(x_after_1s=solver.x().copy(), v_after_1s=solver.v().copy(),
            reaction_force=np.array(forces), tool_velocity=np.array(velocities),
            extent_samples=np.array(extents), inverted_count=np.array(solver.inverted_count()))
        assert data["inverted_count"] == 0 and all(np.all(np.isfinite(v)) for v in data.values())
        save_npz(stem.with_suffix(".npz"), data)
        error = mesh_distance_mm(surface(data["x_after_1s"], data["vol0"], h=.00125),
                                 pv.read(folder / "inputs/target.vtp"))
        record = dict(**config, surface_mm=error, data_sha256=digest(stem.with_suffix(".npz")),
                      elapsed_s=time.monotonic() - start, completed_pinches=4,
                      duration_s=float(command["time"][-1]))
        save_json(stem.with_suffix(".json"), record)
        print(json.dumps(record), flush=True)


def audit(folder):
    protocol = check(folder)
    target = pv.read(folder / "inputs/target.vtp")
    assert target.n_open_edges == 0
    np.testing.assert_allclose(target.volume, .00009, rtol=2e-6)
    specimen = arrays(folder / "inputs/specimen.npz")
    rows = []
    for actual in "AB":
        for planned in "AB":
            name = f"{actual}_plan_{planned}"
            data = arrays(folder / f"{name}.npz")
            command = arrays(folder / f"inputs/motion_{planned}.npz")
            record = read(folder / f"{name}.json")
            assert record["protocol_sha256"] == digest(folder / "protocol.json")
            assert record["data_sha256"] == digest(folder / f"{name}.npz")
            assert record["law"] == read(folder / "inputs/models.json")[f"true_{actual}"]
            assert record["material"] == actual and record["planned_for"] == planned
            assert not record["force_feedback"] and not record["shape_feedback"]
            for key in command:
                np.testing.assert_array_equal(data[key], command[key])
            for key in specimen:
                np.testing.assert_array_equal(data[key], specimen[key])
            assert all(np.all(np.isfinite(v)) for v in data.values()) and data["inverted_count"] == 0
            np.testing.assert_allclose(data["vol0"].sum(dtype=float), .00009, rtol=1e-6)
            np.testing.assert_allclose(np.diff(data["time"]), .004, atol=1e-10)
            before = np.concatenate([data["start_pose"][None], data["tool_centers"][:-1]])
            np.testing.assert_allclose(data["tool_velocity"], (data["tool_centers"] - before) / .004, atol=1e-12)
            normal = data["tool_centers"][:, 1] - data["tool_centers"][:, 0]
            separation = np.linalg.norm(normal, axis=1)
            normal /= separation[:, None]
            measured = (np.einsum("tfc,tc->tf", data["reaction_force"], normal) * [-1, 1]).mean(1)
            pinches = []
            for i in range(4):
                active = data["pinch_id"] == i
                gaps = (separation[active] - .028) * 1000
                np.testing.assert_allclose(gaps[-1], data["gaps_mm"][i], atol=1e-7)
                assert gaps.min() >= 16 - 1e-7 and gaps.max() <= 72 + 1e-7
                speed = np.linalg.norm(data["tool_velocity"][active], axis=-1)
                assert speed.max() <= .020000001
                pinches.append(dict(pinch=i + 1, final_opening_mm=float(gaps[-1]),
                    duration_s=int(active.sum()) * .004, measured_peak_mean_force_n=float(measured[active].max())))
            phases = read(folder / f"inputs/phases_{planned}.json")
            off_contact = [i for i, p in enumerate(phases) if p["name"].endswith(":rotate") or p["name"] == "final:wait"]
            assert np.max(abs(data["reaction_force"][np.isin(data["phase_id"], off_contact)])) == 0
            mesh = surface(data["x_after_1s"], data["vol0"], h=.00125)
            error = mesh_distance_mm(mesh, target)
            np.testing.assert_allclose(error, record["surface_mm"], rtol=1e-10)
            components = mesh.connectivity(extraction_mode="all")
            extents = data["extent_samples"]
            previous = read(folder / f"inputs/previous_force_{name}.json")
            rows.append(dict(**record, pinches=pinches,
                surface_components=int(np.max(components.cell_data["RegionId"])) + 1,
                max_sampled_floor_penetration_mm=float(max(0, .01 - extents[:, 3].min()) * 1000),
                max_sampled_height_mm=float((extents[:, 6].max() - .01) * 1000),
                final_rms_speed_mm_s=float(np.sqrt(np.mean(np.sum(data["v_after_1s"] ** 2, axis=1))) * 1000),
                previous_force_feedback_surface_mm=previous["surface_mm"],
                previous_force_feedback_stops=previous["stops"]))
    summary = dict(results=rows, protocol_sha256=digest(folder / "protocol.json"),
        executed_four_frozen_motions=True, identical_commands_across_materials=True,
        tuning_or_replanning=False, all_cases_retained=True,
        scope=protocol["scope"], floor_penetration="Grid contact permits subcell penetration; no exact-contact claim")
    save_json(folder / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return rows


def render(mesh, scale, color, mask=False):
    plotter = pv.Plotter(off_screen=True, window_size=(650, 700))
    plotter.set_background("white")
    plotter.add_mesh(mesh, color="black" if mask else color, lighting=not mask,
        smooth_shading=True, split_sharp_edges=True, ambient=.4, diffuse=.7, specular=.1)
    if not mask:
        plotter.add_mesh(pv.Plane(center=(.08, .08, .0099), i_size=.13, j_size=.13),
                         color="#f1f2f2", lighting=False)
    focal = np.array([.08, .08, .027])
    plotter.camera_position = [focal + [0, -.13, .24], focal, (0, 1, 0)]
    plotter.enable_parallel_projection()
    plotter.camera.parallel_scale = scale
    if not mask:
        plotter.enable_anti_aliasing("ssaa")
    picture = plotter.screenshot(return_img=True)
    plotter.close()
    return picture


def report(folder):
    rows = audit(folder)
    meshes = {"target": pv.read(folder / "inputs/target.vtp")}
    for row in rows:
        key = f"{row['material']}_plan_{row['planned_for']}"
        data = arrays(folder / f"{key}.npz")
        meshes[key] = surface(data["x_after_1s"], data["vol0"], h=.00125)
    pictures = folder / "preview"
    pictures.mkdir(exist_ok=False)
    for scale in [.055, .061, .068, .076, .085, .10]:
        masks = {k: render(m, scale, "black", mask=True)[..., :3].mean(-1) < 128 for k, m in meshes.items()}
        if all(not (m[:10].any() or m[-10:].any() or m[:, :10].any() or m[:, -10:].any()) for m in masks.values()):
            break
    else:
        raise RuntimeError("Not all surfaces fit the common camera")
    union = np.logical_or.reduce(list(masks.values()))
    yy, xx = np.nonzero(union)
    crop = (int(yy.min()) - 10, int(yy.max()) + 11, int(xx.min()) - 10, int(xx.max()) + 11)
    y0, y1, x0, x1 = crop
    images = {}
    for key, mesh in meshes.items():
        assert masks[key][y0:y1, x0:x1].sum() == masks[key].sum()
        image = render(mesh, scale, "#b5b6b0" if key == "target" else COLORS[key[0]])
        images[key] = image[y0:y1, x0:x1]
        plt.imsave(pictures / f"{key}.png", images[key])
        mesh.save(pictures / f"{key}.vtp")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 7.3))
    for row, material in enumerate("AB"):
        for col, planned in enumerate([None, material, "B" if material == "A" else "A"]):
            ax = axes[row, col]
            key = "target" if planned is None else f"{material}_plan_{planned}"
            ax.imshow(images[key])
            if planned is not None:
                ax.contour(masks["target"][y0:y1, x0:x1], [.5], colors="#253139",
                           linewidths=.9, linestyles=[(0, (3, 2))])
                record = next(r for r in rows if r["material"] == material and r["planned_for"] == planned)
                ax.text(.5, -.035, f"{record['surface_mm']:.3f} mm", transform=ax.transAxes,
                        ha="center", va="top", fontsize=14)
            ax.axis("off")
            if row == 0:
                ax.set_title(["Target", "Plan using matched ID", "Plan using swapped ID"][col], pad=15, fontsize=13)
            if col == 0:
                ax.text(-.13, .5, material, transform=ax.transAxes, ha="center", va="center",
                        color={"A": "#2375aa", "B": "#c96932"}[material], fontsize=21, weight="bold")
    fig.suptitle("X shaping with prescribed finger motions", y=.98, fontsize=17, weight="bold", color="#202a32")
    fig.text(.5, .927, "Four frozen pinches per plan · no force feedback", ha="center", color="#53606a")
    fig.text(.5, .018, "Released shapes at 1 s · dashed outline: target · labels: mean 3D surface error",
             ha="center", fontsize=11, color="#53606a")
    fig.subplots_adjust(left=.09, right=.98, top=.82, bottom=.10, wspace=.19, hspace=.22)
    for suffix in ["png", "pdf"]:
        fig.savefig(pictures / f"motion_comparison.{suffix}", dpi=220)
    plt.close(fig)
    save_json(pictures / "provenance.json", dict(common_camera_scale_m=scale, common_crop_yxyx=crop,
        all_surface_masks_retained=True, surface_colors=COLORS, rows=rows,
        source_sha256=digest(__file__), target_sha256=digest(folder / "inputs/target.vtp"),
        note="Same camera, reconstruction and lighting as the current paper outcomes; no individual alignment or rescaling"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run", "report"])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--material", choices=["A", "B"])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.source, args.out)
    elif args.stage == "run":
        if args.material is None:
            parser.error("run requires --material")
        run(args.out, args.material, args.device)
    else:
        report(args.out)
