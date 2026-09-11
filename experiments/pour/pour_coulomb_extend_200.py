"""Add 180/200 mL pilot commands using bounded, frozen-model MPM searches.

The previous six commands stay unchanged. No measured validation outcomes enter
this program. Numerical-check failures remain disclosed in the pilot package.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import numpy as np

from experiments.pour.pour_coulomb_overnight import (
    CONTACT, case_path, check_result_provenance, digest, next_angle, numerical_problem, write,
)
from experiments.pour.pour_coulomb_pilot_handoff import NOTES
from experiments.pour.pour_coulomb_timestep_followup import gpu_snapshot, eligible
from experiments.pour.pour_navier_reference import ROOT, OUT, ETA, MESH

WORK = OUT / "pilot_extension_200"
OLD = OUT / "overnight_targets"
PACKAGE = OUT / "philip_pilot_20260907_200"
NEW_TARGETS = [180, 200]


def load_case(angle):
    path = case_path(angle) / "result.json"
    r = json.loads(path.read_text())
    check_result_provenance(r, angle, 160, 0., 1.)
    issue = numerical_problem(r)
    if issue or r["particle_count"] != 229280:
        raise RuntimeError(issue or "Particle count changed")
    return dict(angle_deg=angle, receiver_ml=r["receiver_ml"],
                result_path=str(path.relative_to(ROOT)), result_sha256=digest(path))


def validate():
    path = WORK / "protocol.json"
    if digest(path) != (WORK / "protocol.sha256").read_text().strip():
        raise RuntimeError("Extension protocol changed")
    p = json.loads(path.read_text())
    for relative, expected in p["frozen_input_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f"Frozen input changed: {relative}")
    return p


def prepare():
    WORK.mkdir(exist_ok=True)
    (WORK / "logs").mkdir(exist_ok=True)
    if (WORK / "protocol.json").exists():
        return validate()
    baseline = load_case(60.)
    result = json.loads((ROOT / baseline["result_path"]).read_text())
    preflight = json.loads((WORK / "motion_preflight.json").read_text())
    assert preflight["maximum_command_angle_deg"] == 70.
    assert preflight["validation_outcomes_used"] == []
    assert [r["angle_deg"] for r in preflight["checks"]] == [60., 62., 64., 65., 66., 68., 70.]
    for r in preflight["checks"]:
        assert r["minimum_source_mesh_crop_clearance_m"] > .03
        assert r["minimum_joint_limit_margin_rad"] > 0 and r["join_max_rad"] < 1e-6
        assert r["tcp_mm_error"] < 1.5 and r["orientation_deg_error"] < .2
    paths = set(result["input_sha256"])
    paths.update(str(p.relative_to(ROOT)) for p in [
        Path(__file__), MESH, OLD / "predictions.json", OLD / "state.json", OLD / "angle_table.csv",
        WORK / "motion_preflight.json",
        OUT / "recorded60_timestep_followup/comparison.json",
        ROOT / "experiments/pour/pour_coulomb_overnight.py",
        ROOT / "experiments/pour/pour_coulomb_pilot_handoff.py",
        ROOT / "experiments/pour/pour_coulomb_timestep_followup.py"])
    protocol = dict(created_unix=time.time(), targets_ml=NEW_TARGETS,
        purpose="User-requested extension of the frozen pilot table",
        eta_pa_s=ETA, source_contact=CONTACT, grid=160, extent_m=.35, dt_scale=1., phase=0.,
        particle_count=229280, initial_volume_ml=300., dwell_s=2., return_s=2.,
        calibration_episode="09-04-60-2s", calibration_pour_count=1,
        viscosity_method="Original PDF weak spout-edge fit", parameters_refitted=[],
        validation_outcomes_used=[], previous_six_commands_changed=False,
        first_angles_deg=[64., 68.], new_angle_bounds_deg=[61., 70.],
        target_tolerance_ml=1., max_new_runs=8, per_run_timeout_s=2400, wall_budget_s=10800,
        gpu_max_utilization_percent=5, gpu_max_used_memory_mib=2048,
        release_status="Pilot only; previous numerical failures remain unresolved",
        frozen_input_sha256={p: digest(ROOT/p) for p in sorted(paths)})
    write(WORK / "protocol.json", protocol)
    (WORK / "protocol.sha256").write_text(digest(WORK / "protocol.json")+"\n")
    return validate()


class Extension:
    def __init__(self):
        self.protocol = prepare()
        path = WORK / "state.json"
        self.state = json.loads(path.read_text()) if path.exists() else dict(
            started_unix=time.time(), new_runs=0, cases={}, targets={})
        previous = json.loads((OLD / "state.json").read_text())
        self.base = []
        for c in previous["cases"].values():
            if (c.get("status") == "complete" and c["grid"] == 160 and c["phase"] == 0.
                    and c["dt_scale"] == 1.):
                r = load_case(c["angle_deg"])
                assert r["result_sha256"] == c["result_sha256"]
                self.base.append(r)
        for r in self.state["cases"].values():
            assert load_case(r["angle_deg"]) == r

    def status(self, stage, **fields):
        write(WORK / "state.json", self.state)
        write(WORK / "status.json", dict(stage=stage, updated_unix=time.time(),
            new_runs=self.state["new_runs"], completed_new_cases=len(self.state["cases"]),
            completed_targets=sorted(map(int, self.state["targets"])),
            numerical_acceptance_passed=False, **fields))

    def samples(self):
        return sorted(self.base + list(self.state["cases"].values()), key=lambda r:r["angle_deg"])

    def batch(self, angles):
        pending = []
        for angle in sorted(set(angles)):
            if not 61. <= angle <= 70.:
                raise RuntimeError("Proposed angle exceeds the preflight interval")
            path = case_path(angle)
            if (path / "result.json").exists():
                self.state["cases"][str(angle)] = load_case(angle)
            elif path.exists():
                raise RuntimeError(f"Incomplete earlier attempt preserved: {path}")
            else:
                pending.append(angle)
        active = {}
        try:
            while pending or active:
                validate()
                if time.time()-self.state["started_unix"] > self.protocol["wall_budget_s"]:
                    raise RuntimeError("Extension wall-time budget exhausted")
                for angle, task in list(active.items()):
                    proc = task["process"]
                    if proc.poll() is None:
                        if time.monotonic()-task["started"] > self.protocol["per_run_timeout_s"]:
                            raise TimeoutError(f"Replay {angle} exceeded its time budget")
                        continue
                    task["stream"].close()
                    if proc.returncode:
                        raise RuntimeError(f"Replay {angle} failed; inspect {task['log']}")
                    self.state["cases"][str(angle)] = load_case(angle)
                    del active[angle]
                    print(f"DONE {angle:.2f}: {self.state['cases'][str(angle)]['receiver_ml']:.4f} mL", flush=True)
                    self.status("case_complete", angle_deg=angle)
                if pending:
                    reserved = {t["gpu"]["uuid"] for t in active.values()}
                    free = [g for g in gpu_snapshot() if eligible(g, self.protocol) and g["uuid"] not in reserved]
                    for gpu in sorted(free, key=lambda g:g["index"], reverse=True):
                        if not pending:
                            break
                        if self.state["new_runs"] >= self.protocol["max_new_runs"]:
                            raise RuntimeError("Extension simulation-count budget exhausted")
                        angle = pending.pop(0)
                        log = WORK / "logs" / f"angle{angle:g}.log"
                        command = [sys.executable, "-u", "experiments/pour/pour_navier_reference.py",
                            "--wall", "original-separable", "--source-friction", str(CONTACT),
                            "--grid", "160", "--phase", "0", "--dt-scale", "1",
                            "--max-angle", "70", "--angle", str(angle), "--device", "cuda:0"]
                        stream = log.open("w")
                        proc = subprocess.Popen(command, cwd=ROOT,
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu["uuid"]),
                            stdout=stream, stderr=subprocess.STDOUT)
                        active[angle] = dict(process=proc, stream=stream, started=time.monotonic(), gpu=gpu, log=log)
                        self.state["new_runs"] += 1
                        print(f"START {angle:.2f} GPU={gpu['index']} PID={proc.pid}", flush=True)
                self.status("running" if active else "waiting_for_free_gpu" if pending else "batch_complete",
                    pending_angles=pending, active_cases=[dict(angle_deg=a, pid=t["process"].pid,
                        gpu=t["gpu"]["index"], elapsed_s=time.monotonic()-t["started"], log=str(t["log"]))
                        for a,t in active.items()])
                if pending or active:
                    time.sleep(15)
        finally:
            for task in active.values():
                proc = task["process"]
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        proc.kill(); proc.wait()
                task["stream"].close()

    def search(self):
        self.batch(self.protocol["first_angles_deg"])
        while len(self.state["targets"]) < len(NEW_TARGETS):
            points = self.samples()
            candidates = []
            for target in NEW_TARGETS:
                if str(target) in self.state["targets"]:
                    continue
                best = min(points, key=lambda r:abs(r["receiver_ml"]-target))
                if abs(best["receiver_ml"]-target) <= self.protocol["target_tolerance_ml"]:
                    self.state["targets"][str(target)] = dict(target_ml=target,
                        command_angle_deg=best["angle_deg"], tilt_duration_s=round(best["angle_deg"]/10,3),
                        predicted_ml=best["receiver_ml"], prediction_error_ml=best["receiver_ml"]-target,
                        result_path=best["result_path"], result_sha256=best["result_sha256"])
                    self.status("target_selected", target_ml=target)
                    continue
                if max(r["receiver_ml"] for r in points) < target:
                    if any(r["angle_deg"] == 70. for r in points):
                        raise RuntimeError(f"{target} mL not reached within 70-degree bound")
                    candidates.append(70.)
                else:
                    candidates.append(next_angle([(r["angle_deg"],r["receiver_ml"]) for r in points],target))
            if candidates:
                self.batch(candidates)
        validate()
        self.package()

    def package(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.interpolate import PchipInterpolator

        previous = json.loads((OLD / "predictions.json").read_text())
        rows = previous["predictions"] + [self.state["targets"][str(t)] for t in NEW_TARGETS]
        assert len(rows) == 8
        for row in rows:
            result = load_case(row["command_angle_deg"])
            assert result["result_sha256"] == row["result_sha256"]
            assert result["receiver_ml"] == row["predicted_ml"]
        assert all(b["command_angle_deg"] > a["command_angle_deg"] for a,b in zip(rows[:-1],rows[1:]))
        PACKAGE.mkdir(exist_ok=False)
        with (PACKAGE / "angle_table.csv").open("w",newline="") as stream:
            writer=csv.writer(stream)
            writer.writerow(["target_ml","command_angle_deg","tilt_duration_s"])
            for r in rows:
                writer.writerow([r["target_ml"],f"{r['command_angle_deg']:.2f}",f"{r['tilt_duration_s']:.3f}"])
        with (OLD / "angle_table.csv").open(newline="") as stream:
            old_csv=list(csv.reader(stream))
        with (PACKAGE / "angle_table.csv").open(newline="") as stream:
            assert list(csv.reader(stream))[:7] == old_csv
        points=self.samples()
        anchor_angles=list(range(48,62))+[64.,68.]
        if any(r["angle_deg"] == 70. for r in points):
            anchor_angles.append(70.)
        by_angle={r["angle_deg"]:r for r in points}
        aa=np.array(anchor_angles); vv=np.array([by_angle[a]["receiver_ml"] for a in aa])
        if np.any(np.diff(vv)<=0):
            raise RuntimeError("Simulated curve anchors reverse; retain raw results for review")
        fig,ax=plt.subplots(figsize=(9,5.6),constrained_layout=True)
        xx=np.linspace(aa.min(),aa.max(),700)
        ax.plot(xx,PchipInterpolator(aa,vv)(xx),color="#2166ac",label="MPM curve (simulated anchors)")
        shown=[r for r in points if r["angle_deg"]>=48.]
        ax.scatter([r["angle_deg"] for r in shown],[r["receiver_ml"] for r in shown],s=18,alpha=.5,
                   color="#2166ac",label="Full MPM evaluations")
        for i,r in enumerate(rows):
            x,y=r["command_angle_deg"],r["predicted_ml"]
            ax.scatter(x,y,color="#bc3d2b",s=42,zorder=4)
            ax.annotate(f"{r['target_ml']} mL · {x:.2f}°",(x,y),
                xytext=(7 if i<4 else -7,12 if i%2 else -18),textcoords="offset points",
                ha="left" if i<4 else "right",fontsize=9)
        ax.set(xlabel="Commanded pour angle (degrees)",ylabel="Settled receiver volume (mL)",
            title="Pilot predictions · 300 mL initial fill · 2 s hold\nNumerical sensitivity remains unresolved")
        ax.legend(loc="upper left",fontsize=8); ax.grid(alpha=.2)
        fig.savefig(PACKAGE / "angle_table.png",dpi=180); plt.close(fig)
        notes=NOTES.replace("60, 80, 100, 120, 140, 160 mL","60, 80, 100, 120, 140, 160, 180, 200 mL")
        notes=notes.replace("throughout the six trials","throughout the eight trials")
        (PACKAGE / "START_HERE.txt").write_text(notes)
        names=["angle_table.csv","angle_table.png","START_HERE.txt"]
        archive=PACKAGE / "philip_pilot_handoff.zip"
        with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
            for name in names:
                z.write(PACKAGE/name,arcname=name)
        with zipfile.ZipFile(archive) as z:
            assert z.namelist()==names and z.testzip() is None
            for name in names:
                assert z.read(name)==(PACKAGE/name).read_bytes()
        provenance=dict(model=self.protocol,predictions=rows,numerical_acceptance_passed=False,
            robot_accuracy_validated=False,validation_outcomes_used=[],parameters_refitted=[],
            previous_six_commands_changed=False,package_files_sha256={n:digest(PACKAGE/n) for n in names},
            zip_sha256=digest(archive),zip_bytes=archive.stat().st_size)
        write(PACKAGE/"provenance.json",provenance)
        write(WORK/"predictions.json",provenance)
        self.status("completed_pilot",archive=str(archive),zip_bytes=archive.stat().st_size)
        print(f"PILOT PACKAGE {archive}",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare",action="store_true")
    args=parser.parse_args()
    WORK.mkdir(exist_ok=True)
    with (WORK/"controller.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        job=Extension()
        if args.prepare:
            job.status("prepared")
            print("Preflight, frozen inputs and existing MPM results verified",flush=True)
            return
        try:
            job.search()
        except Exception as error:
            job.status("stopped_needs_review",reason=str(error))
            raise


if __name__ == "__main__":
    main()
