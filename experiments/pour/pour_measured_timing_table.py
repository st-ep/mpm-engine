"""Separate full-MPM pilot table with independently measured robot timing.

Preserve all previous results. Freeze the one-pour weak viscosity, same-pour
contact calibration and cup geometry. Only two joint/action histories inform
the motion clock; only new full-MPM outputs inform target command selection.
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
from types import SimpleNamespace
import zipfile

import numpy as np

from experiments.pour import pour_navier_reference as reference
from experiments.pour.pour_measured_timing import MeasuredTimingMotion, LOG_ROOT, TIMING_ANGLES
from experiments.pour.pour_coulomb_overnight import digest, write, next_angle, numerical_problem
from experiments.pour.pour_coulomb_timestep_followup import gpu_snapshot, eligible

ROOT = reference.ROOT
WORK = reference.OUT / "measured_timing_20260907"
OLD = reference.OUT / "philip_pilot_20260907_200"
PACKAGE = WORK / "philip_timing_handoff"
CONTACT = .117
TARGETS = [60, 80, 100, 120, 140, 160, 180, 200]


def result_path(angle):
    return WORK / "replays/original-separable" / (
        f"planned{angle:g}_bNA_n160_phase0_dt1_mu{CONTACT:.6f}") / "result.json"


def validate():
    path = WORK / "protocol.json"
    if digest(path) != (WORK / "protocol.sha256").read_text().strip():
        raise RuntimeError("Timing protocol changed")
    protocol = json.loads(path.read_text())
    for relative, expected in protocol["frozen_input_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f"Frozen input changed: {relative}")
    return protocol


def prepare():
    if (WORK / "protocol.json").exists():
        return validate()
    old = json.loads((OLD / "provenance.json").read_text())
    if digest(OLD / "philip_pilot_handoff.zip") != old["zip_sha256"]:
        raise RuntimeError("Previous package does not match its saved provenance")
    predictions = old["predictions"]
    for r in predictions:
        if digest(ROOT / r["result_path"]) != r["result_sha256"]:
            raise RuntimeError("Previous simulation result changed")
    paths = [Path(__file__), Path(reference.__file__), Path(reference.twin.__file__),
        ROOT / "experiments/pour/pour_measured_timing.py",
        ROOT / "experiments/pour/pour_angle_sweep.py",
        ROOT / "experiments/pour/pour_coulomb_overnight.py",
        ROOT / "experiments/pour/pour_coulomb_timestep_followup.py",
        ROOT / "experiments/pour/pour_weakform_recovery.py",
        ROOT / "experiments/pour/pour_weakform_identify.py",
        ROOT / "experiments/pour/pour_navier_quadratic_wall.py",
        ROOT / "experiments/pour/pour_navier_traction_wall.py",
        reference.GEOMETRY, reference.OBSERVATIONS, reference.MESH,
        WORK / "motion_characterization.json", WORK / "clock_preflight.json",
        *sorted((ROOT / "src/warpmpm").rglob("*.py")),
        *[reference.REFERENCE / n for n in ["actions.jsonl", "states.jsonl", "meta.json"]],
        *[OLD / n for n in ["angle_table.csv", "angle_table.png", "START_HERE.txt",
                            "philip_pilot_handoff.zip", "provenance.json"]],
        *[ROOT / r["result_path"] for r in predictions]]
    for a in TIMING_ANGLES:
        paths.extend(LOG_ROOT / f"recorded_{a:g}" / n for n in ["actions.jsonl", "states.jsonl", "meta.json"])
    protocol = dict(purpose=__doc__, created_unix=time.time(), targets_ml=TARGETS,
        eta_pa_s=reference.ETA, source_contact=CONTACT, grid=160, dt_scale=1.,
        extent_m=.35, phase=0., particle_count=229280, initial_volume_ml=300.,
        commanded_hold_s=2., commanded_return_duration_s=2.,
        viscosity_method="Original PDF weak spout-edge fit, original60 only",
        constitutive_identification_pours=1, constitutive_parameters_refitted=[],
        contact_calibration="Previously frozen, same original60 endpoint only",
        timing_characterization_angles_deg=list(TIMING_ANGLES),
        motion_model="Equal-weight average time-at-progress residuals from two joint/action logs; original spatial path",
        timing_generalization="Seconds residual assumed transferable over the table; two logs do not establish repeatability",
        fluid_validation_outcomes_used=[], measured_endpoints_used_for_this_update=[],
        geometry_changed=False, photos_used_for_correction=False,
        target_tolerance_ml=1., angle_bounds_deg=[45., 68.],
        old_predictions=predictions, first_angles_deg=[48.37, 53.04],
        max_new_runs=24, per_run_timeout_s=1800, wall_budget_s=14400,
        gpu_max_utilization_percent=5, gpu_max_used_memory_mib=2048,
        numerical_acceptance_passed=False, robot_accuracy_validated=False,
        release_status="Pilot; previously identified numerical sensitivity remains unresolved",
        frozen_input_sha256={str(p.relative_to(ROOT)): digest(p) for p in paths})
    write(WORK / "protocol.json", protocol)
    (WORK / "protocol.sha256").write_text(digest(WORK / "protocol.json")+"\n")
    (WORK / "logs").mkdir(exist_ok=True)
    return validate()


def run_case(angle, device):
    protocol = validate()
    if not protocol["angle_bounds_deg"][0] <= angle <= protocol["angle_bounds_deg"][1]:
        raise ValueError("Angle outside timing table bounds")
    original_factory = reference.planned_motion
    original_writer = reference.write_planned_episode
    characterization = json.loads((WORK / "motion_characterization.json").read_text())

    def timing_factory(max_angle):
        base, ep = original_factory(max_angle)
        return MeasuredTimingMotion(base, characterization), ep

    def writer(*args, **kwargs):
        info = original_writer(*args, **kwargs)
        path = args[1] / "meta.json"
        meta = json.loads(path.read_text())
        meta["planning"]["timing_method"] = characterization["method"]
        meta["planning"]["timing_protocol_sha256"] = digest(WORK / "protocol.json")
        path.write_text(json.dumps(meta, indent=2)+"\n")
        return info

    # The original forward runner is unchanged on disk. Its model and numerical
    # implementation are reused; all new outputs have a separate root.
    old_out = reference.OUT
    reference.OUT = WORK
    reference.planned_motion = timing_factory
    reference.write_planned_episode = writer
    try:
        reference.run(SimpleNamespace(source_friction=CONTACT, grid=160,
            wall="original-separable", slip_mm=None, angle=angle, max_angle=70.,
            phase=0., dt_scale=1., device=device))
    finally:
        reference.OUT = old_out
        reference.planned_motion = original_factory
        reference.write_planned_episode = original_writer
    path = result_path(angle)
    result = json.loads(path.read_text())
    issue = numerical_problem(result)
    if issue or result["particle_count"] != protocol["particle_count"]:
        raise RuntimeError(issue or "Particle count changed")
    validate()
    result.update(timing_protocol_sha256=digest(WORK / "protocol.json"),
        timing_characterization_sha256=digest(WORK / "motion_characterization.json"),
        timing_motion_changed=True, spatial_path_changed=False,
        constitutive_parameters_refitted=[], fluid_validation_outcomes_used=[])
    write(path, result)


def read_case(angle):
    path = result_path(angle)
    r = json.loads(path.read_text())
    if (r["timing_protocol_sha256"] != digest(WORK / "protocol.json")
            or r["eta_pa_s"] != reference.ETA or r["source_coulomb_friction"] != CONTACT
            or r["planned_angle_deg"] != angle or r["particle_count"] != 229280
            or r["fluid_validation_outcomes_used"] != [] or numerical_problem(r)):
        raise RuntimeError("Timing result provenance or capture/tail check failed")
    return dict(angle_deg=angle, receiver_ml=r["receiver_ml"],
        result_path=str(path.relative_to(ROOT)), result_sha256=digest(path))


class Search:
    def __init__(self):
        self.protocol = prepare()
        path = WORK / "state.json"
        self.state = json.loads(path.read_text()) if path.exists() else dict(
            started_unix=time.time(), new_runs=0, cases={}, targets={})
        for row in self.state["cases"].values():
            if row != read_case(row["angle_deg"]):
                raise RuntimeError("Previously saved case changed")

    def status(self, stage, **fields):
        write(WORK / "state.json", self.state)
        write(WORK / "status.json", dict(stage=stage, updated_unix=time.time(),
            new_runs=self.state["new_runs"], completed_cases=len(self.state["cases"]),
            completed_targets=sorted(map(int, self.state["targets"])),
            previous_package_preserved=True, **fields))

    def batch(self, angles):
        pending = []
        for angle in sorted(set(angles)):
            angle = round(float(angle), 2)
            if not self.protocol["angle_bounds_deg"][0] <= angle <= self.protocol["angle_bounds_deg"][1]:
                raise RuntimeError("Proposed angle outside bounds")
            if result_path(angle).is_file():
                self.state["cases"][str(angle)] = read_case(angle)
            elif result_path(angle).parent.exists():
                raise RuntimeError("Incomplete run preserved for inspection")
            else:
                pending.append(angle)
        active = {}
        env = dict(os.environ, MUJOCO_GL="egl", PYTHONPATH=str(ROOT),
                   OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1")
        try:
            while active or pending:
                validate()
                if time.time()-self.state["started_unix"] > self.protocol["wall_budget_s"]:
                    raise RuntimeError("Timing search reached its wall-time budget")
                for angle, task in list(active.items()):
                    rc = task["process"].poll()
                    if rc is None:
                        if time.monotonic()-task["started"] > self.protocol["per_run_timeout_s"]:
                            raise RuntimeError("A replay exceeded its time limit")
                        continue
                    task["stream"].close()
                    if rc:
                        raise RuntimeError(f"Replay {angle} failed; inspect {task['log']}")
                    self.state["cases"][str(angle)] = read_case(angle)
                    del active[angle]
                    print(f"DONE {angle:.2f}: {self.state['cases'][str(angle)]['receiver_ml']:.4f} mL", flush=True)
                if pending:
                    reserved = {t["gpu"] for t in active.values()}
                    for gpu in gpu_snapshot():
                        if not pending:
                            break
                        if gpu["index"] in reserved or not eligible(gpu, self.protocol):
                            continue
                        if self.state["new_runs"] >= self.protocol["max_new_runs"]:
                            raise RuntimeError("Timing search reached its declared run cap")
                        angle = pending.pop(0)
                        log = WORK / "logs" / f"angle_{angle:.2f}.log"
                        stream = log.open("w")
                        proc = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                            "--case", str(angle), "--device", f"cuda:{gpu['index']}"],
                            cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                        active[angle] = dict(process=proc, stream=stream, log=str(log),
                                             gpu=gpu["index"], started=time.monotonic())
                        self.state["new_runs"] += 1
                        print(f"START {angle:.2f} on GPU{gpu['index']}", flush=True)
                self.status("running", active_angles=list(active), pending_angles=pending,
                            pids={str(a):t["process"].pid for a,t in active.items()})
                if active or pending:
                    time.sleep(10)
        finally:
            for task in active.values():
                if task["process"].poll() is None:
                    task["process"].terminate()
                    try:
                        task["process"].wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        task["process"].kill()
                        task["process"].wait()
                task["stream"].close()

    def search(self):
        self.batch(self.protocol["first_angles_deg"])
        self.status("initial_pair_complete")
        self.batch([r["command_angle_deg"] for r in self.protocol["old_predictions"]])
        while len(self.state["targets"]) < len(TARGETS):
            points = sorted(self.state["cases"].values(), key=lambda r:r["angle_deg"])
            candidates = []
            for target in TARGETS:
                if str(target) in self.state["targets"]:
                    continue
                best = min(points, key=lambda r:abs(r["receiver_ml"]-target))
                if abs(best["receiver_ml"]-target) <= self.protocol["target_tolerance_ml"]:
                    self.state["targets"][str(target)] = dict(target_ml=target,
                        command_angle_deg=best["angle_deg"], tilt_duration_s=round(best["angle_deg"]/10,3),
                        predicted_ml=best["receiver_ml"], result_path=best["result_path"],
                        result_sha256=best["result_sha256"])
                elif target < min(r["receiver_ml"] for r in points):
                    candidates.append(round(points[0]["angle_deg"]-.5,2))
                elif target > max(r["receiver_ml"] for r in points):
                    candidates.append(round(points[-1]["angle_deg"]+.5,2))
                else:
                    candidates.append(next_angle([(r["angle_deg"],r["receiver_ml"]) for r in points], target))
            self.status("selecting_commands")
            if candidates:
                self.batch(candidates)
        self.package()

    def package(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.interpolate import PchipInterpolator
        validate()
        rows = [self.state["targets"][str(t)] for t in TARGETS]
        for row in rows:
            result = read_case(row["command_angle_deg"])
            if result["result_sha256"] != row["result_sha256"]:
                raise RuntimeError("Selected result changed")
        aa = np.array([r["command_angle_deg"] for r in rows])
        vv = np.array([r["predicted_ml"] for r in rows])
        if np.any(np.diff(aa)<=0) or np.any(np.diff(vv)<=0):
            raise RuntimeError("Selected commands are not increasing; review raw simulations")
        PACKAGE.mkdir(exist_ok=False)
        with (PACKAGE / "angle_table.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["target_ml", "command_angle_deg", "tilt_duration_s"])
            for r in rows:
                writer.writerow([r["target_ml"], f"{r['command_angle_deg']:.2f}", f"{r['tilt_duration_s']:.3f}"])
        fig, ax = plt.subplots(figsize=(9,5.6), constrained_layout=True)
        xx = np.linspace(aa[0],aa[-1],600)
        ax.plot(xx,PchipInterpolator(aa,vv)(xx),color="#2166ac",label="MPM curve through verified commands")
        samples = sorted(self.state["cases"].values(),key=lambda r:r["angle_deg"])
        ax.scatter([r["angle_deg"] for r in samples],[r["receiver_ml"] for r in samples],
                   s=18,alpha=.5,color="#2166ac",label="Full MPM evaluations")
        for i,r in enumerate(rows):
            x,y=r["command_angle_deg"],r["predicted_ml"]
            ax.scatter(x,y,color="#bc3d2b",s=40,zorder=4)
            ax.annotate(f"{r['target_ml']} mL · {x:.2f}°",(x,y),xytext=(7 if i<4 else -7,12 if i%2 else -18),
                        textcoords="offset points",ha="left" if i<4 else "right",fontsize=9)
        ax.set(xlabel="Commanded pour angle (degrees)",ylabel="Settled receiver volume (mL)",
               title="Timing-adjusted MPM · 300 mL initial fill · 2 s hold")
        ax.legend(loc="upper left",fontsize=8); ax.grid(alpha=.2)
        fig.savefig(PACKAGE / "angle_table.png",dpi=180);plt.close(fig)
        notes = """TIMING-ADJUSTED PILOT — 60, 80, 100, 120, 140, 160, 180, 200 mL

Use angle_table.csv: target volume, commanded angle, tilt-command duration.
Before each run, fill the source to 300 mL and empty the receiver. Keep the cup
in the same grasp throughout the session. Use the same starting pose and planner.

Tilt using the table angle and duration. Hold 2.0 seconds AFTER the tilt ACK.
Then issue the usual 2.0-second return command. Do not add an extra robot delay:
the new simulations already account for measured differences in execution timing.

Record joints, action timestamps, side video and the final received amount for
each trial. Keep these commands fixed throughout the eight tests.

Each row has its own full MPM prediction. Viscosity was identified with the weak
spout-edge method from one original 60-degree pour; contact was calibrated on
that same pour. Two additional joint/action recordings characterize robot timing.
Their measured liquid outcomes were not used. Cup geometry is unchanged.

This is a separate pilot table. Numerical sensitivity and timing variability
remain unresolved; +/-10 mL robot accuracy has not been established.
"""
        (PACKAGE / "START_HERE.txt").write_text(notes)
        names = ["angle_table.csv","angle_table.png","START_HERE.txt"]
        archive = PACKAGE / "philip_timing_handoff.zip"
        with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
            for name in names:z.write(PACKAGE/name,arcname=name)
        with zipfile.ZipFile(archive) as z:
            assert z.namelist()==names and z.testzip() is None
            for name in names:assert z.read(name)==(PACKAGE/name).read_bytes()
        provenance = dict(protocol_sha256=digest(WORK/"protocol.json"), predictions=rows,
            all_mpm_evaluations=samples, numerical_acceptance_passed=False,
            robot_accuracy_validated=False, fluid_validation_outcomes_used=[],
            old_package_sha256=digest(OLD/"philip_pilot_handoff.zip"),
            package_files_sha256={n:digest(PACKAGE/n) for n in names},
            zip_sha256=digest(archive),zip_bytes=archive.stat().st_size)
        write(WORK/"predictions.json",provenance)
        write(PACKAGE/"provenance.json",provenance)
        self.status("completed_pilot_needs_review",archive=str(archive))
        print(f"PACKAGE {archive}",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--case",type=float)
    parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args()
    if args.case is not None:
        run_case(args.case,args.device)
        return
    with (WORK/"controller.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        job=Search()
        if args.prepare:
            job.status("prepared")
            return
        try:job.search()
        except Exception as error:
            job.status("stopped_needs_review",reason=str(error))
            raise


if __name__=="__main__":
    main()
