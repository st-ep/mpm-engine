"""Two fixed-model recorded-60 MPM checks, queued until a GPU is idle.

No endpoint fitting or target-angle search. Preserve the previous overnight run.
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

from experiments.pour.pour_coulomb_overnight import (
    CONTACT, case_path, check_result_provenance, digest, numerical_problem, write,
)
from experiments.pour.pour_navier_reference import ETA, MESH, OUT, ROOT, twin

WORK = OUT / "recorded60_timestep_followup"
SCALES = [1., .5, .25]


def result_for(scale):
    path = case_path(60., dt=scale) / "result.json"
    result = json.loads(path.read_text())
    check_result_provenance(result, 60., 160, 0., scale)
    if (result["planned_angle_deg"] is not None
            or result["simulation_episode"] != "09-04-60-2s"
            or result["particle_count"] != 229280):
        raise RuntimeError("Expected the exact recorded motion and identical particle count")
    problem = numerical_problem(result)
    if problem:
        raise RuntimeError(problem)
    return result


def prepare():
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "logs").mkdir(exist_ok=True)
    baseline = result_for(1.)
    protocol_path = WORK / "protocol.json"
    if protocol_path.exists():
        return validate()
    paths = set(baseline["input_sha256"])
    paths.update([str(MESH.relative_to(ROOT)), str(Path(__file__).relative_to(ROOT)),
                  "experiments/pour/pour_coulomb_overnight.py"])
    baseline_path = case_path(60.) / "result.json"
    metrics_path = case_path(60.) / "09-04-60-2s/metrics.csv"
    paths.update(str(p.relative_to(ROOT)) for p in [baseline_path, metrics_path])
    base_substeps = twin.substeps_per_tick(dict(twin.GLYCEROL, eta=ETA), .35/160, 1/twin.FPS)
    protocol = dict(
        created_unix=time.time(), identification_episode="09-04-60-2s",
        eta_pa_s=ETA, contact_coefficient=CONTACT, initial_volume_ml=300.,
        grid=160, phase_xz_cells=0., extent_m=.35, particle_count=229280,
        time_step_scales=SCALES, new_runs=2, baseline_receiver_ml=baseline["receiver_ml"],
        substeps_per_frame={str(s): round(base_substeps/s) for s in SCALES},
        dt_s={str(s): (1/twin.FPS)/round(base_substeps/s) for s in SCALES},
        settling="Fresh settling at each time step with the unchanged quiescence rule",
        parameters_refitted=[], validation_outcomes_used=[], measured_endpoints_used=[],
        purpose="Diagnose temporal sensitivity of the previously calibrated numerical model",
        adjacent_difference_limit_ml=1., outside_limit_ml=3., tail_limit_ml=.5,
        interpretation="A contracting difference is evidence of stabilization, not proof of convergence or robot accuracy",
        gpu_order=[1, 0], gpu_max_utilization_percent=5, gpu_max_used_memory_mib=2048,
        idle_samples_required=3, poll_seconds=30, queue_timeout_s=86400,
        per_run_timeout_s=7200,
        frozen_input_sha256={p: digest(ROOT/p) for p in sorted(paths)},
    )
    write(protocol_path, protocol)
    (WORK / "protocol.sha256").write_text(digest(protocol_path) + "\n")
    status("prepared", completed_scales=[1.])
    return validate()


def validate():
    path = WORK / "protocol.json"
    if digest(path) != (WORK / "protocol.sha256").read_text().strip():
        raise RuntimeError("Follow-up protocol changed")
    protocol = json.loads(path.read_text())
    for relative, expected in protocol["frozen_input_sha256"].items():
        if digest(ROOT/relative) != expected:
            raise RuntimeError(f"Frozen input changed: {relative}")
    return protocol


def status(stage, **fields):
    write(WORK / "status.json", dict(stage=stage, updated_unix=time.time(),
          handoff_ready=False, **fields))


def gpu_snapshot():
    def query(option, fields):
        return subprocess.check_output(
            ["nvidia-smi", f"--query-{option}={fields}", "--format=csv,noheader,nounits"],
            text=True, timeout=15)
    gpu_rows = list(csv.reader(query("gpu", "index,uuid,memory.used,utilization.gpu").splitlines()))
    process_rows = list(csv.reader(query("compute-apps", "gpu_uuid,pid").splitlines()))
    occupied = {r[0].strip() for r in process_rows if len(r) == 2}
    return [dict(index=int(r[0]), uuid=r[1].strip(), memory_mib=int(r[2]),
                 utilization_percent=int(r[3]), compute_process_present=r[1].strip() in occupied)
            for r in gpu_rows]


def eligible(gpu, protocol):
    return (not gpu["compute_process_present"]
            and gpu["memory_mib"] <= protocol["gpu_max_used_memory_mib"]
            and gpu["utilization_percent"] <= protocol["gpu_max_utilization_percent"])


def wait_for_gpu(protocol, scale):
    start = time.monotonic()
    counts = {i: 0 for i in protocol["gpu_order"]}
    while time.monotonic()-start < protocol["queue_timeout_s"]:
        validate()
        snapshot = gpu_snapshot()
        by_index = {g["index"]: g for g in snapshot}
        for i in counts:
            counts[i] = counts[i]+1 if i in by_index and eligible(by_index[i], protocol) else 0
        status("waiting_for_free_gpu", next_dt_scale=scale, gpu_snapshot=snapshot,
               consecutive_idle_samples=counts, waited_s=time.monotonic()-start)
        for i in protocol["gpu_order"]:
            if counts[i] >= protocol["idle_samples_required"]:
                return by_index[i]
        time.sleep(protocol["poll_seconds"])
    raise RuntimeError("No free GPU within the 24-hour queue window")


def report(protocol):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, curves = [], []
    for scale in SCALES:
        result_path = case_path(60., dt=scale) / "result.json"
        if not result_path.exists():
            continue
        result = result_for(scale)
        metrics = case_path(60., dt=scale) / "09-04-60-2s/metrics.csv"
        with metrics.open() as stream:
            series = list(csv.DictReader(stream))
        row = dict(dt_scale=scale, dt_s=protocol["dt_s"][str(scale)],
                   substeps_per_frame=protocol["substeps_per_frame"][str(scale)],
                   receiver_ml=result["receiver_ml"], outside_ml=result["outside_ml"],
                   tail_variation_ml=result["tail_variation_ml"], elapsed_s=result["elapsed_s"],
                   difference_from_previous_ml=None if not rows else result["receiver_ml"]-rows[-1]["receiver_ml"],
                   result_path=str(result_path.relative_to(ROOT)), result_sha256=digest(result_path),
                   metrics_sha256=digest(metrics))
        rows.append(row)
        curves.append(([float(r["t"]) for r in series],
                       [300*int(r["n_rcv"])/229280 for r in series]))
    complete = len(rows) == 3
    comparison = dict(cases=rows, complete=complete, handoff_ready=False,
                      parameters_refitted=[], validation_outcomes_used=[])
    if complete:
        first, second = [rows[i]["difference_from_previous_ml"] for i in [1, 2]]
        comparison.update(differences_contract=abs(second) < abs(first),
                          same_direction=first*second > 0,
                          difference_ratio=None if first == 0 else abs(second/first),
                          half_to_quarter_within_1ml=abs(second) <= protocol["adjacent_difference_limit_ml"])
    write(WORK / "comparison.json", comparison)
    with (WORK / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for row, (ts, volumes) in zip(rows, curves):
        axes[0].plot(ts, volumes, label=f"dt × {row['dt_scale']:g}: {row['receiver_ml']:.2f} mL")
    axes[0].set(xlabel="Replay time (s)", ylabel="Receiver volume (mL)")
    axes[0].legend(fontsize=8)
    axes[1].plot([r["dt_s"]*1e6 for r in rows], [r["receiver_ml"] for r in rows], "o-")
    axes[1].set(xlabel="MPM time step (µs)", ylabel="Final receiver volume (mL)")
    for row in rows:
        axes[1].annotate(f"{row['receiver_ml']:.2f}",
            (row["dt_s"]*1e6, row["receiver_ml"]), xytext=(0, 7), textcoords="offset points", ha="center")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle("Recorded 60° replay · viscosity and contact fixed · " + ("complete" if complete else "in progress"))
    fig.savefig(WORK / "comparison.png", dpi=180)
    plt.close(fig)
    lines = ["# Recorded 60-degree time-step diagnostic", "",
             "Only the time step changes. Viscosity 3.4392377844 Pa s, source contact 0.117, grid 160, 229280 particles, geometry and recorded motion stay fixed.",
             "Fresh settling uses each run's time step and the same stopping rule. This tests the full replay workflow, including its initial settling.",
             "No fitting and no validation measurements are used. This does not certify the target table or real-robot accuracy.", "",
             "| Time-step factor | Final receiver (mL) | Change from previous (mL) |",
             "| --- | --- | --- |"]
    for r in rows:
        delta = "—" if r["difference_from_previous_ml"] is None else f"{r['difference_from_previous_ml']:+.3f}"
        lines.append(f"| {r['dt_scale']:g} | {r['receiver_ml']:.3f} | {delta} |")
    if complete:
        lines += ["", f"Successive differences contract: {comparison['differences_contract']}. Half-to-quarter change within the predeclared 1 mL limit: {comparison['half_to_quarter_within_1ml']}.",
                  "A contracting difference alone does not establish the limiting solution. No extrapolated volume or recalibrated parameter is issued."]
    (WORK / "REPORT.md").write_text("\n".join(lines)+"\n")
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / "controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = prepare()
        report(protocol)
        if args.prepare:
            print("Frozen protocol and baseline checks passed", flush=True)
            return
        for scale in SCALES[1:]:
            destination = case_path(60., dt=scale)
            if (destination / "result.json").exists():
                result_for(scale)
                continue
            if destination.exists():
                raise RuntimeError(f"Incomplete earlier attempt preserved; review before retry: {destination}")
            gpu = wait_for_gpu(protocol, scale)
            validate()
            command = [sys.executable, "-u", "experiments/pour/pour_navier_reference.py",
                       "--wall", "original-separable", "--source-friction", str(CONTACT),
                       "--grid", "160", "--phase", "0", "--dt-scale", str(scale), "--device", "cuda:0"]
            # Restrict this child to the selected physical GPU by UUID.
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu["uuid"])
            log = WORK / "logs" / f"dt{scale:g}.log"
            started = time.monotonic()
            print(f"START dt={scale:g} GPU={gpu['index']}", flush=True)
            with log.open("w") as stream:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                try:
                    while process.poll() is None:
                        status("running", dt_scale=scale, gpu=gpu, child_pid=process.pid,
                               elapsed_s=time.monotonic()-started, log=str(log))
                        if time.monotonic()-started > protocol["per_run_timeout_s"]:
                            raise TimeoutError("Replay exceeded its two-hour limit")
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            pass
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            process.kill(); process.wait()
            if process.returncode:
                raise RuntimeError(f"Replay failed with exit {process.returncode}; inspect {log}")
            validate()
            result = result_for(scale)
            report(protocol)
            print(f"DONE dt={scale:g}: {result['receiver_ml']:.6f} mL", flush=True)
        comparison = report(protocol)
        status("completed_needs_review", completed_scales=SCALES,
               comparison_path=str(WORK / "comparison.json"))
        print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BlockingIOError:
        raise SystemExit("A follow-up controller already holds the lock")
    except Exception as error:
        status("failed", reason=str(error))
        raise
