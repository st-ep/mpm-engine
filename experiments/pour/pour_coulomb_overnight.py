"""Detached target search with the one-pour weak viscosity/contact model frozen.

Only forward MPM volumes enter angle selection. Prior experimental validation
outcomes and retrospective forecasts are never read. Numerical checks diagnose
the selected commands; they do not refit the material/contact model.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import numpy as np

from experiments.pour.pour_navier_reference import ETA, OUT, ROOT

WORK = OUT / "overnight_targets"
TARGETS = [60, 80, 100, 120, 140, 160]
ANCHORS = [float(a) for a in range(48, 62)]
CONTACT = .117


class ProvenanceError(RuntimeError):
    pass


class CaseFailure(RuntimeError):
    pass


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def case_path(angle, grid=160, phase=0., dt=1.):
    label = "recorded60" if angle == 60. else f"planned{angle:g}"
    return OUT / "replays/original-separable" / (
        f"{label}_bNA_n{grid}_phase{phase:g}_dt{dt:g}_mu{CONTACT:.6f}")


def numerical_problem(result):
    values = [result.get(k, float("nan")) for k in
              ["receiver_ml", "outside_ml", "tail_variation_ml"]]
    if not np.isfinite(values).all() or not 0 <= values[0] <= 300:
        return "Nonfinite or impossible volume"
    if values[1] < 0 or values[1] > 3:
        return "More than 3 mL outside the cups, or invalid particle ledger"
    if values[2] < 0 or values[2] > .5:
        return "Receiver has not settled within 0.5 mL"
    return None


def next_angle(points, target, attempted=()):
    """Propose within an observed increasing bracket; retain all raw outcomes."""
    points = sorted(points)
    used = {round(a, 2) for a, _ in points} | {round(a, 2) for a in attempted}
    brackets = [(b[0]-a[0], a, b) for a, b in zip(points[:-1], points[1:])
                if a[1] < target < b[1] and b[0] > a[0]]
    if not brackets:
        raise CaseFailure(f"No simulated increasing bracket for {target} mL")
    for _, low, high in sorted(brackets):
        fraction = np.clip((target-low[1])/(high[1]-low[1]), .1, .9)
        proposed = round(low[0] + fraction*(high[0]-low[0]), 2)
        if low[0] < proposed < high[0] and proposed not in used:
            return proposed
        ticks = np.arange(int(round(low[0]*100))+1, int(round(high[0]*100))) / 100
        available = [float(a) for a in ticks if round(float(a), 2) not in used]
        if available:
            return min(available, key=lambda a: abs(a-.5*(low[0]+high[0])))
    raise CaseFailure("No untried angle at the declared 0.01-degree precision")


def check_result_provenance(result, angle, grid, phase, dt):
    expected = dict(eta_pa_s=ETA, source_coulomb_friction=CONTACT, initial_volume_ml=300.,
                    source_wall="original-separable", n_grid=grid, phase_xz_cells=phase,
                    dt_scale=dt, identification_episode="09-04-60-2s")
    if any(result.get(k) != v for k, v in expected.items()):
        raise ProvenanceError("Forward result differs from the frozen model/configuration")
    observed_angle = result.get("planned_angle_deg")
    if (60. if observed_angle is None else observed_angle) != angle:
        raise ProvenanceError("Forward result belongs to a different command")
    if result.get("validation_outcomes_used") != [] or result.get("measured_endpoints_read_by_forward_runner") != []:
        raise ProvenanceError("Forward runner must not read experimental endpoints")
    for relative, expected_hash in result["input_sha256"].items():
        if digest(ROOT / relative) == expected_hash:
            continue
        snapshot = OUT / "code_snapshots" / f"pour_navier_reference_{expected_hash}.py"
        if (relative != "experiments/pour/pour_navier_reference.py" or not snapshot.exists()
                or digest(snapshot) != expected_hash):
            raise ProvenanceError(f"Changed forward input: {relative}")


class Search:
    def __init__(self):
        self.protocol = json.loads((WORK / "protocol.json").read_text())
        if (self.protocol["eta_pa_s"] != ETA or self.protocol["contact_coefficient"] != CONTACT
                or self.protocol["targets_ml"] != TARGETS or self.protocol["validation_outcomes_used"] != []):
            raise ProvenanceError("Unexpected overnight protocol")
        state_path = WORK / "state.json"
        self.state = json.loads(state_path.read_text()) if state_path.exists() else dict(
            started_unix=time.time(), cases={}, targets={}, checks={}, errors=[], new_runs=0)
        self.validate_frozen_inputs()
        for entry in self.state["cases"].values():
            if entry["status"] != "complete":
                continue
            path = ROOT / entry["result_path"]
            if digest(path) != entry["result_sha256"]:
                raise ProvenanceError("Saved overnight result changed")
            result = json.loads(path.read_text())
            check_result_provenance(result, entry["angle_deg"], entry["grid"], entry["phase"], entry["dt_scale"])
            if numerical_problem(result) or result["receiver_ml"] != entry["receiver_ml"]:
                raise ProvenanceError("Saved overnight state disagrees with its forward result")

    def validate_frozen_inputs(self):
        if digest(WORK / "protocol.json") != (WORK / "protocol.sha256").read_text().strip():
            raise ProvenanceError("Overnight protocol changed")
        for relative, expected in self.protocol["frozen_input_sha256"].items():
            if digest(ROOT / relative) != expected:
                raise ProvenanceError(f"Frozen overnight input changed: {relative}")

    def status(self, stage, **fields):
        write(WORK / "state.json", self.state)
        write(WORK / "status.json", dict(stage=stage, updated_unix=time.time(),
            completed_targets=sorted(map(int, self.state["targets"])),
            completed_cases=sum(c["status"] == "complete" for c in self.state["cases"].values()),
            new_runs=self.state["new_runs"], **fields))

    def case(self, angle, grid=160, phase=0., dt=1.):
        if not self.protocol["angle_bounds_deg"][0] <= angle <= self.protocol["angle_bounds_deg"][1]:
            raise ProvenanceError("Angle exceeds the overnight protocol")
        self.validate_frozen_inputs()
        path = case_path(angle, grid, phase, dt)
        key = path.name
        old = self.state["cases"].get(key, {})
        if old.get("status") == "failed":
            raise CaseFailure(old["reason"])
        result_path = path / "result.json"
        if not result_path.exists():
            if self.state["new_runs"] >= self.protocol["max_new_runs"]:
                raise RuntimeError("Overnight simulation-count budget exhausted")
            if time.time()-self.state["started_unix"] > self.protocol["wall_budget_s"]-2700:
                raise RuntimeError("Insufficient overnight budget to start another replay")
            if path.exists():
                archive = WORK / "interrupted_runs" / f"{key}_{time.time_ns()}"
                archive.parent.mkdir(exist_ok=True)
                path.rename(archive)
                self.state["errors"].append(dict(case=key, reason="Interrupted prior attempt preserved", archive=str(archive)))
            log = WORK / "logs" / f"{key}.log"
            command = [sys.executable, "-u", "experiments/pour/pour_navier_reference.py",
                "--wall", "original-separable", "--source-friction", str(CONTACT),
                "--grid", str(grid), "--phase", str(phase), "--dt-scale", str(dt),
                "--max-angle", "62", "--device", "cuda:1"]
            if angle != 60.:
                command += ["--angle", str(angle)]
            self.state["new_runs"] += 1
            print(f"START {key}", flush=True)
            self.status("running", case=key, command_angle_deg=angle, grid=grid,
                        phase=phase, dt_scale=dt, log=str(log))
            with log.open("w") as stream:
                process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                deadline = time.monotonic() + self.protocol["per_run_timeout_s"]
                while process.poll() is None:
                    if time.monotonic() > deadline:
                        process.terminate()
                        try:
                            process.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            process.kill(); process.wait()
                        break
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        self.status("running", case=key, command_angle_deg=angle, grid=grid,
                                    phase=phase, dt_scale=dt, log=str(log), child_pid=process.pid)
                returncode = process.returncode
            if returncode != 0 or not result_path.exists():
                reason = f"Replay exited {returncode}; inspect {log}"
                self.state["cases"][key] = dict(status="failed", reason=reason)
                self.status("case_failed", case=key, reason=reason)
                raise CaseFailure(reason)
        result = json.loads(result_path.read_text())
        check_result_provenance(result, angle, grid, phase, dt)
        problem = numerical_problem(result)
        entry = dict(status="failed" if problem else "complete", reason=problem,
            angle_deg=angle, grid=grid, phase=phase, dt_scale=dt,
            result_path=str(result_path.relative_to(ROOT)), result_sha256=digest(result_path),
            receiver_ml=result["receiver_ml"], outside_ml=result["outside_ml"],
            tail_variation_ml=result["tail_variation_ml"])
        self.state["cases"][key] = entry
        self.status("case_complete" if not problem else "case_failed", case=key, **entry)
        if problem:
            raise CaseFailure(problem)
        print(f"DONE {key}: {result['receiver_ml']:.4f} mL", flush=True)
        return entry

    def try_case(self, *args, **kwargs):
        try:
            return self.case(*args, **kwargs)
        except CaseFailure as error:
            self.state["errors"].append(dict(arguments=list(args), options=kwargs, reason=str(error)))
            self.status("continuing_after_case_failure", reason=str(error))
            return None

    def production(self):
        return sorted([c for c in self.state["cases"].values()
            if c["status"] == "complete" and (c["grid"], c["phase"], c["dt_scale"]) == (160, 0., 1.)],
            key=lambda c:c["angle_deg"])

    def target(self, target):
        if str(target) in self.state["targets"]:
            return
        attempted = []
        for attempt in range(self.protocol["max_trials_per_target"]+1):
            samples = self.production()
            best = min(samples, key=lambda c:abs(c["receiver_ml"]-target))
            if abs(best["receiver_ml"]-target) <= self.protocol["target_tolerance_ml"]:
                self.state["targets"][str(target)] = dict(target_ml=target,
                    command_angle_deg=best["angle_deg"], tilt_duration_s=best["angle_deg"]/10.,
                    predicted_ml=best["receiver_ml"], prediction_error_ml=best["receiver_ml"]-target,
                    result_path=best["result_path"], result_sha256=best["result_sha256"])
                self.status("target_selected", target_ml=target, selected=self.state["targets"][str(target)])
                return
            if attempt == self.protocol["max_trials_per_target"]:
                break
            angle = next_angle([(c["angle_deg"], c["receiver_ml"]) for c in samples], target, attempted)
            attempted.append(angle)
            self.try_case(angle)
        raise CaseFailure(f"No ±1 mL target match for {target} mL within the trial budget")

    def comparisons(self):
        for target in TARGETS:
            selected = self.state["targets"].get(str(target))
            if not selected:
                continue
            tasks = [("grid_192", dict(grid=192), self.protocol["grid_difference_limit_ml"]),
                     ("half_cell_shift", dict(phase=.5), self.protocol["phase_difference_limit_ml"])]
            if target in self.protocol["half_dt_targets_ml"]:
                tasks += [("half_time_step", dict(dt=.5), self.protocol["dt_difference_limit_ml"])]
            for kind, options, limit in tasks:
                key = f"{target}_{kind}"
                if key in self.state["checks"]:
                    continue
                result = self.try_case(selected["command_angle_deg"], **options)
                difference = None if result is None else result["receiver_ml"]-selected["predicted_ml"]
                self.state["checks"][key] = dict(target_ml=target, check=kind,
                    command_angle_deg=selected["command_angle_deg"], difference_ml=difference,
                    limit_ml=limit, passed=difference is not None and abs(difference) <= limit,
                    result=result)
                self.status("numerical_check_complete", check=self.state["checks"][key])
                self.report()

    def report(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.interpolate import PchipInterpolator

        samples = self.production()
        rows = [self.state["targets"][str(t)] for t in TARGETS if str(t) in self.state["targets"]]
        checks = list(self.state["checks"].values())
        by_angle = {c["angle_deg"]:c for c in samples}
        anchors = [by_angle[a] for a in ANCHORS if a in by_angle]
        monotone = len(anchors) >= 2 and all(b["receiver_ml"] > a["receiver_ml"]
            for a,b in zip(anchors[:-1], anchors[1:]))
        all_anchors = len(anchors) == len(ANCHORS)
        expected_checks = 2*len(TARGETS)+len(self.protocol["half_dt_targets_ml"])
        matched = (len(rows) == len(TARGETS)
                   and all(np.isfinite(r["predicted_ml"]) and abs(r["predicted_ml"]-r["target_ml"]) <= 1. for r in rows)
                   and all(b["command_angle_deg"] > a["command_angle_deg"] for a,b in zip(rows[:-1],rows[1:])))
        ready = (matched and all_anchors and monotone and not self.state.get("fatal_error")
                 and len(checks) == expected_checks and all(c["passed"] for c in checks))
        failed = [c for c in checks if not c["passed"]]
        label = "Robot validation pending" if ready else (
            "Numerical checks require review" if failed or (all_anchors and not monotone) else "Numerical checks pending")
        write(WORK / "predictions.json", dict(model=self.protocol, predictions=rows,
            numerical_checks=checks, all_curve_anchors_complete=all_anchors,
            curve_anchors_strictly_increasing=monotone, handoff_ready=ready,
            validation_outcomes_used=[], physical_validation_done=False, status=label))
        text = io.StringIO()
        writer = csv.writer(text)
        writer.writerow(["target_ml", "command_angle_deg", "tilt_duration_s"])
        for row in rows:
            writer.writerow([row["target_ml"], f"{row['command_angle_deg']:.2f}", f"{row['tilt_duration_s']:.3f}"])
        (WORK / "angle_table.csv").write_text(text.getvalue())
        if samples:
            fig, ax = plt.subplots(figsize=(8.6, 5.4), constrained_layout=True)
            if len(anchors) >= 2:
                aa = np.array([c["angle_deg"] for c in anchors])
                vv = np.array([c["receiver_ml"] for c in anchors])
                if monotone:
                    xx = np.linspace(aa.min(), aa.max(), 600)
                    ax.plot(xx, PchipInterpolator(aa,vv)(xx), color="#2166ac", label="MPM curve (interpolated anchors)")
                else:
                    ax.plot(aa, vv, color="#2166ac", label="MPM anchors (reversal retained)")
            shown = [c for c in samples if 48 <= c["angle_deg"] <= 62]
            ax.scatter([c["angle_deg"] for c in shown], [c["receiver_ml"] for c in shown],
                       s=18, color="#2166ac", alpha=.55, label="Full MPM evaluations")
            for i, row in enumerate(rows):
                angle, volume = row["command_angle_deg"], row["predicted_ml"]
                ax.scatter([angle], [volume], color="#bc3d2b", s=45, zorder=4)
                ax.annotate(f"{row['target_ml']} mL · {angle:.2f}°", (angle, volume),
                    xytext=(7 if i<3 else -7, 12 if i%2 else -19), textcoords="offset points",
                    fontsize=9, ha="left" if i<3 else "right")
            ax.set(xlabel="Commanded pour angle (degrees)", ylabel="Settled receiver volume (mL)")
            ax.grid(alpha=.2); ax.legend(fontsize=8, loc="upper left")
            ax.set_title(f"300 mL initial fill · 2 s hold\n{label}", fontsize=12)
            fig.savefig(WORK / "angle_table.png", dpi=180); plt.close(fig)
        lines = ["# Overnight MPM target predictions", "", label, "",
            "One 60-degree pour supplied the PDF weak spout viscosity (3.4392377844 Pa s) and numerical contact calibration (0.117).",
            "Geometry, grasp, material, contact, timing rule, and production grid stayed frozen during target selection.",
            "No earlier validation outcome or retrospective forecast enters this program.", "",
            "| Target (mL) | Angle (degrees) | MPM (mL) | Error (mL) |", "| --- | --- | --- | --- |"]
        lines += [f"| {r['target_ml']} | {r['command_angle_deg']:.2f} | {r['predicted_ml']:.2f} | {r['prediction_error_ml']:+.2f} |" for r in rows]
        lines += ["", f"Numerical comparisons completed: {len(checks)}/{expected_checks}; failed: {len(failed)}.",
                  "No numerical comparison is a bound on real-robot error. A new prospective experiment is required.",
                  "Raw evaluations, failures, provenance and comparisons remain in state.json and predictions.json.",
                  "Angle selection always uses an actual production-grid MPM evaluation, not the plotted interpolation."]
        (WORK / "SUMMARY.md").write_text("\n".join(lines)+"\n")
        if not ready and (WORK / "philip_handoff.zip").exists():
            # Preserve any superseded handoff as an audit artifact, never leave a
            # failed/provisional run carrying a current ready-to-use ZIP.
            (WORK / "philip_handoff.zip").rename(WORK / f"superseded_handoff_{time.time_ns()}.zip")
        if ready:
            notes = """SIX PROSPECTIVE POURING TESTS

Use angle_table.csv: target volume, commanded pour angle, tilt-command duration.
Start each run with 300 mL and an empty receiver. Hold 2.0 s after the tilt
acknowledgement; use a 2.0 s return command and the same planner/settings.
Keep the cup mounting and liquid conditions consistent with the identification pour.
Record joints, action timestamps, video, and the independently measured final volume.

These commands were evaluated with full MPM using one-pour weak-form viscosity
identification plus same-pour numerical contact calibration. The graph connects
simulated anchors; each marked command has its own full replay. Robot accuracy
remains to be measured. Run the set before changing settings against its outcomes.
"""
            (WORK / "START_HERE.txt").write_text(notes)
            temporary = WORK / "philip_handoff.tmp.zip"
            with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as z:
                for name in ["angle_table.csv", "angle_table.png", "START_HERE.txt"]:
                    z.write(WORK / name, arcname=name)
            temporary.replace(WORK / "philip_handoff.zip")
        return ready

    def run(self):
        for angle in [45., 47.56, 48.18, 48.37, 49.43, 60.]:
            self.try_case(angle)
        for angle in [48., 51., 54., 57., 61.]:
            self.try_case(angle)
            self.report()
        if max(c["receiver_ml"] for c in self.production()) < max(TARGETS)-1:
            self.try_case(62.)
        for target in TARGETS:
            try:
                self.target(target)
            except CaseFailure as error:
                self.state["errors"].append(dict(target_ml=target, reason=str(error)))
            self.report()
        for angle in ANCHORS:
            self.try_case(angle)
            self.report()
        self.comparisons()
        ready = self.report()
        self.status("completed" if ready else "completed_needs_review", handoff_ready=ready)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate frozen inputs without launching simulations")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "logs").mkdir(exist_ok=True)
    with (WORK / "controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        search = Search()
        if args.check:
            print("Frozen input hashes and protocol passed; no simulation launched.")
            return
        try:
            search.run()
        except Exception as error:
            search.state["fatal_error"] = str(error)
            search.state["errors"].append(dict(fatal=True, type=type(error).__name__, reason=str(error)))
            search.status("stopped_needs_review", reason=str(error))
            search.report()
            raise


if __name__ == "__main__":
    main()
