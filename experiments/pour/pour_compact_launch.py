"""Wait for verified cropping, then launch the frozen two-GPU diagnostic batch.

Run under a user systemd service so laptop disconnects do not cancel the work.
No table is exported and no experiment outcome selects the runs.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time

from experiments.pour.pour_compact_verification import ROOT, OUT, check_crop, verify_hashes, write


def main():
    protocol = json.loads((OUT / "refinement_protocol.json").read_text())
    verify_hashes(protocol["input_sha256"])
    deadline = time.monotonic() + 1800
    write(OUT / "launcher_status.json", dict(status="waiting_for_crop", automatic_robot_export=False))
    while True:
        status = subprocess.run(["systemctl", "--user", "show", "pour-compact-crop-check.service",
                                 "--property=ActiveState", "--value"],
                                check=True, capture_output=True, text=True).stdout.strip()
        if status not in ["active", "activating", "deactivating"]:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("Crop verification did not finish in 30 minutes; no refinement launched")
        time.sleep(10)
    crop = check_crop()
    print("Crop passed:", json.dumps(crop["differences"]), flush=True)
    services = []
    for phase, device in [(0., "cuda:0"), (.5, "cuda:1")]:
        verify_hashes(protocol["input_sha256"])
        unit = f"pour-compact-refine-phase{phase:g}"
        log = OUT / f"worker_phase{phase:g}.log"
        command = ["systemd-run", "--user", f"--unit={unit}",
                   "--description=Frozen one-video full-MPM numerical refinement",
                   f"--working-directory={ROOT}",
                   "--setenv=MUJOCO_GL=egl", "--setenv=OPENBLAS_NUM_THREADS=1",
                   "--setenv=OMP_NUM_THREADS=1", "--setenv=PYTHONPATH=.",
                   f"--property=StandardOutput=append:{log}",
                   f"--property=StandardError=append:{log}",
                   sys.executable, "-u", "-m", "experiments.pour.pour_compact_verification",
                   "--run-phase", str(phase), "--device", device]
        subprocess.run(command, cwd=ROOT, check=True)
        services.append(unit + ".service")
        write(OUT / "launcher_status.json", dict(status="launching", services=services,
                                                 automatic_robot_export=False))
    write(OUT / "launcher_status.json", dict(status="launched", services=services,
                                             automatic_robot_export=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write(OUT / "launcher_status.json", dict(status="failed", error=str(exc),
                                                 automatic_robot_export=False))
        raise
