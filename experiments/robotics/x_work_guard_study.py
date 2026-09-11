"""Active X work-command study with a common 12 mm minimum opening.

This entrypoint configures the shared executor before simulation. The archived
4 mm experiment and its source files remain unchanged. Use this entrypoint
for every command on the guarded archive; run one protocol per process.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from experiments.robotics import x_work_control as control
from experiments.robotics import x_work_study as base

OUT = base.ROOT / "out/x_work_guard12_study_20260909"
control.MIN_GAP = .012
base.MIN_GAP = .012
base.OUT = OUT
base.PROTOCOL = dict(base.PROTOCOL,
    minimum_gap_m=.012,
    feedback="Close at 50 mm/s per finger; stop at the first 4 ms tick reaching commanded work, or at the common 12 mm minimum-opening guard",
    guard_reason="The 4 mm pilot left only 1.6 baseline grid cells between the fingers and generated detached reconstructed pieces in A under B's work commands. A 12 mm guard provides 4.8 baseline cells and remains below both models' intended stopping gaps.",
    guard_scope="Same software guard for both materials and both plans; not a measured Franka hardware travel limit",
    prior_guard_pilot="out/x_work_guard12_pilot_20260909; target, material laws, inherited plans, and work commands not adjusted")


def prepare(folder):
    base.prepare(folder)
    source = Path(__file__)
    rel = str(source.relative_to(base.ROOT))
    dest = folder / "source_snapshot" / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    manifest = folder / "source_sha256.json"
    hashes = json.loads(manifest.read_text())
    hashes[rel] = base.digest(source)
    base.save_json(manifest, hashes)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "predict", "selfcheck", "evaluate", "verify"])
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--material", choices=["A", "B"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--settings", choices=list(base.SETTINGS), nargs="+", default=list(base.SETTINGS))
    a = p.parse_args()
    if a.stage == "prepare":
        prepare(a.out)
    elif a.stage == "predict":
        base.predict(a.out, a.material, a.device)
    elif a.stage == "selfcheck":
        base.selfcheck(a.out, a.material, a.device)
    elif a.stage == "evaluate":
        base.evaluate(a.out, a.material, a.device, a.settings)
    else:
        base.verify(a.out)
