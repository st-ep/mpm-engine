"""Paper rendering entrypoint for the common 12 mm work-command protocol."""

import argparse
from pathlib import Path

# Configure the protocol before importing its shared audits and renderer.
from experiments.robotics.x_work_guard_study import OUT
from experiments.robotics.x_work_report import audit, diagnostics, report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["audit", "diagnostics", "report"])
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--dest", type=Path)
    p.add_argument("--render-out", type=Path, help="Separate figure output directory; preserves the simulation archive")
    a = p.parse_args()
    if a.stage == "audit":
        audit(a.out)
    elif a.stage == "diagnostics":
        diagnostics(a.out)
    else:
        report(a.out, a.dest, entrypoint=Path(__file__), render_out=a.render_out)
