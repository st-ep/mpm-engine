"""Run independent model pipelines with explicit device assignment and logs.

Example: --devices cuda:1 cuda:1 cuda:1 cuda:0 cuda:0. Use one device to
run all five pipelines sequentially. This never prepares or overwrites inputs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.robotics.hex_shaping_surface_study import OUT, MODELS, check_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--devices", nargs="+", default=["cuda:1"])
    args = parser.parse_args(); folder = args.out.resolve(); check_sources(folder)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logs = folder / "logs"; logs.mkdir(exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    assignments = {model: args.devices[i % len(args.devices)] for i, model in enumerate(MODELS)}
    (logs / f"devices_{stamp}.json").write_text(json.dumps(assignments, indent=2)+"\n")

    def pipeline(model):
        for stage in ["plan", "calibrate", "selfcheck", "execute", "numerics"]:
            started = time.monotonic()
            cmd = [sys.executable, "-m", "experiments.robotics.hex_shaping_surface_study",
                   stage, "--out", str(folder), "--model", model, "--device", assignments[model]]
            with (logs / f"{stage}_{model}_{stamp}.log").open("w") as log:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            print(json.dumps(dict(model=model, stage=stage, device=assignments[model],
                                  code=result.returncode, seconds=time.monotonic()-started)), flush=True)
            if result.returncode:
                raise RuntimeError(f"{stage} {model} failed; see {logs}")
        return model

    with ThreadPoolExecutor(max_workers=len(args.devices)) as pool:
        for future in as_completed([pool.submit(pipeline, model) for model in MODELS]):
            print("Completed", future.result(), flush=True)


if __name__ == "__main__":
    main()
