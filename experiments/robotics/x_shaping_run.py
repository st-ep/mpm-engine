"""Run independent frozen X-planning pipelines with retained subprocess logs."""
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

from experiments.robotics.x_shaping_study import MODELS, OUT, check_sources


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:0", "cuda:0"])
    p.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    args = p.parse_args(); folder = args.out.resolve(); check_sources(folder)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logs = folder/"logs"; logs.mkdir(exist_ok=True)
    assignments = {m: args.devices[i % len(args.devices)] for i, m in enumerate(args.models)}
    (logs/f"devices_{stamp}.json").write_text(json.dumps(assignments, indent=2)+"\n")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def pipeline(model):
        for stage in ["plan", "evaluate", "numerics"]:
            start = time.monotonic()
            cmd = [sys.executable, "-m", "experiments.robotics.x_shaping_study", stage,
                   "--out", str(folder), "--model", model, "--device", assignments[model]]
            with (logs/f"{stage}_{model}_{stamp}.log").open("w") as log:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            print(json.dumps(dict(model=model, stage=stage, code=result.returncode,
                                   seconds=time.monotonic()-start)), flush=True)
            if result.returncode:
                raise RuntimeError(f"Failed {stage} {model}; see logs")
        return model

    with ThreadPoolExecutor(max_workers=len(args.devices)) as pool:
        for future in as_completed([pool.submit(pipeline, model) for model in args.models]):
            print("Completed", future.result(), flush=True)


if __name__ == "__main__":
    main()
