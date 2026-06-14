"""Single audited import shim for the warp-mpm fork (replaces all sys.path hacks).

The MPM kernels live in the sibling fork at <repo>/wp-mpm/warp-mpm/ (a git submodule of
the parent project). It is reached once here, behind an env override, so the rest of the
package does a normal `from warpmpm._vendor import MPM_Simulator_WARP`. Set WARPMPM_FORK
to point elsewhere (e.g. a pip-installed copy) to drop the relative path entirely.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _fork_path() -> Path:
    env = os.environ.get("WARPMPM_FORK")
    if env:
        return Path(env).expanduser().resolve()
    # src/warpmpm/_vendor.py -> warpmpm -> src -> mpm_engine -> video2sim
    return Path(__file__).resolve().parents[3] / "wp-mpm" / "warp-mpm"


_FORK = _fork_path()
if not (_FORK / "mpm_solver_warp.py").exists():
    raise ImportError(
        f"warp-mpm fork not found at {_FORK}. Set WARPMPM_FORK to the warp-mpm directory."
    )
if str(_FORK) not in sys.path:
    sys.path.insert(0, str(_FORK))

from mpm_solver_warp import MATERIAL_NAME_TO_ID, MPM_Simulator_WARP

__all__ = ["MATERIAL_NAME_TO_ID", "_FORK", "MPM_Simulator_WARP"]
