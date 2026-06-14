"""warpmpm: a fast, modular Warp-MPM engine for robot manipulation of deformable and
granular media (dough first; terrain / rovers next). Sparse-grid + implicit core,
primitive-SDF robot contact with per-link wrench readout, MuJoCo coupling (Isaac-ready)."""
from __future__ import annotations

from warpmpm.core.solver import GridConfig, Solver

__version__ = "0.0.1"
__all__ = ["GridConfig", "Solver", "__version__"]
