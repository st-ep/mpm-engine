"""CPU-default Solver: a small typed wrapper over the warp-mpm fork.

Centralizes device handling (the fork hardcodes cuda in many signatures; on Apple
Silicon we run CPU), Warp init, and the common load/material/collider/step/export calls,
so scenes, tests, and the coupling backend never touch the raw fork or sys.path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp

from warpmpm._vendor import MPM_Simulator_WARP

_INITED = False


def _ensure_warp() -> None:
    global _INITED
    if not _INITED:
        wp.config.quiet = True
        wp.init()
        _INITED = True


@dataclass
class GridConfig:
    n_grid: int = 64
    grid_lim: float = 0.4  # cubic domain edge, metres

    @property
    def dx(self) -> float:
        return self.grid_lim / self.n_grid


@dataclass
class Solver:
    """Thin owner of one MPM_Simulator_WARP instance, CPU by default."""

    grid: GridConfig = field(default_factory=GridConfig)
    device: str = "cpu"
    _sim: Any = field(default=None, init=False, repr=False)
    _step: int = field(default=0, init=False, repr=False)

    def load_particles(self, pos: np.ndarray, vol: np.ndarray) -> Solver:
        import torch

        _ensure_warp()
        self._sim = MPM_Simulator_WARP(len(pos), device=self.device)
        self._sim.load_initial_data_from_torch(
            torch.from_numpy(pos.astype(np.float32)),
            torch.from_numpy(vol.astype(np.float32)),
            n_grid=self.grid.n_grid,
            grid_lim=self.grid.grid_lim,
            device=self.device,
        )
        return self

    def set_material(self, material, **overrides: float) -> Solver:
        """Accepts a composable warpmpm.materials.Material (preferred) or a fork material
        name string. Resolves to the fork's (name, params) and applies it."""
        if hasattr(material, "resolve"):
            name, params = material.resolve()
        else:
            name, params = str(material), {}
        params = {**params, **overrides}
        self._sim.set_parameters_dict(
            {"material": name, "g": [0.0, 0.0, -9.81], **params}, device=self.device
        )
        self._sim.finalize_mu_lam(device=self.device)
        return self

    def add_plane(self, point, normal, surface: str = "sticky", friction: float = 0.0) -> Solver:
        self._sim.add_surface_collider(tuple(point), tuple(normal), surface, friction=friction)
        return self

    def step(self, dt: float, substeps: int = 1) -> Solver:
        for _ in range(substeps):
            self._sim.p2g2p(self._step, dt, device=self.device)
            self._step += 1
        return self

    # --- exports (numpy, off the hot path) -------------------------------------------
    def x(self) -> np.ndarray:
        return self._sim.export_particle_x_to_torch().cpu().numpy()

    def v(self) -> np.ndarray:
        return self._sim.export_particle_v_to_torch().cpu().numpy()

    def F(self) -> np.ndarray:
        return self._sim.export_particle_F_to_torch().cpu().numpy().reshape(-1, 3, 3)

    def stress(self) -> np.ndarray:
        return self._sim.export_particle_stress_to_torch().cpu().numpy().reshape(-1, 3, 3)

    @property
    def n_particles(self) -> int:
        return 0 if self._sim is None else self._sim.n_particles
