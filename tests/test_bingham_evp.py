"""Bingham elasto-viscoplastic solid (fork "visplas", id 3): factory mapping + statics.

The regularized `newtonian.with_yield` FLUID realization of Bingham creeps at rest by
construction (measured tau-independent in the WorldBench shaping guardrail), so it can
never hold a shaped feature. The EVP realization — StVK elastic predictor + Perzyna
overstress return (viscoplasticity_return_mapping_with_StVK) — is elastic below yield
and must hold statics. Its flow-law conventions (steady shear tau = yield_stress/sqrt(3)
+ (plastic_viscosity/2)*gd, i.e. Y = sqrt(3)*tau_y, eta_p = 2*eta, fit to < 0.1%) are
measured by experiments/bingham_evp_couette.py; here we lock the factory mapping and the
statics property.
"""
from __future__ import annotations

import numpy as np
import pytest

from warpmpm import GridConfig, Solver, block, vonmises
from warpmpm.kernels import MATERIAL_NAME_TO_ID


def test_visplas_alias_and_factory_mapping():
    assert MATERIAL_NAME_TO_ID["visplas"] == 3          # the upstream "foam" slot
    name, params = vonmises(E=2.0e5, nu=0.3, yield_stress=3464.0).with_viscosity(
        80.0).resolve()
    assert name == "visplas"
    assert params["plastic_viscosity"] == 80.0
    assert params["yield_stress"] == 3464.0
    # without viscosity the base stays the rate-independent J2 solid
    name2, params2 = vonmises(E=2.0e5, nu=0.3, yield_stress=3464.0).resolve()
    assert name2 == "metal" and "plastic_viscosity" not in params2


@pytest.mark.slow
def test_evp_bingham_holds_statics():
    # A free-standing block whose yield stress exceeds its own ~rho*g*h must NOT creep:
    # elastic below yield. (The fluid realization slumps ~6 mm/s on this scene.)
    grid = GridConfig(n_grid=40, grid_lim=0.4)
    pos, vol, floor = block(grid, size=(0.10, 0.05, 0.05), ppc=2)
    s = Solver(grid=grid).load_particles(pos.copy(), vol)
    # shear yield 2 kPa -> fork params via the measured sqrt(3) / 2x conventions
    s.set_material(vonmises(E=2.0e5, nu=0.3, yield_stress=np.sqrt(3.0) * 2.0e3,
                            density=1000.0).with_viscosity(2.0 * 40.0))
    s.add_plane((0, 0, floor), (0, 0, 1), "sticky")
    seconds, dt, sub = 0.25, 2.0e-4, 4
    for _ in range(round(seconds / (dt * sub))):
        s.step(dt, sub)
    x = s.x()
    assert np.isfinite(x).all()
    top_drop_mm = (pos[:, 2].max() - x[:, 2].max()) * 1e3
    disp_p95_mm = np.quantile(np.linalg.norm(x - pos, axis=1), 0.95) * 1e3
    assert top_drop_mm < 1.5, f"EVP block creeps: top dropped {top_drop_mm:.2f} mm"
    assert disp_p95_mm < 1.5, f"EVP block creeps: p95 displacement {disp_p95_mm:.2f} mm"
