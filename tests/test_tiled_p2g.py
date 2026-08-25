"""Tiled P2G scatter (wp.tile shared-memory staging): the opt-in kernel must deposit
the same grid momentum and mass as the per-particle scatter. Atomic order differs
between the two kernels, so node values match to float tolerance; conserved totals
match tightly because the stress and affine terms cancel over the stencil (quadratic
B-splines reproduce constants and linears), leaving total grid momentum equal to
sum m v and total grid mass equal to sum m up to rounding."""
from __future__ import annotations

import numpy as np
import pytest

from warpmpm import GridConfig, Solver
from warpmpm.materials import elastic, newtonian


def _scene(tiled=False, sort_interval=1, n=4000, seed=3, material=None, mls=False):
    grid = GridConfig(n_grid=32, grid_lim=0.4)
    rng = np.random.default_rng(seed)
    pts = (rng.random((n, 3), dtype=np.float32) * 0.16 + 0.12).astype(np.float32)
    vol = np.full(n, 1.0e-7, np.float32)
    s = Solver(grid=grid, device="cpu", fused=False, sort_interval=sort_interval,
               tiled_p2g=tiled).load_particles(pts, vol)
    s.set_material(material if material is not None
                   else newtonian(eta=2.0, density=1000.0, bulk_modulus=2.0e5))
    if mls:
        s._sim.set_grid_semantics(mls_transfer=True)
    s.add_plane((0, 0, 3 * grid.dx), (0, 0, 1), "separate", friction=0.3)
    s.add_domain_walls()
    # a bulk velocity keeps the momentum totals far from zero
    v0 = rng.standard_normal((n, 3)).astype(np.float32) * 0.05
    v0[:, 2] -= 0.3
    s.set_v(v0)
    return s


def _grids(s):
    st = s._sim.mpm_state
    return st.grid_m.numpy().copy(), st.grid_v_in.numpy().copy()


def _compare_deposits(ref, tld, ticks=3, dt=2.0e-4, substeps=6):
    """Run both solvers on the plain path (bitwise identical on CPU), switch one to
    tiled, take one substep each, and compare the raw P2G deposits node by node."""
    for _ in range(ticks):
        ref.step(dt, substeps)
        tld.step(dt, substeps)
    np.testing.assert_array_equal(tld.x(), ref.x())  # same kernels, same bits
    tld.tiled_p2g = True
    m = tld._sim.mpm_state.particle_mass.numpy().copy()
    v = tld.v().copy()  # v() aliases the live buffer on CPU; freeze the pre-step value
    ref.step(dt, 1)
    tld.step(dt, 1)
    gm_r, gv_r = _grids(ref)
    gm_t, gv_t = _grids(tld)

    m_scale = np.abs(gm_r).max()
    v_scale = np.abs(gv_r).max()
    np.testing.assert_allclose(gm_t, gm_r, rtol=1e-5, atol=1e-9 * m_scale)
    np.testing.assert_allclose(gv_t, gv_r, rtol=1e-5, atol=1e-9 * v_scale)

    # conservation on the tiled deposits: totals against the particle sums, both
    # accumulated in float64 so the check's own summation rounding stays below
    # the tolerance (the deposits themselves are float32)
    m64 = m.astype(np.float64)
    mass_err = abs(gm_t.astype(np.float64).sum() - m64.sum()) / m64.sum()
    mom = (m64[:, None] * v.astype(np.float64)).sum(0)
    mom_err = np.abs(gv_t.astype(np.float64).sum(axis=(0, 1, 2)) - mom) / np.abs(mom)
    assert mass_err < 1e-6, f"mass conservation {mass_err:.2e}"
    assert mom_err.max() < 1e-6, f"momentum conservation {mom_err}"
    rel = np.abs(gv_t - gv_r).max() / v_scale
    return rel


def test_tiled_matches_plain_deposits():
    ref = _scene(sort_interval=1)
    tld = _scene(sort_interval=1)
    rel = _compare_deposits(ref, tld)
    assert rel < 1e-5


def test_tiled_matches_plain_deposits_mls_apic():
    # mls_transfer routes the stress through the affine channel; three ticks of
    # motion first so particle_C is nonzero and the APIC term is exercised
    ref = _scene(material=elastic(E=5.0e4), mls=True)
    tld = _scene(material=elastic(E=5.0e4), mls=True)
    rel = _compare_deposits(ref, tld)
    assert rel < 1e-5


def test_stale_tables_route_drifted_particles_through_fallback():
    # tables rebuild only on sort ticks; with sort_interval=4 the intermediate
    # ticks run on stale tables and drifted particles take the global-atomic path
    ref = _scene(sort_interval=0)
    tld = _scene(sort_interval=4, tiled=True)
    dt, substeps = 2.0e-4, 6
    for _ in range(3):
        ref.step(dt, substeps)
        tld.step(dt, substeps)
    # confirm the fallback is exercised: some particles left their table window
    tb = tld._sim.tiled_p2g_blocks
    starts = tb["start"].numpy()
    counts = tb["count"].numpy()
    lo = tb["lo"].numpy()
    base = np.floor(tld.x() / tld.grid.dx - 0.5).astype(np.int64)
    drifted = 0
    for s0, c0, l0 in zip(starts, counts, lo):
        off = base[s0:s0 + c0] - l0
        drifted += int(np.count_nonzero(((off < 0) | (off > 3)).any(axis=1)))
    assert drifted > 0, "no particle left its block window; the test is inert"
    xr, xt = ref.x(), tld.x()
    np.testing.assert_allclose(xt[np.lexsort(xt.T)], xr[np.lexsort(xr.T)],
                               rtol=1e-4, atol=1e-6)
    assert np.isfinite(xt).all()


def test_tiled_requires_block_sort():
    s = _scene(sort_interval=0, tiled=True, n=500)
    with pytest.raises(RuntimeError, match="sort_interval"):
        s.step(2.0e-4, 1)


def test_tiled_requires_split_pipeline():
    s = _scene(sort_interval=1, tiled=True, n=500)
    s.fused = True
    with pytest.raises(RuntimeError, match="fused"):
        s.step(2.0e-4, 1)


def test_flag_off_leaves_plain_launch():
    s = _scene(sort_interval=1, n=500)
    s.step(2.0e-4, 2)
    assert s._sim.tiled_p2g_blocks is None
