"""Manufactured Robin boundary fields and moving-wall momentum accounting."""
import numpy as np
import pytest
import warp as wp

from warpmpm import GridConfig, Solver, newtonian
from experiments.pour.pour_film_benchmark import plane_sdf
from experiments.pour.pour_navier_quadratic_wall import install_navier_quadratic_wall
from experiments.pour.pour_navier_traction_wall import install_navier_traction_wall


@pytest.mark.parametrize("phase,angle,slip", [(0., 0., .001), (.5, 0., .00025),
                                               (.25, 23., .004)])
def test_quadratic_robin_field_and_moving_wall(phase, angle, slip):
    length, n = .032, 24
    dx = length / n
    bottom = .008 + phase * dx
    solver = Solver(grid=GridConfig(n_grid=n, grid_lim=length), device="cpu")
    solver.load_particles(np.array([[.016, .016, .016]], dtype=np.float32),
                          np.array([dx**3], dtype=np.float32))
    solver.set_material(newtonian(eta=3.44, density=1260., bulk_modulus=9e5))
    theta = np.radians(angle)
    rot = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]])
    quaternion = (0., np.sin(theta/2), 0., np.cos(theta/2))
    wall_velocity, omega = np.array([.2, -.1, .03]), np.array([.1, -.2, .05])
    handle = solver.add_sdf_collider(plane_sdf(length, bottom), center=(0, 0, 0),
                                    quat=quaternion, velocity=wall_velocity, omega=omega,
                                    band=.5*dx, surface="separable", friction=0.)
    install_navier_quadratic_wall(solver, handle, slip_length_m=slip, eta_pa_s=3.44)
    sim = solver._sim
    shape = sim.mpm_state.grid_m.shape
    xyz = np.stack(np.meshgrid(*[np.arange(a)*dx for a in shape], indexing="ij"), -1)
    distance = (xyz @ rot)[..., 2] - bottom
    tangent, normal = rot[:, 0], rot[:, 2]
    wall = wall_velocity + np.cross(omega, xyz)
    speed = 15.*(slip + distance) - 600.*distance**2
    before = wall + speed[..., None]*tangent
    mass = ((distance >= -.4*dx) & (distance <= .012)).astype(np.float32) * 1e-4
    sim.mpm_state.grid_m.assign(mass)
    sim.mpm_state.grid_v_out.assign(before.astype(np.float32))
    dt = 1e-5
    wp.launch(sim.grid_postprocess[-1], dim=shape,
              inputs=[0., dt, sim.mpm_state, sim.mpm_model, sim.collider_params[handle], wp.vec3i(0, 0, 0)],
              device="cpu")
    after = sim.mpm_state.grid_v_out.numpy()
    interior = (mass > 0) & (distance <= .5*dx)
    for axis in range(3):
        interior &= (xyz[..., axis] > 4*dx) & (xyz[..., axis] < length - 4*dx)
    assert interior.sum() >= 4
    np.testing.assert_allclose(after[interior], before[interior], atol=2e-5, rtol=0.)
    # Every velocity change has an equal and opposite accumulated wall impulse.
    impulse = np.sum(mass[..., None] * (before.astype(np.float32) - after), axis=(0, 1, 2))
    np.testing.assert_allclose(sim.collider_params[handle].force.numpy()[0], impulse, atol=2e-8)


def test_invalid_slip_is_rejected_before_contact_is_installed():
    with pytest.raises(ValueError, match="nonnegative"):
        install_navier_quadratic_wall(None, 0, slip_length_m=-.001, eta_pa_s=3.44)


@pytest.mark.parametrize("slip", [.00025, .001, .004])
def test_traction_dissipates_relative_energy_and_accounts_for_wall_impulse(slip):
    length, n = .032, 16
    dx, bottom = length/n, .008
    solver = Solver(grid=GridConfig(n_grid=n, grid_lim=length), device="cpu")
    solver.load_particles(np.array([[.016, .016, .016]], dtype=np.float32),
                          np.array([dx**3], dtype=np.float32))
    solver.set_material(newtonian(eta=3.44, density=1260., bulk_modulus=9e5))
    wall_velocity, omega = np.array([.2, -.1, .03]), np.array([.1, -.2, .05])
    handle = solver.add_sdf_collider(plane_sdf(length, bottom), center=(0, 0, 0),
                                    velocity=wall_velocity, omega=omega,
                                    band=.5*dx, surface="separable", friction=0.)
    install_navier_traction_wall(solver, handle, slip_length_m=slip, eta_pa_s=3.44)
    sim = solver._sim
    shape = sim.mpm_state.grid_m.shape
    xyz = np.stack(np.meshgrid(*[np.arange(a)*dx for a in shape], indexing="ij"), -1)
    wall = wall_velocity + np.cross(omega, xyz)
    rng = np.random.default_rng(83)
    before = (wall + rng.normal(0, .15, size=(*shape, 3))).astype(np.float32)
    mass = rng.uniform(1e-7, 1e-5, size=shape).astype(np.float32)
    mass[xyz[..., 2] < bottom - dx] = 0
    sim.mpm_state.grid_m.assign(mass)
    sim.mpm_state.grid_v_out.assign(before)
    wp.launch(sim.grid_postprocess[-1], dim=shape,
              inputs=[0., .001, sim.mpm_state, sim.mpm_model, sim.collider_params[handle], wp.vec3i(0, 0, 0)],
              device="cpu")
    after = sim.mpm_state.grid_v_out.numpy()
    old_energy = .5*mass*np.sum((before-wall)**2, axis=-1)
    new_energy = .5*mass*np.sum((after-wall)**2, axis=-1)
    assert np.all(new_energy <= old_energy + 2e-12)
    assert new_energy.sum() < old_energy.sum()
    contact = (mass > 0) & (xyz[..., 2] <= bottom + .5*dx)
    assert np.min((after-wall)[..., 2][contact]) >= -1e-7
    impulse = mass[..., None]*(before-after)
    np.testing.assert_allclose(sim.collider_params[handle].force.numpy()[0], impulse.sum(axis=(0,1,2)), atol=2e-8)
    np.testing.assert_allclose(sim.collider_params[handle].torque.numpy()[0],
                               np.cross(xyz, impulse).sum(axis=(0,1,2)), atol=2e-9)
