"""Experimental SDF Robin extrapolation for u_t = b * partial_n u_t.

Extrapolate the tangential velocity from readable fluid-side nodes to the
physical wall. Read nodes must be outside the entire write band, avoiding
in-place GPU read/write races. Signed distances set the interpolation factor;
the grid spacing is not fitted to an experimental endpoint.
"""
import numpy as np
import warp as wp

from warpmpm.kernels.mpm_solver_warp import sdf_trilerp, sdf_trilerp_vec
from warpmpm.kernels.warp_utils import MPMModelStruct, MPMStateStruct, SDFCollider


def install_navier_robin_wall(solver, handle, *, slip_length_m, eta_pa_s):
    if not np.isfinite(slip_length_m) or slip_length_m < 0:
        raise ValueError("Slip length must be finite and nonnegative")
    sim = solver._sim
    if handle != len(sim.collider_params) - 1 or len(sim.grid_postprocess) != len(sim.collider_params):
        raise ValueError("Install immediately after the SDF collider")
    b = wp.constant(float(slip_length_m))

    @wp.kernel
    def contact(time: float, dt: float, state: MPMStateStruct, model: MPMModelStruct,
                param: SDFCollider, lo: wp.vec3i):
        i, j, k = wp.tid()
        i += lo[0]
        j += lo[1]
        k += lo[2]
        mass = state.grid_m[i, j, k]
        if mass <= 0.0 or time < param.start_time or time >= param.end_time:
            return
        x = wp.vec3(float(i), float(j), float(k)) * model.dx
        rel = x - param.center
        body = wp.quat_rotate_inv(param.quat, rel)
        fidx = (body - param.origin) / param.cell
        last = float(param.res - 1)
        if (fidx[0] < 0.0 or fidx[1] < 0.0 or fidx[2] < 0.0
                or fidx[0] > last or fidx[1] > last or fidx[2] > last):
            return
        sd = sdf_trilerp(param.sdf_val, param.res, fidx)
        if sd > param.band:
            return
        g = sdf_trilerp_vec(param.sdf_grad, param.res, fidx)
        if wp.length(g) < 1.0e-12:
            return
        normal = wp.quat_rotate(param.quat, wp.normalize(g))
        wall = param.velocity + wp.cross(param.omega, rel)
        before = state.grid_v_out[i, j, k]
        vn = wp.max(wp.dot(before - wall, normal), 0.0)
        target = (x + (param.band + model.dx - sd) * normal) / model.dx
        base = wp.vec3i(int(wp.floor(target[0])), int(wp.floor(target[1])), int(wp.floor(target[2])))
        fraction = target - wp.vec3(float(base[0]), float(base[1]), float(base[2]))
        total = float(0.0)
        exterior_distance = float(0.0)
        exterior_velocity = wp.vec3(0.0)
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    ii, jj, kk = base[0] + di, base[1] + dj, base[2] + dk
                    if (ii < 0 or jj < 0 or kk < 0 or ii >= state.grid_m.shape[0]
                            or jj >= state.grid_m.shape[1] or kk >= state.grid_m.shape[2]):
                        continue
                    sample_rel = wp.vec3(float(ii), float(jj), float(kk)) * model.dx - param.center
                    sample_body = wp.quat_rotate_inv(param.quat, sample_rel)
                    fi = (sample_body - param.origin) / param.cell
                    if (fi[0] < 0.0 or fi[1] < 0.0 or fi[2] < 0.0
                            or fi[0] > last or fi[1] > last or fi[2] > last):
                        continue
                    sample_sd = sdf_trilerp(param.sdf_val, param.res, fi)
                    if sample_sd <= param.band or state.grid_m[ii, jj, kk] <= 0.0:
                        continue
                    wi = fraction[0] if di == 1 else 1.0 - fraction[0]
                    wj = fraction[1] if dj == 1 else 1.0 - fraction[1]
                    wk = fraction[2] if dk == 1 else 1.0 - fraction[2]
                    weight = wi * wj * wk
                    sample_wall = param.velocity + wp.cross(param.omega, sample_rel)
                    exterior_velocity += weight * (state.grid_v_out[ii, jj, kk] - sample_wall)
                    exterior_distance += weight * sample_sd
                    total += weight
        tangent = before - wall - wp.dot(before - wall, normal) * normal
        if total > 1.0e-6:
            exterior_velocity /= total
            exterior_distance /= total
            ratio = wp.clamp((b + sd) / (b + exterior_distance), -1.0, 1.0)
            tangent = ratio * (exterior_velocity - wp.dot(exterior_velocity, normal) * normal)
        else:
            # No readable exterior velocity: a dry/thinner-than-grid contact.
            # Preserve its tangential motion rather than invent a measured shear.
            tangent = tangent
        after = wall + tangent + vn * normal
        impulse = mass * (before - after)
        wp.atomic_add(param.force, 0, impulse)
        wp.atomic_add(param.torque, 0, wp.cross(rel, impulse))
        state.grid_v_out[i, j, k] = after

    sim.grid_postprocess[-1] = contact
    return dict(method="SDF Robin extrapolation", slip_length_m=float(slip_length_m),
                eta_pa_s=float(eta_pa_s), boundary_relation="u_t = b * partial_n u_t",
                readable_exterior_rule="Mass-bearing nodes strictly outside write band",
                no_exterior_rule="Preserve tangential velocity; enforce normal contact")
