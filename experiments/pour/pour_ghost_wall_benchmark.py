"""Experimental physical-wall extrapolation for the flat-film verification only.

Values inside the wall extend the velocity linearly through zero at the actual
wall location. The sampled exterior node is at least one grid spacing outside,
so the kernel never reads another node it writes. No pouring data enters this
discretization experiment, and it is not enabled in the production simulator.
"""
import warp as wp
from warpmpm.kernels.warp_utils import Dirichlet_collider, MPMStateStruct, MPMModelStruct


@wp.kernel
def ghost_plane(time: float, dt: float, state: MPMStateStruct, model: MPMModelStruct,
                param: Dirichlet_collider, lo: wp.vec3i):
    i,j,k=wp.tid()
    i=i+lo[0];j=j+lo[1];k=k+lo[2]
    if state.grid_m[i,j,k] <= 0.0:
        return
    distance=float(k)*model.dx-param.point[2]
    if distance <= 0.0 and time >= param.start_time and time < param.end_time:
        outside=int(wp.ceil(param.point[2]/model.dx))+1
        exterior_distance=float(outside)*model.dx-param.point[2]
        velocity=wp.vec3(0.0)
        if state.grid_m[i,j,outside] > 0.0:
            velocity=state.grid_v_out[i,j,outside]*(distance/exterior_distance)
        state.grid_v_out[i,j,k]=velocity


def add_ghost_plane(solver,bottom):
    solver.add_plane((0,0,bottom),(0,0,1),'sticky')
    solver._sim.grid_postprocess[-1]=ghost_plane
