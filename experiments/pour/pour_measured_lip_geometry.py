"""Keep the measured rim while retaining outward collision padding below it.

No new fitted dimensions: the transition uses the existing measured spout depth
and the geometry model's existing rim-blend depth. This is an experimental SDF
variant; preservation of the solid does not alone guarantee MPM leak resistance.
"""
from contextlib import contextmanager

import numpy as np

from warpmpm.geometry import measuring_cup as cup

ORIGINAL_SOLID = cup.solid_sdf_local


def measured_lip_solid(points, spec, extra_wall=0., extra_base=0.):
    z = np.asarray(points)[..., 2]
    start, end = spec.spout_z0, spec.rim_z - spec.rim_blend
    if end <= start:
        raise ValueError("Spout region must extend below the rim blend")
    u = np.clip((z - start) / (end - start), 0., 1.)
    padding = np.asarray(extra_wall) * (1. - u*u*(3. - 2.*u))
    return ORIGINAL_SOLID(points, spec, padding, extra_base)


@contextmanager
def measured_lip_field():
    """Opt-in process-local field override for existing SDF/audit helpers."""
    old = cup.solid_sdf_local
    cup.solid_sdf_local = measured_lip_solid
    try:
        yield
    finally:
        cup.solid_sdf_local = old
