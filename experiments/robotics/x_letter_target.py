"""Rounded capital-X targets, distinct from the frozen notched-square study."""
from functools import lru_cache
import numpy as np
import pyvista as pv

TARGETS = {
    "balanced": dict(width=78., height=86., end_width=20., cx=12., cy=8., inner_radius=14., outer_radius=2.),
    "slimmer": dict(width=78., height=86., end_width=18., cx=10., cy=8., inner_radius=14., outer_radius=2.),
}


def outline(spec):
    a, b, t = spec["width"]/2, spec["height"]/2, spec["end_width"]
    cx, cy = spec["cx"], spec["cy"]
    p = np.array([[-a, -b], [-a+t, -b], [0, -cy], [a-t, -b], [a, -b],
                  [cx, 0], [a, b], [a-t, b], [0, cy], [-a+t, b], [-a, b], [-cx, 0]], float)
    parts, distances = [], []
    for i, v in enumerate(p):
        u, w = p[(i-1) % len(p)]-v, p[(i+1) % len(p)]-v
        u /= np.linalg.norm(u); w /= np.linalg.norm(w)
        turn = np.linalg.det(np.stack([-u, w]))
        r = spec["inner_radius"] if turn < 0 else spec["outer_radius"]
        theta = np.arccos(np.clip(u@w, -1, 1)); d = r/np.tan(theta/2)
        c = v+(u+w)/np.linalg.norm(u+w)*r/np.sin(theta/2)
        start, end = v+u*d, v+w*d
        a0, a1 = np.arctan2(*(start-c)[::-1]), np.arctan2(*(end-c)[::-1])
        delta = (a1-a0) % (2*np.pi) if turn > 0 else -((a0-a1) % (2*np.pi))
        angles = np.linspace(a0, a0+delta, 65)
        parts.append(c+r*np.column_stack([np.cos(angles), np.sin(angles)]))
        distances.append(d)
    for i in range(len(p)):
        assert distances[i]+distances[(i+1) % len(p)] < np.linalg.norm(p[(i+1) % len(p)]-p[i])
    xy = np.concatenate(parts)/1000
    area = .5*np.sum(xy[:, 0]*np.roll(xy[:, 1], -1)-xy[:, 1]*np.roll(xy[:, 0], -1))
    return xy, float(area)


@lru_cache
def mesh(name, subdivisions=3):
    xy, area = outline(TARGETS[name]); xy += .08
    points = np.column_stack([xy, np.full(len(xy), .010)])
    face = pv.PolyData(points, np.r_[len(points), np.arange(len(points))])
    result = face.extrude((0, 0, .060*.060*.025/area), capping=True).triangulate().clean()
    result = result.compute_normals(auto_orient_normals=True, consistent_normals=True)
    result = result.subdivide(subdivisions, subfilter="linear")
    assert result.n_open_edges == 0
    np.testing.assert_allclose(result.volume, .00009, rtol=2e-6)
    return result
