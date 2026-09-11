"""Smooth geometric X proposals inspired by, but not copied from, the A pilot.

The outline has four identical reflected quadrants. Side and end-notch
curvatures differ because the last pinch acts along x. Heights conserve 90 mL.
No particle positions enter this geometry generator.
"""
from __future__ import annotations

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

PROPOSALS = {
    "gentle": dict(width=66., height=82., waist=30., notch_y=23.,
                   shoulder_y=25., side_radius=14., notch_radius=8.,
                   inner_tip_x=18., corner_radius=4., cap_control=6., side_control=12.),
    "rounded": dict(width=64., height=80., waist=30., notch_y=23.,
                    shoulder_y=22., side_radius=14., notch_radius=6.,
                    inner_tip_x=17., corner_radius=3., cap_control=6., side_control=12.),
    "slimmer": dict(width=64., height=80., waist=28., notch_y=23.,
                    shoulder_y=22., side_radius=14., notch_radius=6.,
                    inner_tip_x=19., corner_radius=3., cap_control=6., side_control=12.),
}

# Selected after the explicitly post-hoc geometric comparison on 2026-09-09.
# Subsequent planning must freeze this geometry before optimizing controls.
SELECTED = "slimmer"


def cubic(points, samples=65):
    p=np.asarray(points,float);t=np.linspace(0,1,samples)[:,None]
    return (1-t)**3*p[0]+3*(1-t)**2*t*p[1]+3*(1-t)*t*t*p[2]+t**3*p[3]


def outline(spec, samples=65):
    """CCW XY boundary in metres; spec dimensions and curvature radii are mm."""
    a,b,w=spec["width"]/2,spec["height"]/2,spec["waist"]/2
    s,n,r=spec["shoulder_y"],spec["notch_y"],spec["corner_radius"]
    tip=spec["inner_tip_x"]
    assert 0<w<a and 0<n<b and 0<s<b-r and tip<a-r
    # At the waist, curvature radius is 3*c_side**2/(2*(a-w)).
    c_side=np.sqrt(2*spec["side_radius"]*(a-w)/3)
    flank=cubic([[w,0],[w,c_side],[a,s-spec["side_control"]],[a,s]],samples)
    upright=np.linspace([a,s],[a,b-r],17)
    theta=np.linspace(0,np.pi/2,samples)
    corner=np.array([a-r,b-r])+r*np.column_stack([np.cos(theta),np.sin(theta)])
    cap=np.linspace([a-r,b],[tip,b],17)
    # The end notch has the specified radius at its lowest point.
    c_notch=np.sqrt(2*spec["notch_radius"]*(b-n)/3)
    notch=cubic([[tip,b],[tip-spec["cap_control"],b],[c_notch,n],[0,n]],samples)
    q=np.concatenate([part[:-1] for part in [flank,upright,corner,cap]]+[notch])
    xy=np.concatenate([q[:-1],(q[::-1]*[-1,1])[:-1],-q[:-1],(q[::-1]*[1,-1])[:-1]])/1000
    area=.5*np.sum(xy[:,0]*np.roll(xy[:,1],-1)-xy[:,1]*np.roll(xy[:,0],-1))
    assert area>0 and np.all(np.isfinite(xy))
    for reflection in [[-1,1],[1,-1],[-1,-1]]:
        assert cKDTree(xy).query(xy*reflection)[0].max()<1e-12
    return xy,float(area)


def mesh(spec, subdivisions=3):
    xy,area=outline(spec)
    points=np.column_stack([xy+.08,np.full(len(xy),.01)])
    shape=pv.PolyData(points,np.r_[len(points),np.arange(len(points))])
    shape=shape.extrude((0,0,.00009/area),capping=True).triangulate().clean()
    shape=shape.compute_normals(auto_orient_normals=True,consistent_normals=True)
    shape=shape.subdivide(subdivisions,subfilter="linear")
    assert shape.n_open_edges==0
    np.testing.assert_allclose(shape.volume,.00009,rtol=2e-6)
    return shape
