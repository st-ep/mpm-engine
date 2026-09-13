import numpy as np
import pytest
from experiments.robotics.plate_observable_probe import PROTOCOL,opening,plate_sdf
from experiments.robotics.plate_observable_field import basis,unit_stress,volume_quadrature,height_edges
from experiments.robotics.plate_observable_divfree import virtual


@pytest.mark.parametrize('degree',[3,5,7])
def test_square_field_gradient_and_symmetry(degree):
    p=dict(PROTOCOL,cross_section_degree=degree);q=np.array([[.108,.092,.063],[.092,.108,.063]])
    M,G=basis(q,p,.00625);c=np.random.default_rng(3).normal(size=M.shape[-1])*.0001
    for j in range(3):
        dq=np.zeros(3);dq[j]=1e-7
        numerical=(basis(q+dq,p,.00625)[0]-basis(q-dq,p,.00625)[0])/(2e-7)
        np.testing.assert_allclose(numerical@c,np.einsum('ndj,j->nd',G[:,:,j],c),atol=1e-8)
    u=M@c;np.testing.assert_allclose(u[0,[1,0,2]],u[1],atol=1e-15)


def test_virtual_fields_pressure_and_boundary():
    p=PROTOCOL;X=np.array([[[.104,.096,.05],[.105,.103,.066]]]);w,g=virtual(X,p)
    np.testing.assert_allclose(np.trace(g,axis1=-2,axis2=-1),0,atol=1e-12)
    np.testing.assert_allclose(w[0,:,0,2],0)
    for j in range(3):
        dq=np.zeros(3);dq[j]=1e-7
        numerical=(virtual(X+dq,p)[0]-virtual(X-dq,p)[0])/(2e-7)
        np.testing.assert_allclose(numerical,g[...,j],rtol=1e-8,atol=1e-7)
    pressure=np.array([[123.,567.]])
    np.testing.assert_allclose(np.einsum('tn,ij,tmnij->tm',pressure,np.eye(3),g),0,atol=1e-8)


def test_return_map_elastic_limit_rotation_and_yield():
    a=np.linspace(0,.25,30);F=np.array([np.diag(np.exp([x/2,x/2,-x])) for x in a])[:,None]
    s=unit_stress(F,.01,.3,full=True);dev=s-np.trace(s,axis1=-2,axis2=-1)[...,None,None]*np.eye(3)/3
    assert np.linalg.norm(dev[-1,0])<=.01+1e-12
    np.testing.assert_allclose(np.trace(s,axis1=-2,axis2=-1),0,atol=1e-12)
    elastic=unit_stress(F,10.,.3,full=True)
    np.testing.assert_allclose(elastic[-1,0],np.diag([a[-1]/2,a[-1]/2,-a[-1]])/1.3,atol=1e-12)
    theta=.4;R=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
    rotated=unit_stress(R@F@R.T,.01,.3,full=True)
    np.testing.assert_allclose(rotated,R@s@R.T,atol=1e-12)


def test_common_plate_motion():
    for t,h in PROTOCOL['knots']:np.testing.assert_allclose(opening(t,PROTOCOL),h,atol=1e-12)
    for t,_ in PROTOCOL['knots'][1:-1]:
        assert abs(opening(t+1e-7,PROTOCOL)-opening(t-1e-7,PROTOCOL))<1e-10


def test_plate_sdf_contract_and_surface():
    field=plate_sdf(PROTOCOL)
    assert field.values.shape==(field.res,)*3
    center=np.rint(-field.origin/field.cell).astype(int)
    np.testing.assert_allclose(field.values[tuple(center)],-.003,atol=1e-8)
    top=center.copy();top[2]+=round(.003/field.cell)
    np.testing.assert_allclose(field.values[tuple(top)],0,atol=1e-8)
    np.testing.assert_allclose(field.grads[tuple(top)],[0,0,1],atol=1e-6)


def test_volume_quadrature_integrates_reference_moments():
    q,w=volume_quadrature(PROTOCOL,.00625,6)
    np.testing.assert_allclose(sum(w),1,atol=1e-14)
    np.testing.assert_allclose(w@q,PROTOCOL['center'],atol=1e-14)
    np.testing.assert_allclose(w@((q-PROTOCOL['center'])**2),np.array(PROTOCOL['size'])**2/12,atol=1e-14)


def test_height_spans_do_not_leave_a_short_unobserved_tail():
    edges=height_edges(PROTOCOL,.004)
    assert len(edges)==7
    np.testing.assert_allclose(np.diff(edges),PROTOCOL['size'][2]/6,atol=1e-14)


def test_camera_rectification_is_invariant_to_world_translation():
    import copy
    from experiments.robotics.plate_observable_camera import CAMERA
    from experiments.robotics.plate_observable_identify import homography
    from experiments.elastic.strip_camera_observe import calibration
    p=copy.deepcopy(PROTOCOL);p['surface_y']=p['center'][1]-p['size'][1]/2
    shift=np.array([-.04,-.04,-.02]);b=copy.deepcopy(p);b['center']=(np.array(p['center'])+shift).tolist()
    b['floor']+=shift[2];b['surface_y']+=shift[1]
    camera=copy.deepcopy(CAMERA);camera['centers']=(np.array(camera['centers'])+shift).tolist();camera['target']=(np.array(camera['target'])+shift).tolist()
    for a,c in zip(calibration(CAMERA),calibration(camera),strict=True):
        np.testing.assert_allclose(homography(a,p),homography(c,b),atol=1e-10)


def test_interior_virtual_fields_ignore_contact_location_and_pressure():
    from experiments.robotics.plate_observable_interior import virtual as interior
    p=PROTOCOL;X=np.array([[[.105,.094,.049],[.105,.094,.066],[.106,.093,.059]]]);w,g=interior(X,p)
    np.testing.assert_allclose(w[0,:,0],0,atol=1e-12)
    np.testing.assert_allclose(w[0,:,1],np.tile([0,0,1],(3,1)),atol=1e-12)
    np.testing.assert_allclose(g[0,:,:2],0,atol=1e-12)
    np.testing.assert_allclose(np.trace(g,axis1=-2,axis2=-1),0,atol=1e-12)
    for j in range(3):
        dq=np.zeros(3);dq[j]=1e-8
        derivative=(interior(X+dq,p)[0]-interior(X-dq,p)[0])/(2e-8)
        np.testing.assert_allclose(derivative,g[...,j],rtol=1e-7,atol=1e-6)


def test_height_smoothing_preserves_affine_height_fields():
    from experiments.robotics.plate_observable_field import height_penalty
    p=dict(PROTOCOL,surface_y=.085);spacing=.025/8
    q=np.column_stack([np.full(100,.107),np.full(100,.085),np.linspace(.05,.075,100)])
    M,_=basis(q,p,spacing);R=height_penalty(p,spacing)
    # An affine expansion/translation in height lies in the zero-penalty space.
    target=np.column_stack([.001*(q[:,2]-.05)/.025,np.zeros(100),np.zeros(100)])
    n=M.shape[-1]//5
    c=np.zeros(M.shape[-1]);c[:n]=np.linalg.lstsq(M[:,0,:n],target[:,0],rcond=None)[0]
    np.testing.assert_allclose(R@c,0,atol=1e-12)
    # Curvature is actually penalized, rather than acting as a strain constraint.
    c[:n]=np.linalg.lstsq(M[:,0,:n],.001*((q[:,2]-.05)/.025)**2,rcond=None)[0]
    assert np.linalg.norm(R@c)>1e-3
