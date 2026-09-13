import numpy as np
from experiments.robotics.plate_observable_probe import PROTOCOL
from experiments.robotics.plate_observable_field import basis
from experiments.robotics.plate_observable_interior_modes import modes


def test_interior_modes_preserve_observed_sides_and_have_correct_gradients():
    p=PROTOCOL;s=.00625
    q=np.array([[.105,.096,.062],[.085,.106,.064],[.115,.106,.064],
                [.106,.085,.064],[.106,.115,.064],[.105,.097,.05],[.105,.097,.075]])
    n=basis(q,p,s)[0].shape[-1];c=np.random.default_rng(91).normal(size=(3,n))*.0001
    result=modes(q,p,s,c,c*.2)
    for dx,dv,G in result:
        np.testing.assert_allclose(dx[:,1:5],0,atol=1e-17)
        np.testing.assert_allclose(dv,dx*.2,atol=1e-17)
    np.testing.assert_allclose(result[1][0][:,5:,2],0,atol=1e-17)
    for axis in range(3):
        dq=np.zeros(3);dq[axis]=1e-7
        plus=modes(q+dq,p,s,c,c*.2);minus=modes(q-dq,p,s,c,c*.2)
        for j in range(2):np.testing.assert_allclose((plus[j][0]-minus[j][0])/(2e-7),result[j][2][...,axis],atol=1e-9,rtol=1e-7)


def test_full_balance_matches_static_compression_with_gravity():
    from experiments.robotics.plate_observable_interior_modes import balance
    p=dict(PROTOCOL,force_tick=.002)
    u,w=np.polynomial.legendre.leggauss(4);hmin=.016
    edges=np.unique(np.r_[0,np.array([.1,.2,.3,.7,.8,.9])*hmin,p['size'][2]])
    z=np.concatenate([(a+b)/2+(b-a)*u/2 for a,b in zip(edges[:-1],edges[1:],strict=True)])
    wz=np.concatenate([w*(b-a)/(2*p['size'][2]) for a,b in zip(edges[:-1],edges[1:],strict=True)])
    q=np.column_stack([np.full(len(z),.1),np.full(len(z),.1),z+p['floor']])
    t=np.arange(141)*.01;X=np.broadcast_to(q,(len(t),*q.shape));V=np.zeros_like(X)
    sensor=np.zeros(700,dtype=[('time',float),('Fz',float)]);sensor['time']=(np.arange(700)+.5)*.002;sensor['Fz']=2.
    W,b,g=balance(X,V,t,p,sensor,wz)
    stress=np.zeros((len(t),len(q),3,3));area=p['size'][0]*p['size'][1]
    stress[...,2,2]=-2/area-p['density']*9.81*(p['size'][2]-z)
    lhs=(W@np.einsum('tnij,tmnij,n->tm',stress,g,wz)).ravel()*np.prod(p['size'])
    np.testing.assert_allclose(lhs,b,atol=1e-12,rtol=1e-11)
