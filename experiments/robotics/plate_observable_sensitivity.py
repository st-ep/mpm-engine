"""Local parameter sensitivity from the frozen observation-only weak system.

This is a conditioning diagnostic, not a statistical confidence interval.
It reads inferred parameters and observations, never true material values.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments.robotics.plate_observable_field import reconstruct,unit_stress,volume_quadrature
from experiments.robotics.plate_observable_divfree import weak
from experiments.elastic.strip_camera_observe import save


def sensitivity(root,k):
    p=json.loads((root/f'inputs_{k}/known.json').read_text());d=np.load(root/f'fit_{k}/tracks.npz')
    f=json.loads((root/f'divfree_{k}/identification.json').read_text());selected=json.loads((root/'field_selection.json').read_text())['selected'];s=selected['spacing']
    p['cross_section_degree']=selected.get('cross_section_degree',3)
    order=f.get('quadrature_order');q,w=(None,None) if order is None else volume_quadrature(p,s,order)
    X,V,F,q,_=reconstruct(d,p,s,quadrature=q,smoothing=selected.get('smoothing',0.))
    if w is None:w=np.full(len(q),1/len(q))
    sensor=np.genfromtxt(root/f'inputs_{k}/force.csv',names=True,delimiter=',')
    if f.get('constant_contact_virtual_displacement'):
        from experiments.robotics.plate_observable_interior import weak as interior_weak
        W,b,g=interior_weak(X,V,d['time'],p,sensor,w)
    else:W,b,g=weak(X,V,d['time'],p,sensor,(1,2,3),w)
    scale=np.ones_like(b);centers=np.arange(.20,p['fit_end']-.159,.06)+.08
    for lo,hi in f.get('balanced_identification_intervals') or []:
        rows=np.repeat((centers>=lo)&(centers<hi),g.shape[1]);scale[rows]=1/np.linalg.norm(b[rows])
    def prediction(logE,logY):
        E,Y=np.exp([logE,logY]);tau=unit_stress(F,Y/E,p['nu'],True)
        return (W@np.einsum('tnij,tmnij,n->tm',tau,g,w)).ravel()*np.prod(p['size'])*E*scale
    logs=np.log([f['E_pa'],f['yield_pa']]);columns=[];h=.001
    for j in range(2):
        step=np.zeros(2);step[j]=h
        columns.append((prediction(*(logs+step))-prediction(*(logs-step)))/(2*h))
    J=np.stack(columns,axis=1)/np.linalg.norm(b*scale);sv=np.linalg.svd(J,compute_uv=False)
    result=dict(scope='Local derivative with respect to log parameters, normalized by measured weak-load norm; no assumed noise distribution',
                balanced_identification_intervals=f.get('balanced_identification_intervals'),
                singular_values=sv.tolist(),condition_number=float(sv[0]/sv[-1]),
                stiffness_sensitivity_norm=float(np.linalg.norm(J[:,0])),yield_sensitivity_norm=float(np.linalg.norm(J[:,1])))
    save(root/f'sensitivity_{k}.json',result);np.savez(root/f'sensitivity_windows_{k}.npz',time=np.arange(.20,p['fit_end']-.159,.06)+.08,J=J.reshape(-1,3,2))
    print(k,result,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--material',choices=['A','B'],required=True)
    a=ap.parse_args();sensitivity(a.out,a.material)
