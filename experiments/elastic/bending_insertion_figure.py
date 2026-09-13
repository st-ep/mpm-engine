"""Compact paper figure from frozen textured bending and audited rod insertion.

No simulation or fitting is run. Panel (a) shows A at the prescribed end of
loading; all four results in (b) use the final recorded time and the same view.
"""
from pathlib import Path
import argparse
import shutil
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from experiments.elastic.block_drop_study import sha,save
from experiments.elastic.strip_texture_observe import meshes,texture
from experiments.elastic.strip_bending_study import pusher_position
from experiments.elastic.rod_insertion_report import surface,fixture,read,COLORS
from experiments.elastic.rod_insertion_franka import Panda,OFFSET

INK='#26343d';MUTED='#637581';BG='#f3f5f6'


def projected(pl,point):
    import vtk
    c=vtk.vtkCoordinate();c.SetCoordinateSystemToWorld();c.SetValue(*point)
    return np.asarray(c.GetComputedDoubleDisplayValue(pl.renderer))/pl.window_size


def bend(root,out):
    import pyvista as pv
    p=read(root/'protocol.json');x=np.load(root/'truth_A/x.npy',mmap_mode='r');t=np.load(root/'truth_A/time.npy')
    i=int(np.argmin(abs(t-p['bend_end'])));axes=[np.unique(x[0,:,k]) for k in range(3)]
    interp=RegularGridInterpolator(axes,x[i].reshape(tuple(map(len,axes))+(3,)),bounds_error=False,fill_value=None)
    (front,ref),others=meshes(p)
    pl=pv.Plotter(off_screen=True,window_size=(1000,1184));pl.set_background(BG);pl.enable_anti_aliasing('ssaa')
    front.points=interp(ref);pl.add_mesh(front,texture=pv.numpy_to_texture(texture()),lighting=False)
    for mesh,q in others:mesh.points=interp(q);pl.add_mesh(mesh,color='#929fa5',smooth_shading=True)
    yf=p['center'][1]-p['size'][1]/2;yb=p['center'][1]+p['size'][1]/2
    for b in [(.101,.139,yf-.012,yf,.165,.193),(.101,.139,yb,yb+.012,.165,.193),(.101,.139,yf-.012,yb+.012,.193,.204)]:
        pl.add_mesh(pv.Box(bounds=b),color='#acb7bf',ambient=.25,specular=.2)
    push=pusher_position(t[i],p)
    pl.add_mesh(pv.Cylinder(center=push,direction=(0,1,0),radius=p['pusher_radius'],height=p['pusher_length'],resolution=96),color='#8f9faa',smooth_shading=True,specular=.3)
    # Dashed undeformed silhouette; drawn as a reference, not another specimen.
    zlo=p['center'][2]-p['size'][2]/2;zup=p['grip_lower_z'];xl=p['center'][0]-p['size'][0]/2;xr=xl+p['size'][0]
    pts=np.array([[xl,yf-.0003,zup],[xl,yf-.0003,zlo],[xr,yf-.0003,zlo],[xr,yf-.0003,zup]])
    for a,b in zip(pts[:-1],pts[1:]):
        L=np.linalg.norm(b-a)
        for start in np.arange(0,L,.0025):
            end=min(start+.0014,L);pl.add_mesh(pv.Line(a+(b-a)*start/L,a+(b-a)*end/L),color='#8fa1ac',line_width=3.2,lighting=False)
    focal=np.array([.116,.12,.14]);pl.camera.position=focal+[.055,-.4,.025];pl.camera.focal_point=focal;pl.camera.up=(0,0,1)
    pl.enable_parallel_projection();pl.camera.parallel_scale=.070;pl.reset_camera_clipping_range();pl.render()
    locations={name:projected(pl,pt).tolist() for name,pt in dict(initial=[xr,yf,.130],pusher=push+[.003,-p['pusher_length']/2,0]).items()}
    pl.screenshot(out/'bending.png');pl.close()
    return dict(material='A',frame_index=i,time_s=float(t[i]),state_sha256=sha(root/'truth_A/x.npy'),annotations=locations)


def insertion(root,m,k,out):
    import pyvista as pv
    case=root/f'fine_{m}_plan_{k}';p=read(case/'config.json')['protocol'];r=read(root/'report.json')['executions'][case.name]
    assert r['all_frames_completed'] and r['inverted_count']==0 and r['material_clear_of_rendered_support']
    x=np.load(case/'x.npy',mmap_mode='r');ref=np.load(case/'reference.npy');fk=np.load(root/'franka'/f'plan_{k}.npz')
    robot=Panda(root/'franka/panda_model_snapshot/panda.xml');robot.forward(fk['q'][-1])
    pl=pv.Plotter(off_screen=True,window_size=(1100,515));pl.set_background(BG);pl.enable_anti_aliasing('ssaa');pl.enable_lightkit()
    mesh,idx,w=surface(ref,p);mesh.points=np.einsum('nk,nkj->nj',w,x[-1,idx]);mesh.compute_normals(inplace=True)
    pl.add_mesh(mesh,color=COLORS[m],smooth_shading=True,ambient=.28,diffuse=.72,specular=.3,specular_power=30)
    pl.add_mesh(fixture(p),color='#abb9c2',opacity=.32)
    th=np.linspace(0,2*np.pi,129)
    for side in [-1,1]:
        q=np.c_[np.full(len(th),p['wall_x']+side*p['wall_thickness']/2),p['center_y']+p['opening_radius']*np.cos(th),p['opening_z']+p['opening_radius']*np.sin(th)]
        pl.add_mesh(pv.lines_from_points(q),color='#425e70',line_width=2.5)
    pl.add_mesh(pv.Box(bounds=(p['wall_x']-.015,p['wall_x']+.015,p['center_y']-.042,p['center_y']+.042,-OFFSET[2],p['opening_z']-p['wall_half_height'])),color='#c6cfd3')
    for g,body,color in robot.meshes(False):pl.add_mesh(body,color=color,smooth_shading=True,ambient=.3,specular=.25)
    focal=np.array([.19,.16,.145]);pl.camera.position=focal+[-.18,-.26,.10];pl.camera.focal_point=focal;pl.camera.up=(0,0,1)
    pl.enable_parallel_projection();pl.camera.parallel_scale=.0415;pl.reset_camera_clipping_range();pl.render();pl.screenshot(out/f'{m}_plan_{k}.png');pl.close()
    return dict(frame_index=len(x)-1,time_s=float(fk['time'][-1]),state_sha256=sha(case/'x.npy'),motion_sha256=sha(root/'franka'/f'plan_{k}.npz'),metrics=r)


def make(bending,rods,out):
    out.mkdir(parents=True,exist_ok=True);renders=out/'renders';renders.mkdir(exist_ok=True)
    meta=dict(bending_source=str(bending.resolve()),insertion_source=str(rods.resolve()),bending=bend(bending,renders),results={})
    for m in 'AB':
        for k in 'AB':meta['results'][m+k]=insertion(rods,m,k,renders)
    W,H=7.16,2.88
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.4,'pdf.fonttype':42})
    fig=plt.figure(figsize=(W,H),facecolor='white')
    def txt(x,y,s,**kw):return fig.text(x/W,y/H,s,color=INK,fontsize=7.4,**kw)
    def picture(x,y,w,h,file):
        ax=fig.add_axes([x/W,y/H,w/W,h/H]);ax.imshow(plt.imread(file),aspect='auto');ax.axis('off');return ax
    fig.text(.015/W,2.76/H,'(a) Bending probe',fontsize=8.5,weight='bold',color=INK)
    fig.text(2.43/W,2.76/H,'(b) Insertion',fontsize=8.5,weight='bold',color=INK)
    a=picture(.015,.09,2.18,2.58,renders/'bending.png')
    loc=meta['bending']['annotations']
    a.annotate('Initial',xy=loc['initial'],xycoords='axes fraction',xytext=(.83,.48),textcoords='axes fraction',ha='center',fontsize=7.4,color=MUTED,
               arrowprops=dict(arrowstyle='-',color=MUTED,lw=.6,shrinkB=3))
    a.annotate('Pusher',xy=loc['pusher'],xycoords='axes fraction',xytext=(.80,.15),textcoords='axes fraction',ha='center',fontsize=7.4,color=INK,
               arrowprops=dict(arrowstyle='-',color=MUTED,lw=.6,shrinkB=3))
    xs=[2.51,4.88];width=2.24
    for x,label in zip(xs,['Plan using matched ID','Plan using swapped ID']):txt(x+width/2,2.48,label,ha='center')
    for m,y in zip('AB',[1.30,.13]):
        fig.text(2.40/W,(y+.525)/H,m,fontsize=8,weight='bold',color=COLORS[m],ha='center',va='center')
        for x,k in zip(xs,[m,'B' if m=='A' else 'A']):
            picture(x,y,width,1.05,renders/f'{m}_plan_{k}.png')
            success=meta['results'][m+k]['metrics']['success']
            fig.text((x+width/2)/W,(y-.08)/H,'Inserted' if success else 'Wall contact',fontsize=7.4,
                     color='#347866' if success else '#a85439',ha='center')
    for suffix in ['pdf','png']:fig.savefig(out/f'bending_insertion.{suffix}',dpi=350,pad_inches=0)
    plt.close(fig);save(out/'figure_provenance.json',meta)
    shutil.copy2(Path(__file__),out/Path(__file__).name)
    save(out/'figure_hashes.json',{p.name:sha(p) for p in [out/'bending_insertion.pdf',out/'bending_insertion.png',out/Path(__file__).name]})

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--bending',type=Path,default=Path('out/strip_texture_20260912'));ap.add_argument('--rods',type=Path,default=Path('out/rod_insertion_franka_20260912'));ap.add_argument('--out',type=Path,default=Path('out/bending_insertion_figure_20260912'))
    a=ap.parse_args();make(a.bending,a.rods,a.out)
