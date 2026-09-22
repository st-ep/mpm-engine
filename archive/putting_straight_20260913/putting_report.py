"""Render stored particle trajectories and actual Panda poses, without retiming physics."""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
from PIL import Image,ImageDraw
import imageio.v2 as imageio
from experiments.elastic.block_drop_report import font
from experiments.elastic.putting import save,digest
from experiments.elastic.rod_insertion_franka import Panda,OFFSET

COLORS={'A':'#3184af','B':'#c37647'};INK='#233540';MUTED='#667982'

def read(path):return json.loads(Path(path).read_text())

def surface(ref,p):
    """Reference union surface, deformed by local affine particle interpolation.

    This interpolation is only for rendering. Metrics use the rigid ball state.
    """
    import pyvista as pv
    h=.0006;lo=np.array([-.008,-.016,-.078]);hi=np.array([.008,.016,.010])
    axes=[np.arange(a,b+h/2,h) for a,b in zip(lo,hi)]
    q=np.stack(np.meshgrid(*axes,indexing='ij'),-1)
    shaftlo=np.array([-p['shaft_width']/2,-p['shaft_depth']/2,p['shaft_bottom']]);shafthi=np.array([p['shaft_width']/2,p['shaft_depth']/2,p['shaft_top']])
    def box(c,half):
        d=abs(q-c)-half
        return np.linalg.norm(np.maximum(d,0),axis=-1)+np.minimum(d.max(-1),0)
    sdf=np.minimum(box((shaftlo+shafthi)/2,(shafthi-shaftlo)/2),box(np.array(p['head_center']),np.array(p['head_size'])/2))
    pts,faces,_,_=marching_cubes(sdf,0,spacing=(h,h,h));pts+=lo
    d,idx=cKDTree(ref).query(pts,k=32);local=ref[idx]-pts[:,None,:]
    A=np.concatenate([np.ones((*idx.shape,1)),local/d[:,-1,None,None]],axis=2)
    w=np.exp(-3*(d/d[:,-1,None])**2);G=np.einsum('nki,nk,nkj->nij',A,w,A)
    weights=np.einsum('ni,nki,nk->nk',np.linalg.pinv(G)[:,0,:],A,w)
    np.testing.assert_allclose(np.einsum('nk,nkj->nj',weights,ref[idx]),pts,atol=1e-9)
    return pv.PolyData(pts,np.c_[np.full(len(faces),3),faces].ravel()),idx,weights

def render_case(root,m,k,prefix='execution',view='course',preview=False):
    import pyvista as pv
    case=root/f'{prefix}_{m}_plan_{k}';cfg=read(case/'config.json');p=cfg['protocol']
    data=np.load(case/'trajectory.npz');signals=data['signals'];xs=data['x'];ref=data['reference']
    provenance=read(case/'execution_provenance.json');path=Path(provenance['motion'])
    assert digest(path)==provenance['motion_sha256'];fk=np.load(path)
    robot=Panda(root/'franka/panda_model_snapshot/panda.xml')
    media=root/'media';media.mkdir(exist_ok=True);name=f'{prefix}_{m}_{k}_{view}'
    if view=='detail':ids=np.flatnonzero((signals[:,0]>=.10)&(signals[:,0]<=.80));W,H=800,640
    else:ids=np.arange(0,len(signals),round(.04/(signals[1,0]-signals[0,0])));W,H=(1200,780) if view=='overview' else (960,470)
    if preview:ids=np.array([0,int(np.argmin(abs(signals[:,0]-.35))),len(signals)-1])
    frames=np.lib.format.open_memmap(media/f'{name}_frames.npy',mode='w+',dtype='uint8',shape=(len(ids),H,W,3))
    pl=pv.Plotter(off_screen=True,window_size=(W,H));pl.set_background('#f4f5f3');pl.enable_anti_aliasing('ssaa');pl.enable_lightkit()
    # Table and target geometry match the numerical floor and success criterion.
    pl.add_mesh(pv.Box(bounds=p['table_bounds']),color='#466959',ambient=.45,diffuse=.65)
    pl.add_mesh(pv.Box(bounds=(.55,.65,.02,.16,-OFFSET[2],p['table_bounds'][4])),color='#7b8789')
    theta=np.linspace(0,2*np.pi,181);target=np.array(p['target']);r=p['target_radius']
    ring=np.c_[target[0]+r*np.cos(theta),target[1]+r*np.sin(theta),np.full(len(theta),p['floor']+.0004)]
    pl.add_mesh(pv.lines_from_points(ring).tube(radius=.0014),color='#f0e6c5')
    pl.add_mesh(pv.Disc(center=(*target,p['floor']+.0002),inner=0,outer=r,normal=(0,0,1),c_res=120),color='#84967a',opacity=.55)
    # Small starting-location ring remains after the ball leaves.
    initial=np.array(p['ball_initial']);ring0=np.c_[initial[0]+.010*np.cos(theta),initial[1]+.010*np.sin(theta),np.full(len(theta),p['floor']+.0003)]
    pl.add_mesh(pv.lines_from_points(ring0),color='#a0b6a5',line_width=1)
    pl.add_mesh(pv.Plane(center=(.35,.1,-OFFSET[2]-.002),direction=(0,0,1),i_size=2.5,j_size=2.0),color='#e6e9e7')
    club,idx,weights=surface(ref,p);pl.add_mesh(club,color=COLORS[m],smooth_shading=True,ambient=.26,diffuse=.75,specular=.22,specular_power=35)
    sphere=pv.Sphere(radius=p['ball_radius'],theta_resolution=40,phi_resolution=32)
    base=sphere.points.copy();sphere.points=base+initial
    if signals.shape[1]>=21:
        rgb=np.tile(np.array([255,253,243],dtype=np.uint8),(len(base),1));rgb[abs(base[:,0])<.0008]=[67,80,81]
        pl.add_mesh(sphere,scalars=rgb,rgb=True,smooth_shading=True,ambient=.4,specular=.35)
    else:pl.add_mesh(sphere,color='#fffdf3',smooth_shading=True,ambient=.4,specular=.35)
    robot.forward(fk['q'][0]);geoms={};include_arm=view!='detail'
    for g,mesh,color in robot.meshes(include_arm=include_arm):
        geoms[g]=mesh;pl.add_mesh(mesh,color=color,smooth_shading=True,ambient=.32,specular=.2)
    if view=='overview':focal=np.array([.39,.18,.11]);eye=focal+[-.72,-1.2,.83];scale=.63
    elif view=='detail':focal=np.array([.122,.118,.066]);eye=focal+[-.06,-.30,.085];scale=.097
    else:focal=np.array([.57,.14,.035]);eye=focal+[.06,-.85,.43];scale=.29
    pl.camera.position=eye;pl.camera.focal_point=focal;pl.camera.up=(0,0,1);pl.enable_parallel_projection();pl.camera.parallel_scale=scale
    for j,i in enumerate(ids):
        club.points=np.einsum('nk,nkj->nj',weights,xs[i,idx]);club.compute_normals(inplace=True)
        if signals.shape[1]>=21:
            from scipy.spatial.transform import Rotation
            sphere.points=base@Rotation.from_quat(signals[i,17:21]).as_matrix().T+signals[i,1:4]
            sphere.compute_normals(inplace=True)
        else:sphere.points=base+signals[i,1:4]
        ti=signals[i,0];qi=int(round(ti/.01));robot.forward(fk['q'][qi])
        for g,mesh,_ in robot.meshes(include_arm=include_arm):geoms[g].points=mesh.points
        pl.reset_camera_clipping_range();pl.render();frames[j]=pl.screenshot(return_img=True)[:,:,:3]
        if preview:Image.fromarray(frames[j]).save(media/f'{name}_preview_{j}.png')
        if j%50==0:print(name,j,flush=True)
    frames.flush();pl.close();np.save(media/f'{name}_time.npy',signals[ids,0])
    return name

def compose(root,prefix='execution',view='course'):
    media=root/'media';names={(m,k):f'{prefix}_{m}_{k}_{view}' for m in 'AB' for k in 'AB'}
    frames={key:np.load(media/f'{name}_frames.npy',mmap_mode='r') for key,name in names.items()}
    t=np.load(media/f'{names["A","A"]}_time.npy');n,h,w,_=frames['A','A'].shape
    slow=4 if view=='detail' else 1;fps=1/((t[1]-t[0])*slow)
    p=read(root/f'{prefix}_A_plan_A/config.json')['protocol']
    grid_label=f" · {1000*p['domain']/p['grid']:.1f} mm grid" if view=='course' else ''
    def canvas(i):
        im=Image.new('RGB',(2*w+80,2*h+180),'white');d=ImageDraw.Draw(im)
        for col,title in enumerate(['Plan using matched ID','Plan using swapped ID']):
            d.text((65+col*w,18),title,font=font(28,True),fill=INK)
        for row,m in enumerate('AB'):
            y=65+row*(h+40)
            d.text((15,y+h//2-15),m,font=font(30,True),fill=COLORS[m])
            for col,k in enumerate([m,'B' if m=='A' else 'A']):
                x=55+col*(w+10);im.paste(Image.fromarray(frames[m,k][i]),(x,y))
                if view=='course' and i>=n-2:
                    r=read(root/f'{prefix}_{m}_plan_{k}/result.json')
                    d.text((x+15,y+h+4),f"Final error {r['error_mm']:.1f} mm",font=font(23),fill=INK)
        d.text((55,2*h+151),f"{'4× slow motion' if slow==4 else 'Real time'}{grid_label} · {t[i]:.2f} s",font=font(20),fill=MUTED)
        return np.asarray(im)
    name='putting_comparison' if view=='course' else 'putting_contact_closeup'
    with imageio.get_writer(media/f'{name}.mp4',fps=fps,codec='libx264',quality=8,macro_block_size=2,pixelformat='yuv420p') as writer:
        for i in range(n):writer.append_data(canvas(i))
        for _ in range(round(fps)):writer.append_data(canvas(n-1))
    Image.fromarray(canvas(n-1)).save(media/f'{name}.png')

def overview(root,prefix='execution'):
    media=root/'media';name=f'{prefix}_A_A_overview';frames=np.load(media/f'{name}_frames.npy',mmap_mode='r');t=np.load(media/f'{name}_time.npy')
    with imageio.get_writer(media/'franka_putting.mp4',fps=25,codec='libx264',quality=8,macro_block_size=2,pixelformat='yuv420p') as writer:
        for i,frame in enumerate(frames):
            im=Image.fromarray(frame);d=ImageDraw.Draw(im);d.text((25,22),'Elastic club · Material A · Plan using A',font=font(28,True),fill=INK)
            d.text((25,745),f'Kinematic Franka · Open-loop stroke · {t[i]:.2f} s',font=font(20),fill=MUTED);writer.append_data(np.asarray(im))
        for _ in range(25):writer.append_data(np.asarray(im))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['render','compose','overview']);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--material',default='A');ap.add_argument('--plan',default='A');ap.add_argument('--prefix',default='execution')
    ap.add_argument('--view',default='course',choices=['course','detail','overview']);ap.add_argument('--preview',action='store_true')
    a=ap.parse_args()
    if a.command=='render':render_case(a.out,a.material,a.plan,a.prefix,a.view,a.preview)
    elif a.command=='compose':compose(a.out,a.prefix,a.view)
    else:overview(a.out,a.prefix)
