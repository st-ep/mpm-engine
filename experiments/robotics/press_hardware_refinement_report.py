"""Compare exploratory hardware candidates without changing baseline metrics."""
from pathlib import Path
import argparse
import numpy as np
from experiments.robotics.press_hardware_observe import read,save

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'out/press_hardware_validation_20260913'


def metrics(folder,episode,profiles):
    sim=np.load(folder/'trajectory.npz');f=sim['force_log'];shape=sim['shape_log'];t=shape[:,0]
    # Short historical pilots do not cover the recovery assessment window.
    if not len(t) or not len(f) or t[-1]<13.35 or f[-1,0]<13.35:
        return None
    old=np.load(OLD/episode/'motion.npz');h=np.interp(t,f[:,0],f[:,3]);observed_h=np.interp(t,old['time'],old['raw_height'])
    raw=old['raw_shape_samples'];width=2*np.interp(t,raw[:,0],raw[:,2]);gerr=(h-observed_h)*1000;werr=(shape[:,1]-width)*1000
    clean=np.load(profiles/episode/'motion.npz');new_width=np.interp(t,clean['time'],clean['observed_max_width']);new_werr=(shape[:,1]-new_width)*1000
    load=(t>=.4)&(t<=10.4);unload=(t>=11)&(t<=13.4);hold=(t>=3)&(t<=10.4)
    force_hold=(f[:,0]>=3)&(f[:,0]<=10.4)
    result=dict(episode=episode,loading_gap_rmse_mm=float(np.sqrt(np.mean(gerr[load]**2))),
        loading_width_rmse_mm=float(np.sqrt(np.mean(werr[load]**2))),unloading_gap_rmse_mm=float(np.sqrt(np.mean(gerr[unload]**2))),
        cleaned_width_rmse_mm=float(np.sqrt(np.mean(new_werr[load]**2))),
        hold_gap_rmse_mm=float(np.sqrt(np.mean(gerr[hold]**2))),hold_width_rmse_mm=float(np.sqrt(np.mean(werr[hold]**2))),
        final_gap_error_mm=float(np.interp(10,t,gerr)),
        force_hold_relative_l2=float(np.linalg.norm((f[:,2]-f[:,1])[force_hold])/np.linalg.norm(f[force_hold,1])))
    result['passes_geometry_screen']=bool(result['loading_gap_rmse_mm']<=2 and result['loading_width_rmse_mm']<=3 and result['unloading_gap_rmse_mm']<=2)
    result['passes_cleaned_width_screen']=bool(result['loading_gap_rmse_mm']<=2 and result['cleaned_width_rmse_mm']<=3 and result['unloading_gap_rmse_mm']<=2)
    if shape.shape[1]>=8:
        result['projected_x_width_rmse_mm']=float(np.sqrt(np.mean(((shape[load,4]-width[load])*1000)**2)))
        result['projected_y_width_rmse_mm']=float(np.sqrt(np.mean(((shape[load,5]-width[load])*1000)**2)))
        extent=shape[np.argmin(abs(t-10)),4:8]*1000
        result['four_azimuth_extents_at_10s_mm']=extent.tolist()
    result['parameters']=read(folder/'completion.json')['parameters'];result['folder']=str(folder)
    return result


def evaluate(scratch,out):
    out.mkdir(exist_ok=True,parents=True);result={};materials=['play_doh','butter_slime','plasticine']
    for name,source in [('baseline',OLD)]+[(p.name,p) for p in scratch.iterdir() if p.is_dir() and (p/'protocol.json').exists()]:
        records=[]
        for i,m in enumerate(materials):
            folders=list((source/m).glob('*/completion.json')) if (source/m).exists() else []
            for j in range(4):
                ep=f'ep{4*i+j:04}';folders.extend((source/'cases'/ep/m).glob('*/completion.json'))
            for completion in folders:
                folder=completion.parent;c=read(completion);ep=c['episode']
                if not c.get('complete',False):continue
                values=metrics(folder,ep,scratch/'clean_moving')
                if values is None:continue
                record=dict(material=m,role='excluded from parameter fitting' if int(ep[2:])%4==2 else 'training-load consistency',**values)
                if folder.name=='validation':records.append(record)
                else:result.setdefault(name+'/'+folder.name,[]).append(record)
        if records:result[name]=records
    save(out/'candidate_predictions.json',dict(scope='Exploratory comparisons; original gap/width metrics preserved, cleaned-profile width reported separately; no pristine confirmatory split',candidates=result))
    for name,records in result.items():
        print(name,flush=True)
        for r in records:
            print(r['material'],r['episode'],[round(r[k],2) for k in ['loading_gap_rmse_mm','loading_width_rmse_mm','unloading_gap_rmse_mm','cleaned_width_rmse_mm']],r['passes_geometry_screen'],flush=True)


def plots(scratch,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels={'play_doh':'Play-Doh','butter_slime':'Butter slime','plasticine':'Plasticine'}
    fig,axes=plt.subplots(3,2,figsize=(11,10),constrained_layout=True)
    for i,(m,label) in enumerate(labels.items()):
        ep=f'ep{4*i+2:04}';d=np.load(OLD/ep/'motion.npz');r=d['raw_shape_samples']
        axes[i,0].plot(d['time'],d['raw_height']*1000,color='black',label='Recorded motion + assumed initial height')
        axes[i,1].plot(r[:,0],r[:,2]*2000,color='black',label='Original silhouette width estimate')
        for source,name,color,style in [(OLD,'Previous model (grid 80)','0.6','--'),(scratch/'crossload','Refined candidate (grid 80)','#2768a4','-')]:
            sim=np.load(source/m/'validation/trajectory.npz');f=sim['force_log'];s=sim['shape_log']
            axes[i,0].plot(f[:,0],f[:,3]*1000,style,color=color,label=name)
            axes[i,1].plot(s[:,0],s[:,1]*1000,style,color=color,label=name)
        fine=scratch/'crossload'/m/'check_grid96/trajectory.npz'
        if fine.exists():
            sim=np.load(fine);f=sim['force_log'];s=sim['shape_log']
            axes[i,0].plot(f[:,0],f[:,3]*1000,':',color='#d65e00',label='Same fit, grid 96')
            axes[i,1].plot(s[:,0],s[:,1]*1000,':',color='#d65e00',label='Same fit, grid 96')
        for j in range(2):
            ax=axes[i,j];ax.set(title=label+(': plate gap' if j==0 else ': width'),ylabel='mm',xlabel='Time from baseline (s)',xlim=(0,13.4));ax.grid(alpha=.2);ax.axvspan(10.8,13.4,color='gray',alpha=.08);ax.legend(fontsize=7)
    fig.suptitle('Exploratory 21 N comparisons | force is prescribed; geometry and recovery are predictions')
    fig.savefig(out/'comparison_curves.png',dpi=150);plt.close(fig)
    records=read(out/'candidate_predictions.json')['candidates']['crossload']
    records=sorted(records,key=lambda r:int(r['episode'][2:]));keys=['loading_gap_rmse_mm','loading_width_rmse_mm','unloading_gap_rmse_mm'];limits=np.array([2.,3.,2.])
    values=np.array([[r[k] for k in keys] for r in records]);ratios=values/limits
    fig,ax=plt.subplots(figsize=(8,7),constrained_layout=True);ax.imshow(ratios,vmin=0,vmax=3,cmap='RdYlGn_r',aspect='auto')
    ax.set_xticks(range(3),['Compression gap\nTarget ≤2 mm','Width\nTarget ≤3 mm','Unloading gap\nTarget ≤2 mm'])
    ax.set_yticks(range(len(records)),[f"{labels[r['material']]} | {[6,11,21,31][int(r['episode'][2:])%4]} N"+(' *' if int(r['episode'][2:])%4==2 else '') for r in records])
    for i in range(len(records)):
        for j in range(3):ax.text(j,i,f'{values[i,j]:.2f} mm',ha='center',va='center',fontsize=10)
    ax.set_title('Frozen parameters across all loads | grid 80\n* 21 N excluded from fitting; Play-Doh fails the finer-grid width check')
    fig.savefig(out/'crossload_errors.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--scratch',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--plots',action='store_true');args=a.parse_args();evaluate(args.scratch,args.out)
    if args.plots:plots(args.scratch,args.out)
