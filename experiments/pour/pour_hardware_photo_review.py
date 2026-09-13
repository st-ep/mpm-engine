"""Reproduce a review of photographed glycerin endpoints; never runs MPM.

Receiver-primary revision after user visual audit rejected source annotations.
Manual image coordinates are inputs, separate from the provisional command map.
Measurements use only image pixels and printed graduations. Targets and MPM
outputs enter only the subsequent comparison. Original photographs are untouched.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import median_filter

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / 'pouring_real_data/pouring_figs/measurments_5seeds'
OUT = ROOT / 'out/pour_hardware_receiver_remap_review_20260911'
PREVIOUS = ROOT / 'out/pour_hardware_photo_review_20260911'
PRIOR_MAPPING = ROOT / 'out/pour_hardware_receiver_review_20260911'
ANNOTATIONS = OUT / 'annotations.json'
PHOTO_MAPPING = ROOT / 'pouring_real_data/pouring_figs/photo_target_mapping.csv'
PHOTO_ORDER_NOTE = PHOTO_MAPPING.with_name('PHOTO_ORDER.md')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def target_assignment(row, annotations):
    default_target=40+20*row['slot']
    overrides=annotations['mapping']['overrides']
    stem=Path(row['image']).stem
    if stem in overrides:
        assert row['seed']==1
    return dict(target_ml=overrides.get(stem,default_target),
        original_filename_target_ml=default_target,
        mapping_status='user_proposed_reassignment_unverified' if stem in overrides
                       else 'provisional_filename_order')


def measure_receiver(im, a):
    x=a['read_x_px']
    rgb=np.asarray(im,dtype=float)
    # The abrupt loss of blue at the amber/white interface. A vertical median
    # removes narrow red graduation strokes before edge selection.
    ratio=np.median(rgb[:,x:x+31,2]/np.maximum(rgb[:,x:x+31,0],1.),axis=1)
    profile=gaussian_filter1d(median_filter(ratio,size=17),2.)
    ticks={float(v):float(y) for v,y in a['ticks_y_px'].items()}
    lo,hi=int(ticks[250]+15),int(ticks[50]-12)
    y=int(lo+np.argmax(-np.gradient(profile)[lo:hi]))
    # Contrast quartiles document the visible transition width.
    above=float(np.median(profile[y-35:y-15]))
    below=float(np.median(profile[y+15:y+35]))
    assert above-below>.08, (a,y,above,below)
    ys=np.arange(y-15,y+16)
    edges=[int(ys[np.argmin(abs(profile[ys]-(above*(1-f)+below*f)))]) for f in [.25,.75]]
    positions=np.array([ticks[v] for v in sorted(ticks,reverse=True)])
    volumes=np.array(sorted(ticks,reverse=True))
    assert np.all(np.diff(positions)>0)
    assert positions[0]<y<positions[-1]
    ml=float(np.interp(y,positions,volumes))
    j=int(np.searchsorted(positions,y))
    return dict(receiver_ml=ml, receiver_read_low_ml=ml-12.5,
                receiver_read_high_ml=ml+12.5,receiver_meniscus_y_px=y,
                receiver_transition_low_y_px=min(edges),receiver_transition_high_y_px=max(edges),
                receiver_read_x_px=x,receiver_upper_tick_ml=float(volumes[j-1]),
                receiver_upper_tick_y_px=float(positions[j-1]),
                receiver_lower_tick_ml=float(volumes[j]),receiver_lower_tick_y_px=float(positions[j]),
                optical_contrast_blue_red=above-below)


def receiver_audit(ax,im,m,title):
    x,y=m['receiver_read_x_px'],m['receiver_meniscus_y_px']
    box=(x-180,y-165,x+300,y+165)
    ax.imshow(im.crop(box),extent=(box[0],box[2],box[3],box[1]))
    for key in ['upper','lower']:
        yy,v=m[f'receiver_{key}_tick_y_px'],m[f'receiver_{key}_tick_ml']
        ax.plot([x-15,x+150],[yy,yy],color='#1976b2',lw=1.2)
        ax.text(x+155,yy,f'{v:.0f} mL',va='center',fontsize=8,color='#064c77',
                bbox=dict(fc='white',ec='none',alpha=.9,pad=1))
    ax.plot([x-100,x+80],[y,y],color='#a6241d',lw=1.4)
    ax.plot(x+15,y,'+',color='#a6241d',ms=8)
    ax.set_title(title,loc='left',fontsize=9)
    ax.set_xlabel(f'Receiver: {m["receiver_ml"]:.1f} mL; reading convention: ±12.5 mL\n'
                  'Primary endpoint reading; cup accuracy is uncalibrated',fontsize=8)
    ax.set_xticks([]);ax.set_yticks([])


def comparison_report(rows, annotations):
    """Join receiver readings to the authorized provisional order, then plot."""
    provenance_path=ROOT/'out/pour_navier_calibration/philip_corrected_labeled_20260908/provenance.json'
    provenance=json.loads(provenance_path.read_text())
    handoff=ROOT/'final/philip_corrected_handoff.zip'
    assert sha(handoff)==provenance['zip_sha256']
    with zipfile.ZipFile(handoff) as z:
        for name, expected in provenance['package_file_sha256'].items():
            assert hashlib.sha256(z.read(name)).hexdigest()==expected
        commands=list(csv.DictReader(io.StringIO(z.read('angle_table.csv').decode())))
    predictions={p['target_ml']:p for p in provenance['predictions']}
    checked_commands=[]
    for command in commands:
        target=int(command['target_ml'])
        p=predictions[target]
        assert float(command['command_angle_deg'])==p['command_angle_deg']
        assert float(command['tilt_duration_s'])==p['tilt_duration_s']
        for kind in ['result','review']:
            assert sha(ROOT/p[f'{kind}_path'])==p[f'{kind}_sha256']
        result=json.loads((ROOT/p['result_path']).read_text())
        assert abs(result['receiver_ml']-p['predicted_ml'])<1e-10
        checked_commands.append(dict(**command,mpm_receiver_ml=p['predicted_ml'],
            mpm_source_depletion_ml=result['source_depletion_ml'],
            mpm_outside_ml=result['outside_ml'],
            within_original_1ml_target_tolerance=p['original_simulation_target_tolerance_passed'],
            hardware_repeats=5 if target<=160 else 0,
            result_path=p['result_path'],result_sha256=p['result_sha256']))
    write_csv(OUT/'verified_commands.csv',checked_commands)
    targets=np.array([60,80,100,120,140,160])
    summary=[]
    mapped=[]
    assigned=[dict(**r,**target_assignment(r,annotations)) for r in rows]
    for slot,target in enumerate(targets,start=1):
        rr=sorted([r for r in assigned if r['target_ml']==target],key=lambda r:r['seed'])
        assert [r['seed'] for r in rr]==[1,2,3,4,5]
        p=predictions[int(target)]
        for r in rr:
            mapped.append(dict(command_angle_deg=p['command_angle_deg'],
                tilt_duration_s=p['tilt_duration_s'],mpm_receiver_ml=p['predicted_ml'],
                review_flag='user_proposed_mapping_needs_record_confirmation'
                    if r['mapping_status']=='user_proposed_reassignment_unverified' else '',**r))
        measured=np.array([r['receiver_ml'] for r in rr])
        summary.append(dict(target_ml=int(target),command_angle_deg=p['command_angle_deg'],
            tilt_duration_s=p['tilt_duration_s'],mpm_receiver_ml=p['predicted_ml'],
            **{f'seed{r["seed"]}_ml':r['receiver_ml'] for r in rr},
            mean_ml=float(measured.mean()),sample_sd_ml=float(measured.std(ddof=1)),
            n=5,measurement_basis='receiver_cup',
            reading_convention_half_width_ml=12.5,
            mapping_status='includes_user_proposed_reassignment' if target>=120
                           else 'provisional_filename_order'))
    assert len(mapped)==30 and len({r['image_sha256'] for r in mapped})==30
    write_csv(OUT/'command_photo_mapping.csv',mapped)
    write_csv(OUT/'summary.csv',summary)
    changed=[r for r in mapped if r['mapping_status']=='user_proposed_reassignment_unverified']
    assert len(changed)==3
    write_csv(OUT/'mapping_changes.csv',[
        dict(image=r['image_path'],original_file_slot=r['slot'],
            previous_target_ml=r['original_filename_target_ml'],proposed_target_ml=r['target_ml'],
            unchanged_receiver_ml=r['receiver_ml'],
            previous_command_angle_deg=predictions[r['original_filename_target_ml']]['command_angle_deg'],
            proposed_command_angle_deg=r['command_angle_deg'],
            status='User-proposed; not independently verified') for r in changed])
    old_summary={int(r['target_ml']):r for r in csv.DictReader((PRIOR_MAPPING/'summary.csv').open())}
    write_csv(OUT/'summary_before_after.csv',[
        dict(target_ml=r['target_ml'],previous_mean_ml=old_summary[r['target_ml']]['mean_ml'],
            proposed_mean_ml=r['mean_ml'],previous_sd_ml=old_summary[r['target_ml']]['sample_sd_ml'],
            proposed_sd_ml=r['sample_sd_ml']) for r in summary])
    headers=['Target','MPM','Seed 1','Seed 2','Seed 3','Seed 4','Seed 5','Mean ± SD']
    table=[]
    for r in summary:
        table.append([str(r['target_ml']),f'{r["mpm_receiver_ml"]:.1f}',
            *[f'{r[f"seed{s}_ml"]:.0f}' for s in range(1,6)],
            f'{r["mean_ml"]:.1f} ± {r["sample_sd_ml"]:.1f}'])
    md='| '+' | '.join(headers)+' |\n| '+' | '.join(['---:']*len(headers))+' |\n'
    md+='\n'.join('| '+' | '.join(line)+' |' for line in table)+'\n'
    (OUT/'table.md').write_text('All volumes in mL. PROVISIONAL user-proposed seed1 reassignment.\n\n'+md+
        '\nReceiver readings rounded to 1 mL for display; statistics use unrounded interpolations. '
        'SD is sample standard deviation across all five repeats. '
        'Seed1 assignments: IMG_0888=120, IMG_0886=140, IMG_0887=160 mL. '
        'Proposed by the user after endpoint review; not independently confirmed.\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    fig,ax=plt.subplots(figsize=(8.5,5.8))
    fig.subplots_adjust(left=.12,right=.97,bottom=.13,top=.89)
    ax.plot([50,170],[50,170],color='#92979e',ls='--',lw=1.4,label='Target',zorder=1)
    ax.plot(targets,[r['mpm_receiver_ml'] for r in summary],'-s',
        color='#256ba4',lw=1.15,ms=5,mfc='white',mew=1.4,label='MPM prediction',zorder=4)
    means=np.array([r['mean_ml'] for r in summary]);sd=np.array([r['sample_sd_ml'] for r in summary])
    ax.errorbar(targets,means,yerr=sd,color='#bd5624',fmt='o-',lw=1.5,ms=6,
        capsize=4,label='Hardware mean ± SD (n = 5)',zorder=5)
    for seed in range(1,6):
        rr=sorted([r for r in mapped if r['seed']==seed],key=lambda r:r['target_ml'])
        ax.scatter(targets+(seed-3)*1.7,[r['receiver_ml'] for r in rr],
            color='#3f444b',s=22,alpha=.65,zorder=3,label='Individual repeats' if seed==2 else None)
    ax.set(xlim=(50,170),ylim=(40,178),xticks=targets,yticks=np.arange(40,181,20),
        xlabel='Target volume (mL)',ylabel='Receiver volume (mL)')
    ax.grid(axis='y',color='#e5e7e9',lw=.7)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left',frameon=False,fontsize=10)
    ax.set_title('Glycerin pouring',loc='left',fontsize=15,weight='bold',pad=15)
    for ext in ['png','pdf','svg']:
        fig.savefig(OUT/f'pouring_comparison.{ext}',dpi=220)
    plt.close(fig)
    # A standalone table image is convenient for side-by-side visual review.
    fig,ax=plt.subplots(figsize=(10.4,3.9));ax.axis('off')
    fig.subplots_adjust(left=.02,right=.98,top=.79,bottom=.2)
    display=[line.copy() for line in table]
    for line in display[-3:]:line[2]+=' *'
    t=ax.table(cellText=display,colLabels=headers,cellLoc='center',loc='center',
               colWidths=[.10,.11,.11,.11,.11,.11,.11,.23])
    t.auto_set_font_size(False);t.set_fontsize(11);t.scale(1,1.6)
    for (i,j),cell in t.get_celld().items():
        cell.set_edgecolor('#dce1e5');cell.set_linewidth(.6)
        cell.set_facecolor('#edf2f5' if i==0 else ('#f8fafb' if i%2==0 else 'white'))
        if i==0:cell.set_text_props(weight='bold')
    for i in [4,5,6]:t[(i,2)].set_text_props(color='#a22832',weight='bold')
    fig.text(.035,.92,'Glycerin endpoint readings — all volumes in mL',fontsize=15,weight='bold')
    fig.text(.035,.84,'PROVISIONAL user-proposed seed 1 reassignment; five repeats per target',fontsize=10,color='#725045')
    fig.text(.035,.13,'* Seed 1: IMG_0888 → 120, IMG_0886 → 140, IMG_0887 → 160 mL; unverified assignments.\n'
        'Seed readings rounded to 1 mL. Mean and sample SD use unrounded interpolations.\n'
        'The ±12.5 mL reading convention is separate from SD; total accuracy is uncalibrated.',fontsize=9,linespacing=1.5)
    for ext in ['png','pdf']:fig.savefig(OUT/f'pouring_table.{ext}',dpi=220)
    plt.close(fig)
    report=dict(status='Receiver-primary review; provisional command assignment',
        measurement_basis='receiver_cup',
        counts=dict(photos=30,seeds=5,targets_with_hardware=6,excluded_repeat_photos=0),
        receiver_values_unchanged_from_user_reviewed_audit=True,
        command_and_simulation_hashes_verified=True,
        receiver_reading_convention_half_width_ml=12.5,
        total_measurement_uncertainty_ml=None,
        source_readings_status='Rejected by user visual review; excluded from current measurements and validation',
        previous_source_receiver_validation_claim='Withdrawn: rejected source readings cannot validate receiver readings',
        mapping_flags=[dict(image=r['image_path'],previous_target_ml=r['original_filename_target_ml'],
            proposed_target_ml=r['target_ml'],receiver_ml=r['receiver_ml'],
            status='User-proposed after endpoint review; independent confirmation outstanding') for r in changed],
        initial_fill=annotations['initial_fill'],mapping=annotations['mapping'],
        measurement_selection=annotations['measurement_selection'],
        simulations_launched=0,manuscript_edited=False,
        preserved_numerical_limitations={k:provenance[k] for k in [
            'calibration_interpretation','original_time_check_passed','numerical_acceptance_passed',
            'original_time_difference_ml','all_command_simulation_target_checks_passed',
            'failed_original_simulation_target_checks_ml','simulation_target_tolerance_ml']})
    (OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    manifest=dict(previous_review=str(PREVIOUS.relative_to(ROOT)),
        previous_mapping_review=str(PRIOR_MAPPING.relative_to(ROOT)),measurement_basis='receiver_cup',
        dataset_order=dict(path=str(PHOTO_MAPPING.relative_to(ROOT)),sha256=sha(PHOTO_MAPPING),
            note_path=str(PHOTO_ORDER_NOTE.relative_to(ROOT)),note_sha256=sha(PHOTO_ORDER_NOTE)),
        handoff=dict(path=str(handoff.relative_to(ROOT)),sha256=sha(handoff)),
        simulation_provenance=dict(path=str(provenance_path.relative_to(ROOT)),sha256=sha(provenance_path)),
        predictions=provenance['predictions'],
        photos=[dict(path=r['image_path'],sha256=r['image_sha256']) for r in mapped],
        artifacts={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ANNOTATIONS,OUT/'README.md',
            PHOTO_MAPPING,PHOTO_ORDER_NOTE,OUT/'share_plot_revision.json',
            OUT/'revision.json',OUT/'receiver_readings.csv',OUT/'mapping_changes.csv',OUT/'summary_before_after.csv',
            OUT/'command_photo_mapping.csv',OUT/'summary.csv',OUT/'verified_commands.csv',
            OUT/'validation.json',OUT/'pouring_comparison.png',OUT/'pouring_comparison.pdf',
            OUT/'pouring_table.png',OUT/'receiver_reading_audit.pdf',OUT/'IMG_0883_review.png']})
    (OUT/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(md)


def main():
    annotation_data=json.loads(ANNOTATIONS.read_text())
    dataset_rows=list(csv.DictReader(PHOTO_MAPPING.open()))
    dataset_mapping={(int(r['seed']),Path(r['image']).name):r for r in dataset_rows}
    assert len(dataset_rows)==len(dataset_mapping)==30
    # Verify that choosing the receiver did not silently revise its measurements.
    revision=json.loads((OUT/'revision.json').read_text())
    assert sha(PREVIOUS/'paired_readings.csv')==revision['previous_paired_readings_sha256']
    assert sha(PRIOR_MAPPING/'receiver_readings.csv')==revision['previous_receiver_readings_sha256']
    assert sha(PRIOR_MAPPING/'provenance.json')==revision['previous_mapping_provenance_sha256']
    previous={r['image']:r for r in csv.DictReader((PREVIOUS/'paired_readings.csv').open())}
    old_annotations=json.loads((PREVIOUS/'annotations.json').read_text())
    assert annotation_data['receiver']==old_annotations['receiver']
    rows=[]
    with PdfPages(OUT/'receiver_reading_audit.pdf') as pdf:
        for seed in range(1,6):
            files=sorted((INPUT/f'seed{seed}').glob('*.jpg'))
            assert len(files)==6
            fig,axes=plt.subplots(3,2,figsize=(11.7,10.6),layout='constrained')
            for slot,(p,ax) in enumerate(zip(files,axes.flat),start=1):
                with Image.open(p) as im:
                    m=measure_receiver(im,annotation_data['receiver'][p.stem])
                    assignment=target_assignment(dict(seed=seed,slot=slot,image=p.name),annotation_data)
                    data_row=dataset_mapping[(seed,p.name)]
                    assert int(data_row['target_ml'])==assignment['target_ml']
                    assert int(data_row['original_filename_slot'])==slot
                    assert data_row['image_sha256']==sha(p)
                    suffix=' *' if assignment['mapping_status']=='user_proposed_reassignment_unverified' else ''
                    receiver_audit(ax,im,m,f'seed{seed} / {p.name} / file slot {slot} / '
                        f'{assignment["target_ml"]} mL{suffix}')
                    if p.stem=='IMG_0883':
                        detail,da=plt.subplots(1,2,figsize=(11.5,4.8))
                        x,y=m['receiver_read_x_px'],m['receiver_meniscus_y_px']
                        da[0].imshow(im.crop((x-180,y-165,x+300,y+165)))
                        da[0].axis('off');da[0].set_title('Original pixels: receiver crop',fontsize=11)
                        receiver_audit(da[1],im,m,'Existing annotation retained')
                        detail.suptitle('seed1 / IMG_0883.jpg / slot 2 — close-up review',fontsize=13)
                        detail.tight_layout(rect=(0,.04,1,.95))
                        detail.savefig(OUT/'IMG_0883_review.png',dpi=180)
                        plt.close(detail)
                assert abs(m['receiver_ml']-float(previous[p.name]['receiver_ml']))<1e-10
                assert m['receiver_meniscus_y_px']==float(previous[p.name]['receiver_meniscus_y_px'])
                rows.append(dict(seed=seed,slot=slot,image=p.name,
                    image_path=str(p.relative_to(ROOT)),image_sha256=sha(p),**m))
            fig.suptitle('Receiver-cup primary measurements\n'
                'Red: liquid boundary; blue: graduations; * user-proposed target reassignment',fontsize=12)
            pdf.savefig(fig,dpi=170)
            fig.savefig(OUT/f'receiver_audit_seed{seed}.jpg',dpi=170)
            plt.close(fig)
    write_csv(OUT/'receiver_readings.csv',rows)
    assert sha(OUT/'receiver_readings.csv')==sha(PRIOR_MAPPING/'receiver_readings.csv')
    comparison_report(rows,annotation_data)


if __name__=='__main__':
    main()
