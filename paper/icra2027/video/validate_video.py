"""Validate the actual encoded deliverable and expose reviewable evidence."""
from pathlib import Path
import hashlib, json, subprocess, csv, argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from opening_preview.render_opening import REVEALS, FADE_SECONDS
from build_video import HERE,ROOT,REVIEW,TIMELINE,SOURCES,FPS,TOTAL,draw_section,TEXT_LOG

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

parser=argparse.ArgumentParser()
parser.add_argument('--working-cut',action='store_true',help='Allow duration over 180 s during user review; report the overrun.')
args=parser.parse_args()
out=HERE/'ICRA2027_video.mp4'
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(out)]))
video=probe['streams'][0]
assert len(probe['streams'])==1 and video['codec_type']=='video'
assert video['codec_name']=='h264' and video['pix_fmt']=='yuv420p'
# Some FFmpeg builds omit stream.field_order for progressive H.264. Check every
# decoded frame instead of interpreting absent metadata as either pass or fail.
frame_info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
    '-show_frames','-show_entries','frame=interlaced_frame','-of','json',str(out)]))
assert len(frame_info['frames'])==TOTAL*FPS
assert all(f['interlaced_frame']==0 for f in frame_info['frames'])
assert (video['width'],video['height'])==(1280,720)
assert video['r_frame_rate']=='25/1'
assert int(video['nb_frames'])==TOTAL*FPS
assert abs(float(probe['format']['duration'])-TOTAL)<.05
assert (args.working_cut or TOTAL<=180) and out.stat().st_size<20_000_000
for tags in [probe['format'].get('tags',{}),video.get('tags',{})]:
    assert not any(k.lower() in ['author','artist','comment','description','copyright','title'] for k in tags)
decode=subprocess.run(['ffmpeg','-v','error','-i',str(out),'-f','null','-'],capture_output=True,text=True)
assert decode.returncode==0 and not decode.stderr,decode.stderr
cap=cv2.VideoCapture(str(out));cap_master=cv2.VideoCapture(str(HERE/'ICRA2027_video_master.mp4'))
actual=REVIEW/'encoded';actual.mkdir(exist_ok=True)
font=ImageFont.truetype('/usr/share/fonts/truetype/lato/Lato-Regular.ttf',18)
chapters=[];scores=[];clock=0;frames=[]
for name,duration,title in TIMELINE:
    times=[clock+.2,clock+duration/2,clock+duration-.12]
    chapters.append(dict(section=name,start_s=clock,end_s=clock+duration,title=title))
    for i,t in enumerate(times):
        cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,bgr=cap.read();assert ok
        im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));im.save(actual/f'{name}_{i}.png')
        cap_master.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,ref=cap_master.read();assert ok
        scores.append({'time_s':t,'psnr_db':float(cv2.PSNR(ref,bgr))})
        thumb=im.resize((512,288));cell=Image.new('RGB',(512,316),'white');cell.paste(thumb,(0,28))
        ImageDraw.Draw(cell).text((8,5),f'{t:.2f} s | {name}',fill='#21333e',font=font);frames.append(cell)
    clock+=duration
cap.release();cap_master.release()
for page in range(0,len(frames),9):
    sheet=Image.new('RGB',(1536,948),'white')
    for j,im in enumerate(frames[page:page+9]):sheet.paste(im,((j%3)*512,(j//3)*316))
    sheet.save(actual/f'contact_{page//9+1}.jpg',quality=94)
# Representative pages for easy review outside the source tree.
pages=[Image.open(actual/f'{name}_1.png').convert('RGB') for name,_,_ in TIMELINE]
pages[0].save(REVIEW/'storyboard.pdf',save_all=True,append_images=pages[1:],resolution=96.)
# Exercise every text state, including progressive method reveals and final outcomes.
for name,duration,_ in TIMELINE:
    for t in [0,.2,4.49,4.5,9.99,10,12,duration-.04]:
        if t<duration:draw_section(name,t)
assert min(x['size'] for x in TEXT_LOG)>=18
validation={
 'checks':{'full_decode':True,'duration':True,'size_under_20_decimal_MB':True,'progressive':True,
           'h264_yuv420p':True,'fps_at_least_20':True,'height_at_least_480':True,
           'frame_count':True,'no_personal_metadata_tags':True,'text_bounds_all_states':True},
 'review_status':'Working cut; timing not finalized' if args.working_cut else 'Submission timing checked',
 'submission_duration_under_180s':TOTAL<=180,
 'duration_seconds':TOTAL,'size_bytes':out.stat().st_size,'frame_count':TOTAL*FPS,
 'video':video,'container':probe['format'],
 'compression_sample_psnr_db':scores,
 'visual_review':'See REVIEW.md. Pixel checks do not substitute for visual review.'}
(REVIEW/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
hw=json.loads((HERE/'hardware_clip.json').read_text())
source_paths=[*SOURCES.values(),ROOT/'paper/videos/provenance.json',ROOT/'paper/videos/README.md',
 ROOT/'paper/icra2027/paper.tex',ROOT/'paper/icra2027/paper.pdf',
 HERE/'references/2027_ICRA_FORM.pdf',
 HERE/'method01_preview/render_method01.py',
 HERE/'method01_preview/camera_designs/refine_compact.py',
 HERE/'method01_preview/camera_designs/make_comparison.py',
 HERE/'method01_preview/camera_designs/camera_A_refined.png',
 HERE/'method01_preview/camera_designs/camera_A_refined_geometry.json',
 HERE/'sources/stereo_press_A/provenance.json',
 HERE/'method01_preview/animation_provenance.json',
 HERE/'method01_preview/stereo_surface_cloud_provenance.json',
 HERE/'method01_preview/animation_cache_checks.json',
 HERE/'method01_preview/hardware_animation_cache_provenance.json',
 HERE/'method01_preview/animation_hardware_rgb.npz',
 ROOT/'press_real_data/ep0001/hand_rgb.mp4',
 ROOT/'press_real_data/ep0001/frames_hand.jsonl',
 HERE/'method01_preview/animation_stereo_kinematics.npz',
 ROOT/'press_real_data/handoff_ep0-7/ep0001/particles.npz',
 ROOT/'out/press_observation_assessment_20260913/ep0001/robot_force.npz',
 ROOT/'out/press_separated_20260913/monotonic320/fit_A/tracks.npz',
 HERE/'opening_preview/make_preview.py',HERE/'opening_preview/render_opening.py',
 HERE/'opening_preview/opening_proposal_720p.png',HERE/'opening_preview/provenance.json',
 ROOT/'paper/icra2027/figs/method_overview.png',ROOT/'paper/icra2027/figs/method_overview.provenance.json',
 ROOT/'paper/icra2027/figs/identification_plastic_shaping.png',
 ROOT/'out/hardware_press_shape_figure_20260915/provenance.json',
 ROOT/'out/press_separated_20260913/monotonic320/inputs_A/force.csv',
 ROOT/'out/pour_hardware_receiver_remap_review_20260911/summary.csv',
 ROOT/'out/pour_water_figure_20260915/water_receiver_readings.csv',
 *list((HERE/'sources/hardware_shaping').glob('*.mp4')),
 *sorted((HERE/'opening_preview/source_images').glob('*.jpg'))]
prov={'scope':'Display-only private edit. No new physics simulations, material fits, or measurements. Existing stereo reconstruction is re-evaluated for display with unchanged inputs/settings and checked against its frozen frame.',
 'chapters':chapters,'hardware':hw,'sources':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in dict.fromkeys(source_paths)],
 'assets':[{'path':str(p.relative_to(HERE)),'sha256':sha(p)} for p in sorted((HERE/'assets').glob('*.png'))],
 'outputs':{p.name:{'sha256':sha(p),'size_bytes':p.stat().st_size} for p in [out,HERE/'ICRA2027_video_master.mp4']},
 'editing':{'source_geometry':'Measured source imagery is unchanged apart from documented crops/resizing. Method 1 shows synchronized surface envelopes of saved inferred particles and a schematic triangulation diagram with actual two-camera texture crops; source kinematics are unchanged.',
  'method01':json.loads((HERE/'method01_preview/animation_provenance.json').read_text()),
  'timing':'Opening uses timed reveals. Method 1 uses slowed source intervals documented in method01 provenance; other method diagrams remain static. Insertion 1×; golf 0.5×; simulation shaping 1.25×; pressing 1×; hardware shaping 4.5×; pouring 1×. Final holds explicit.',
  'audio':'None; self-contained on-screen text.',
  'opening':{'duration_s':13,'fade_seconds':FADE_SECONDS,'reveals':REVEALS,
             'source':'Approved static opening; nine photos from final reference PDF Figure 1.'},
  'water_comparison':'User confirmed water executed the same glycerol-planned commands; one trial per target.',
  'source_qualifications':'paper/videos/README.md and provenance.json remain applicable.'},
 'build_script_sha256':sha(HERE/'build_video.py')}
(HERE/'provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
links='\n'.join(f'<button onclick="v.currentTime={c["start_s"]};v.play()">{int(c["start_s"])//60}:{int(c["start_s"])%60:02d} {c["title"]}</button>' for c in chapters)
(HERE/'WATCH.html').write_text('''<!doctype html><meta charset="utf-8"><title>ICRA 2027 video</title>
<style>body{font:17px system-ui;background:#f4f6f7;color:#21333e;max-width:1100px;margin:32px auto;padding:16px}
video{width:100%;background:#fff}button{display:block;background:white;border:1px solid #d9e1e5;padding:10px;margin:6px 0;border-radius:6px;cursor:pointer;text-align:left;width:100%}</style>
<h1>ICRA 2027 accompanying video</h1><p>'''+f'{TOTAL//60}:{TOTAL%60:02d}'+(' · Working cut' if args.working_cut else '')+''' · 720p · Self-contained, no audio</p>
<video id="v" controls preload="metadata" src="ICRA2027_video.mp4"></video><h2>Chapters</h2>'''+links)
print(json.dumps({'duration':TOTAL,'size_MB':out.stat().st_size/1e6,'full_decode':'PASS','frame_count':TOTAL*FPS,
                 'submission_duration_under_180s':TOTAL<=180,
                 'compression_sample_min_psnr':min(s['psnr_db'] for s in scores)},indent=2))
