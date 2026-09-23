"""Validate the actual encoded deliverable and expose reviewable evidence."""
from pathlib import Path
import hashlib, json, subprocess, csv, argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from opening_preview.render_opening import REVEALS, FADE_SECONDS
from build_video import HERE,ROOT,REVIEW,TIMELINE,SOURCES,FPS,TOTAL,draw_section,TEXT_LOG
from narrative_captions import ALL_SECTIONS,CUES,caption_at

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
assert all(x['size']>=18 or (x['size']==15 and x['bbox'][1]>=643 and x['bbox'][3]<665) for x in TEXT_LOG)
# Review every spoken cue in the encoded video and verify a fixed subtitle band.
caption_cap=cv2.VideoCapture(str(out));caption_clock=0;caption_review=[]
for name,duration in ALL_SECTIONS:
    end_previous=0
    for cue_start,cue_end,caption in CUES[name]:
        assert cue_start==end_previous and cue_end>cue_start
        assert len(caption)/(cue_end-cue_start)<=20
        end_previous=cue_end
        when=caption_clock+(cue_start+cue_end)/2
        caption_cap.set(cv2.CAP_PROP_POS_MSEC,when*1000);ok,bgr=caption_cap.read();assert ok
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        # Center-padding rows are dark on every caption regardless of text width.
        for row in [668,707]:
            assert np.abs(rgb[row,600:680].astype(float)-[38,53,62]).mean()<10,(name,when,row)
        Image.fromarray(rgb).save(actual/f'caption_{name}_{cue_start:g}.png')
        caption_review.append({'section':name,'start_s':caption_clock+cue_start,'end_s':caption_clock+cue_end,'text':caption})
    assert end_previous==duration
    caption_clock+=duration
caption_cap.release()
validation={
 'checks':{'full_decode':True,'duration':True,'size_under_20_decimal_MB':True,'progressive':True,
           'h264_yuv420p':True,'fps_at_least_20':True,'height_at_least_480':True,
           'frame_count':True,'no_personal_metadata_tags':True,'text_bounds_all_states':True,'captions_every_section':True,'caption_band_same_position':True},
 'caption_band_y_px':[665,710],'caption_cues':caption_review,
 'review_status':'Working cut; timing not finalized' if args.working_cut else 'Submission timing checked',
 'submission_duration_under_180s':TOTAL<=180,
 'duration_seconds':TOTAL,'size_bytes':out.stat().st_size,'frame_count':TOTAL*FPS,
 'video':video,'container':probe['format'],
 'compression_sample_psnr_db':scores,
 'visual_review':'See REVIEW.md. Pixel checks do not substitute for visual review.'}
(REVIEW/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
hw=json.loads((HERE/'hardware_clip.json').read_text())
source_paths=[HERE/'hardware_slide.py',HERE/'prepare_hardware_slide.py',
 HERE/'assets/hardware_identification.mp4',HERE/'assets/hardware_identification.json',
 ROOT/'out/press_weakform_restart_20260914/protocol.json',
 *[ROOT/f'out/press_weakform_restart_20260914/{m}/fit.json' for m in ['play_doh','butter_slime','plasticine']],
 ROOT/'out/hardware_press_shape_figure_20260915/provenance.json',
 HERE/'render_shaping_hand.py',HERE/'assets/shaping_hand/target.vtp',
 *sorted((HERE/'assets/shaping_hand').glob('*_plan_?.json')),
 ROOT/'out/x_consistent_force_study_20260911/franka/panda_model_snapshot/panda.xml',
 ROOT/'experiments/robotics/x_shaping_franka.py',ROOT/'experiments/robotics/plastic_shaping_figure.py',
 HERE/'prepare_shaping_identification.py',
 HERE/'assets/shaping_identification.npz',HERE/'assets/shaping_identification.json',
 ROOT/'out/press_separated_20260913/monotonic320/separated_A/identification.json',
 ROOT/'out/press_separated_20260913/monotonic320/separated_B/identification.json',
 ROOT/'out/press_paper_update_20260913/execution_summary.json',
 ROOT/'out/press_paper_update_20260913/planning_spec.json',
 ROOT/'out/putting_flat_sky_20260913/plan_A/plan.json',
 ROOT/'out/putting_flat_sky_20260913/plan_B/plan.json',
 ROOT/'out/putting_flat_sky_20260913/final_sources/putting.py',
 *SOURCES.values(),ROOT/'paper/videos/provenance.json',ROOT/'paper/videos/README.md',
 ROOT/'out/strip_texture_20260912/fit_A/identification.json',
 ROOT/'out/strip_texture_20260912/fit_B/identification.json',
 HERE/'prepare_insertion_probe.py',HERE/'assets/insertion_probe.npz',HERE/'assets/insertion_probe.json',
 ROOT/'out/strip_texture_20260912/media/truth_A_frames.npy',
 ROOT/'out/strip_texture_20260912/media/truth_B_frames.npy',
 ROOT/'out/strip_texture_20260912/media/video_time.npy',
 ROOT/'paper/icra2027/paper.tex',ROOT/'paper/icra2027/paper.pdf',
 HERE/'references/2027_ICRA_FORM.pdf',
 HERE/'method01_preview/render_method01.py',
 HERE/'method02_preview/render_method02_animated.py',
 HERE/'method02_preview/render_method02_flow.py',
 HERE/'method02_preview/method02_block1_animation_provenance.json',
 HERE/'method02_preview/shaping_pair/pair_provenance.json',
 HERE/'method_slide_frame.py',HERE/'narrative_captions.py',
 HERE/'pouring_slide.py',HERE/'assets/pouring_storyboard_provenance.json',
 HERE/'prepare_pouring_targets.py',HERE/'render_pouring_targets.py',HERE/'build_pouring_target_clips.py',
 HERE/'check_pouring_targets.py',HERE/'assets/pouring_targets/validated_clips.json',
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
prov={'scope':'Private video edit. Frozen target-pour physics plans are replayed to capture display states and checked against their original volume histories; no new optimization, material calibration, or measurements. Existing stereo reconstruction is re-evaluated for display with unchanged inputs/settings and checked against its frozen frame.',
 'chapters':chapters,'hardware':hw,'sources':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in dict.fromkeys(source_paths)],
 'assets':[{'path':str(p.relative_to(HERE)),'sha256':sha(p)} for p in sorted((HERE/'assets').glob('*.png'))],
 'outputs':{p.name:{'sha256':sha(p),'size_bytes':p.stat().st_size} for p in [out,HERE/'ICRA2027_video_master.mp4']},
 'editing':{'source_geometry':'Measured source imagery is unchanged apart from documented crops/resizing. Method 1 shows synchronized surface envelopes of saved inferred particles and a schematic triangulation diagram with actual two-camera texture crops; source kinematics are unchanged.',
  'method01':json.loads((HERE/'method01_preview/animation_provenance.json').read_text()),
  'method02':json.loads((HERE/'method02_preview/method02_block1_animation_provenance.json').read_text()),
  'pouring_combined':json.loads((HERE/'assets/pouring_storyboard_provenance.json').read_text()),
  'hardware_combined':{'duration_s':30,'replaces':['pressing 16 s','hardware 14 s','scans 10 s'],
      'layout':'Three columns of materials on the left, three rows: recorded press, MPM press, final scans. Identified-law graph top right, real four-pinch execution bottom right. Press-to-law arrow starts at actual video right edge x737 and meets law card at x812. Straight return arrow y365 exits just above the rounded card corner and meets MPM at x737. Execution bottom y652 matches score-text bounding-box bottoms exactly.',
      'reveals_s':{'identification_recordings':0,'law_and_prediction':6,'execution':12,'final_scans':26},
      'fades':'Future content at 6 percent opacity, smooth 0.6 s reveals including corresponding arrows.',
      'identification':json.loads((HERE/'assets/hardware_identification.json').read_text()),
      'evaluation':'At 6 s top row crossfades to the separate 21 N recordings and synchronizes with force-driven MPM below. Both loop physical source 0–2.0 s in synchronized 3 s cycles: 0.2 s start hold, 2.5 s at 0.8x speed, 0.3 s end hold. No reversed playback. The initial identification excerpts also loop this early interval; their full recordings still supplied the unchanged fits. Coefficients come from separate 31 N recordings, not the evaluation trials.',
      'graph':'Same monotone isochoric coaxial Hencky/perfect-J2 scalar response as simulation slide, now using the actual hardware frozen E/Y and known nu=.45. Deviatoric Kirchhoff stress norm vs deviatoric total logarithmic strain norm. Colors match material headings. Conditional identified effective models, not measured stress-strain data.',
      'hardware_execution':'User-confirmed red matched-ID four-pinch plan. Complete existing source at 4.5x from section time 12 s, then final hold. Fixed crop [480,180,1530,700], uniform aspect-preserving resize.',
      'scans':'Existing paper scan images and dashed targets. XY footprint IoU 75.7, 72.4, 77.8 percent; rigid alignment, no scaling; missing surfaces interpolated. Displayed as target overlap (IoU), not surface error in mm.',
      'new_physics_or_identification':False},
  'simulation_shaping':{'layout':'Identification left; 2x2 shaping grid right with exactly 38 px horizontal and vertical panel gaps, row labels outside panels; 95 px connector at y=411, aligned with the expanded law-card top. No separate target thumbnail.',
      'identification':'Recorded independent A/B presses, 0–2 and 2–4 s. Display-only blue/orange specimen tint preserves texture and shading; fades to original gray at 4–4.25 s. Mask provenance in shaping_identification.json. No new physics or fits.',
      'laws':'One shared Hencky/J2 constitutive-response graph, blue A and orange B, with actual frozen E/Y annotations. Each curve appears after its press. Monotone isochoric coaxial loading: deviatoric Kirchhoff stress norm = min(E/(1+nu) * deviatoric logarithmic strain norm, Y), known nu=0.3. Not a measured force-displacement trace.',
      'hand_renders':{p.stem:json.loads(p.read_text()) for p in sorted((HERE/'assets/shaping_hand').glob('*_plan_?.json'))},
      'hand_scope':'Kinematic Panda hand illustration at saved tool poses, following the same model and adapter convention as the paper. Does not assert a new robot dynamics simulation.',
      'outline':'Exact frozen inputs/target.vtp, common camera and scale. Dashed silhouette appears only in the last source frame, held for final comparison; no outline during moving shaping footage. No per-case registration.',
      'time':'23 s section, same four-second identification and 16-second shaping sequence plus three-second final hold. All six pinches retained. Low paper-style view during contact, common final view after withdrawal.',
      'row_labels_xy':[[656,280],[656,526]],'panel_bounds_xywh':[[678,178,242,208],[958,178,242,208],[678,424,242,208],[958,424,242,208]],
      'evaluation':'Unchanged frozen surface errors explicitly labeled Surface error in regular black 21 px text with bold numeric values in each panel at 20–23 s alongside final target silhouettes.'},
  'golf_layout':{'header':'Green Results · Simulation / prefix and dark title, matching insertion',
      'identification':'Previous bending identification reused; no second identification demonstration',
      'task':'Stop the ball in the target by planning forward-stroke duration and aim angle (horizontal yaw); backswing prescribed',
      'source':'paper/videos/02_golf_matched_swapped.mp4',
      'crops_xyxy':[[34,113,818,425],[824,113,1608,425],[34,579,818,891],[824,579,1608,891]],
      'panel_bounds_xywh':[[84,198,568,226],[668,198,568,226],[84,428,568,226],[668,428,568,226]],
      'crop_qualification':'Identical vertical crop and uniform resize. Some upper robot wrist/arm and empty sky/foreground removed; flexible clubs, grippers, ball paths, and targets retained.',
      'playback':'Three synchronized 5 s loops: full source playback mapped to 4.5 s, then 0.5 s actual final-state hold',
      'labels':'Matched/swapped planning-model definitions, A/B rows; no sidebar or small playback footer'},
  'insertion_identification':{'source':'out/strip_texture_20260912/media/truth_A_frames.npy and truth_B_frames.npy',
      'material':'Actual material A loading followed by actual material B loading',
      'physical_loading_interval_s':[0.30,0.90],
      'playback':'A bends at 0–1.6 s, B at 2–3.6 s (0.375x physical speed). Each episode then held. Insertion at 1x from section time 4 s.',
      'parameter_reveal_s':{'A':[1.6,1.76],'B':[3.6,3.76]},
      'color':'Display-only specimen tint: blue A, orange B, return to original gray at 3.8–4 s. Masks use original geometry/camera and preserve fixture occlusion.',
      'comparison_reveal_s':[4,4.5], 'small_playback_footer_removed':True},
  'insertion_layout':{'header':'Single-line green results prefix and dark experiment title; transfer to a different action and geometry',
      'identification':'Identify material law heading; each observed bend reveals its sigma_A/B = E_A/B T_nu(F) equation in the Identified material laws card; the complete card points to the tests.',
      'law_notation':{'equivalent_to':'The reference paper Table I fixed-corotated elastic law',
          'known_poisson_ratio':0.45,
          'T_nu(F)':'(F-R) F^T / [J(1+nu)] + nu (J-1) I / [(1+nu)(1-2nu)]',
          'display':'One card contains two equations with numeric E in kPa colored by material, the Colored values: identified stiffness E legend, and T_nu(F) labeled Deformation response. All black mathematical notation, including subscripts and legend E, uses the same regular 22 px math face. All three explanatory text labels use regular 22 px black text.'},
      'fitted_E_kpa':{'A':80.60889355693937,'B':248.11232605994397},
      'alignment':{'law_card_xyxy':[260,238,495,604],'divider_and_arrow_y':428,'arrows_moved':False},
      'task_label':'In the top subtitle: Identify the material law from bending. Use it to insert the rod through the hole by controlling gripper height and tilt. No extra task text below the diagram.',
      'comparison':'Compact 2 by 2 grid; headings explain the model used for planning'},
  'timing':'Opening uses timed reveals. Method 1 lasts 24 seconds, with reconstruction revealed at 4 seconds and inputs at 16 seconds. Method 2 lasts 28 seconds, with reveals at 12 and 20 seconds; pressing inputs loop at half speed, and the matched shaping pair is phase-aligned across the final 8 seconds. Insertion 1×; golf three 5 s loops, each with 4.5 s complete source playback plus 0.5 s final hold; simulation shaping A/B pressing 1× then all six pinches at 1.4825× source speed, with a 3 s final hold; hardware pressing source 0–2.0 s at 0.8x speed in 3 s forward-only loops; hardware shaping 4.5×; pouring upper recorded-motion pair at 1× then final hold; six lower MPM particle replays of the frozen target plans each compress the full motion to 2 s, followed by a 0.5 s angle transfer and measured-result reveal. Water appears at 31 s. Final holds explicit.',
  'global_progress':'Redrawn during assembly from actual current timeline duration; cached section bars are covered.',
  'captions':{'source':'narrative_captions.py','band_y_px':[665,710],'text_center_y_px':687.5,'font_size_px':24,'style':'Timed spoken sentences; fixed white-on-dark subtitle band across every slide.'},
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
