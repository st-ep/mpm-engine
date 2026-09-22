from pathlib import Path
import cv2,numpy as np,json,subprocess,hashlib
from PIL import Image,ImageDraw,ImageFont
P=Path(__file__).resolve().parent;video=P/'method01_animated.mp4'
cap=cv2.VideoCapture(str(video));frames=[]
for t in [0,5.2,10.4,15.8]:
 cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,bgr=cap.read();assert ok
 im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));im.save(P/f'encoded_method01_{t:g}s.png');frames.append(np.array(im))
cap.release()
regions={'hardware_input':[44,201,314,368],'simulation_input':[44,453,314,647],'rgbd_surface':[387,213,558,356],'constant_volume':[581,213,753,356],'rgbd_particles':[774,213,947,356],'stereo_texture':[382,473,566,627],'triangulation':[587,473,748,627],'interior_field':[782,473,947,627],'output_motion':[1014,213,1234,356],'force':[1039,503,1219,637]}
checks={}
for k,(x,y,r,b) in regions.items():
 dif=np.mean(abs(frames[0][y:b,x:r].astype(float)-frames[-1][y:b,x:r].astype(float)))
 checks[k]={'mean_pixel_change':float(dif),'animated':bool(dif>.4)};assert dif>.4,(k,dif)
info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(video)]));v=info['streams'][0]
assert v['codec_name']=='h264' and v['pix_fmt']=='yuv420p' and int(v['nb_frames'])==400
assert abs(float(info['format']['duration'])-16)<.02
r=subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],capture_output=True,text=True);assert r.returncode==0 and not r.stderr
# Verify geometry scale and the timestamp match, not only visible pixel changes.
from render_method01 import data,lerp_frame,profile,centered_positions,playback_times,draw_method01,reveal_progress,FAINT_OPACITY,DIVIDER_Y,cameras
# Verify the same source motion and complete rendered frame repeat each cycle.
loop_checks=[]
for t in [0.,1.2,3.8]:
 ref=np.asarray(draw_method01(t,reveal=False))
 for cycle in [1,2,3]:
  assert playback_times(t)==playback_times(t+4*cycle)
  assert np.array_equal(ref,np.asarray(draw_method01(t+4*cycle,reveal=False)))
  loop_checks.append({'phase_s':t,'cycle':cycle,'identical_render':True})
assert np.allclose(np.subtract(playback_times(2),playback_times(1)),[.5,.5])
# Presentation opacity progresses independently of the repeating source clock.
assert reveal_progress(3.99,4)==0 and reveal_progress(4.61,4)==1
assert reveal_progress(7.99,8)==0 and reveal_progress(8.61,8)==1
assert 0<FAINT_OPACITY<.1
for t in [9.,10.5,11.9]:
 assert np.array_equal(np.asarray(draw_method01(t)),np.asarray(draw_method01(t+4)))
D=data();scales=[p.scale for p in D['hproj']];assert np.allclose(scales,scales[0])
errors=[];projection_errors=[]
for source_t in np.linspace(.05,2,350):
 errors.append(float(np.min(abs(D['rgb']['time']-source_t))))
points,_,_=lerp_frame(D['hp']['pos'],D['hp']['times'],.05)
assert np.max(np.linalg.norm(points-D['reference'],axis=1))==0
for source_t in [.05,1,2]:
 points,_,_=lerp_frame(D['hp']['pos'],D['hp']['times'],source_t);mesh=profile(centered_positions(points,D['reference'])).reshape(-1,3)
 q=[p(mesh)-p.center for p in D['hproj'][:3]]
 projection_errors.append(float(max(np.max(abs(q[j]-q[0])) for j in [1,2])))
assert max(errors)<.018 and max(projection_errors)<1e-10
assert np.allclose([p.center[1] for p in D['hproj']],284.5)
assert (453+647)/2 == 550
assert DIVIDER_Y==(137+691)/2
assert abs((DIVIDER_Y-36-368)-(461-(DIVIDER_Y+36)))<=1
for sprite,xy,lens in cameras():
 assert abs(xy[1]+sprite.height/4-DIVIDER_Y)<.3
assert D['surface_proj'].center[1]==D['sproj'].center[1]==550
assert 367-326 == 1007-966 == 41
assert 503+134 == 637  # Force X-axis aligns with caption tops.
source=D['hp']['pos'].astype(float);display=D['display_pos']
assert np.array_equal(source[...,2],display[...,2])
assert np.max(abs(np.ptp(source,axis=1)-np.ptp(display,axis=1)))<1e-12
assert np.max(abs(display[...,:2].mean(1)-D['reference'].astype(float)[:,:2].mean(0)))<1e-12
for idx in [24,25,50]:
 a=source[idx]-source[idx,0];b=display[idx]-display[idx,0]
 assert np.max(abs(a-b))<1e-12
before=np.linalg.norm(D['hproj'][3](source[25]).mean(0)-D['hproj'][3](source[24]).mean(0))
after=np.linalg.norm(D['hproj'][3](display[25]).mean(0)-D['hproj'][3](display[24]).mean(0))
assert after<.3
centering={'operation':'Display-only common XY translation; world Z unchanged','source_onset_centroid_shift_px':float(before),'display_onset_centroid_shift_px':float(after),'within_frame_geometry':'unchanged','source_and_displacement_colors':'original robot frame'}
sync={'output_alignment' :'upper motion y=284.5; force X-axis y=637, aligned to lower caption tops', 'max_rgb_time_error_s':max(errors),'upper_diagrams_and_output_scales_px_per_m':scales,'max_shape_projection_difference_px':max(projection_errors),'initial_displacement_m':0,'fixed_displacement_range_mm':[0,D['disp_max_m']*1000]}
(P/'animation_validation.json').write_text(json.dumps({'decode':'pass','frames':400,'duration_s':16,'resolution':[1280,720],'fps':25,'sha256':hashlib.sha256(video.read_bytes()).hexdigest(),'regions':checks,'loop_period_s':4,'source_playback_rate':.5,'loop_checks':loop_checks,'stage_reveal':{'starts_s':[4,8],'fade_s':.6,'initial_opacity':FAINT_OPACITY},'synchronization':sync,'display_centering':centering},indent=2)+'\n')
sheet=Image.new('RGB',(1280,760),'white');d=ImageDraw.Draw(sheet);font=ImageFont.truetype('/usr/share/fonts/truetype/lato/Lato-Regular.ttf',15)
for i,(t,fr) in enumerate(zip([0,5.2,10.4,15.8],frames)):
 x=(i%2)*640;y=(i//2)*380;sheet.paste(Image.fromarray(fr).resize((640,360)),(x,y+20));d.text((x+8,y+1),f'{t:g} seconds',font=font,fill='#21333e')
sheet.save(P/'animation_contact_sheet.png')
print('PASS: 400 decoded frames; input videos, six method panels and both outputs animate; shared shapes/scales and timing verified.')
