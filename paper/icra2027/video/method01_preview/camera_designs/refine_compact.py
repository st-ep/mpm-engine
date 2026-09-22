"""Watertight rounded camera body, isolated design preview only."""
from pathlib import Path
import math,json
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial import ConvexHull
from make_comparison import S,BG,INK,TEAL,MUTED,poly,line,ellipse,lens,text,paste_fit
P=Path(__file__).resolve().parent

def refined_compact():
 im=Image.new('RGBA',(240*S,190*S));d=ImageDraw.Draw(im)
 w,h,r=146,96,10
 outline=[]
 for cx,cy,start in [(w-r,r,-90),(w-r,h-r,0),(r,h-r,90),(r,r,180)]:
  for i in range(17):
   a=math.radians(start+90*i/16);outline.append([cx+r*math.cos(a),cy+r*math.sin(a)])
 local=np.array(outline);front=local@np.array([[1,.14],[0,1]])+np.array([28,49]);back=front+[29,-18]
 both=np.vstack([front,back]);poly(d,both[ConvexHull(both).vertices],'#304a59')
 # Every side shares its exact boundary with the rounded front. No floating
 # roof, separate border frame or independently transformed front panel.
 top=np.array([110,135,147]);side=np.array([48,74,89])
 for j in range(len(local)):
  k=(j+1)%len(local);edge=local[k]-local[j];normal=np.array([edge[1],-edge[0]])
  weight=max(0.,-normal[1])/(abs(normal).sum()+1e-12)
  color=tuple(np.rint(side*(1-weight)+top*weight).astype(int))
  poly(d,[front[j],back[j],back[k],front[k]],color)
 poly(d,front,'#405966')
 # Optical details only: the front housing has no inset contour that could
 # be mistaken for a loose plate. The top and side are uninterrupted.
 details=Image.new('RGBA',im.size);q=ImageDraw.Draw(details)
 lens(q,47,h/2,37)
 q.rounded_rectangle((101*S,25*S,130*S,52*S),radius=3*S,fill='#293e49')
 line(q,[(106,45),(125,45)],'#7293a0',1.5)
 ellipse(q,(119,69,124,74),'#67c7b8')
 details=details.transform(im.size,Image.Transform.AFFINE,(1,0,-28*S,-.14,1,(-49+.14*28)*S),resample=Image.Resampling.BICUBIC)
 im.alpha_composite(details)
 return im

def main():
 camera=refined_compact();camera.save(P/'camera_A_refined.png')
 (P/'camera_A_refined_geometry.json').write_text(json.dumps({'canvas_px':list(camera.size),'lens_center_px':[(28+47)*S,(49+48+.14*47)*S],'design':'Approved rounded compact camera A','generator':'refine_compact.py'},indent=2)+'\n')
 sheet=Image.new('RGBA',(1040*S,650*S),BG);d=ImageDraw.Draw(sheet)
 text(d,46,28,'CAMERA A · REFINED',18,TEAL,True)
 text(d,46,63,'One continuous, rounded body',32,INK,True)
 paste_fit(sheet,camera,390,162,420)
 text(d,810,243,'At slide size',18,MUTED,False,'mt')
 paste_fit(sheet,camera,810,287,86)
 text(d,520,550,'Clean joins. No protrusion. Same lens and palette.',21,MUTED,False,'mt')
 sheet.convert('RGB').resize((1040,650),Image.Resampling.LANCZOS).save(P/'camera_A_refined_preview.png')
 print(P/'camera_A_refined_preview.png')
if __name__=='__main__':main()
