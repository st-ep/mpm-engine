"""Standalone camera design study. Does not modify slide/video assets."""
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import math
P=Path(__file__).resolve().parent;S=4
BG='#f4f6f7';INK='#21333e';TEAL='#167b76';MUTED='#566975'
def poly(d,points,fill):d.polygon([(x*S,y*S) for x,y in points],fill=fill)
def line(d,points,fill,width=1):d.line([(x*S,y*S) for x,y in points],fill=fill,width=round(width*S),joint='curve')
def ellipse(d,box,fill,outline=None,width=1):d.ellipse(tuple(v*S for v in box),fill=fill,outline=outline,width=round(width*S))
def lens(d,cx,cy,r):
 for frac,col in [(1,'#182b35'),(.91,'#a1b4b9'),(.85,'#304a55'),(.73,'#101e29'),(.62,'#355c6d'),(.56,'#153847')]:
  a=r*frac;ellipse(d,(cx-a,cy-a,cx+a,cy+a),col)
 # Quiet glass gradient; no decorative lights or eye-like pupil.
 for j in range(28,0,-1):
  a=r*.52*j/28;v=j/28;col=(int(19+11*(1-v)),int(48+39*(1-v)),int(64+41*(1-v)))
  ellipse(d,(cx-a,cy-a,cx+a,cy+a),col)
 d.arc(((cx-r*.7)*S,(cy-r*.7)*S,(cx+r*.7)*S,(cy+r*.7)*S),205,290,fill='#bacdd1',width=2*S)
 d.arc(((cx-r*.55)*S,(cy-r*.55)*S,(cx+r*.55)*S,(cy+r*.55)*S),25,95,fill='#52889c',width=S)
 ellipse(d,(cx-r*.34,cy-r*.39,cx-r*.05,cy-r*.19),'#729ba6')
def face(im,xy,w,h,color,kind='compact'):
 local=Image.new('RGBA',im.size);d=ImageDraw.Draw(local)
 d.rounded_rectangle((0,0,w*S,h*S),radius=9*S,fill=color)
 d.rounded_rectangle((3*S,3*S,(w-3)*S,(h-3)*S),radius=7*S,outline='#536975',width=S)
 if kind=='compact':
  lens(d,46,h/2,36)
  d.rounded_rectangle((94*S,21*S,125*S,51*S),radius=3*S,fill='#243843',outline='#738893',width=S)
  line(d,[(100,40),(119,40)],'#698b98',2)
  ellipse(d,(113,64,118,69),'#5bc9ba')
 else:
  lens(d,w/2,h/2,36)
  for x,y in [(10,10),(w-10,10),(10,h-10),(w-10,h-10)]:
   ellipse(d,(x-2,y-2,x+2,y+2),'#526872');line(d,[(x-1,y),(x+1,y)],'#b2c4ca',.7)
 x,y=xy
 transformed=local.transform(im.size,Image.Transform.AFFINE,(1,0,-x*S,-.14,1,(-y+.14*x)*S),resample=Image.Resampling.BICUBIC)
 im.alpha_composite(transformed)
def compact():
 im=Image.new('RGBA',(240*S,190*S));d=ImageDraw.Draw(im)
 poly(d,[(26,45),(61,26),(201,45.6),(166,64.6)],'#6e8793')
 poly(d,[(166,64.6),(201,45.6),(201,137.6),(166,156.6)],'#304a59')
 line(d,[(62,28),(198,47)],'#a5b8bf',1)
 face(im,(26,45),140,92,'#405966')
 d=ImageDraw.Draw(im);poly(d,[(94,30),(104,24),(130,27.6),(120,34)],'#334a55')
 line(d,[(180,74),(191,68),(191,120),(180,126),(180,74)],'#607d8b',1)
 return im

def industrial():
 im=Image.new('RGBA',(240*S,190*S));d=ImageDraw.Draw(im)
 poly(d,[(39,43),(100,17),(202,31.3),(141,57.3)],'#aec0c6')
 poly(d,[(141,57.3),(202,31.3),(202,125.3),(141,151.3)],'#66808d')
 for x in [151,161,171,181,191]:
  y=57.3-(x-141)*26/61
  line(d,[(x,y+4),(x,y+89)],'#3d5a69',2)
  line(d,[(x+2,y+3),(x+2,y+87)],'#91a7b0',1)
 face(im,(39,43),102,94,'#8ea3ad','industrial')
 d=ImageDraw.Draw(im);ellipse(d,(183,53,188,58),'#5bc9ba')
 return im

def cinema():
 im=Image.new('RGBA',(240*S,190*S));d=ImageDraw.Draw(im)
 # Twin film reels above a compact, left-facing camera body.
 poly(d,[(66,85),(97,69),(194,83),(163,99)],'#778d99')
 poly(d,[(163,99),(194,83),(194,146),(163,163)],'#304958')
 poly(d,[(66,85),(163,99),(163,163),(66,149)],'#496270')
 for cx,cy in [(95,52),(152,61)]:
  ellipse(d,(cx-30,cy-30,cx+30,cy+30),'#263d4b')
  ellipse(d,(cx-26,cy-26,cx+26,cy+26),'#91a5ad')
  ellipse(d,(cx-22,cy-22,cx+22,cy+22),'#617d8b')
  for j in range(5):
   a=j*math.tau/5-.8;x=cx+14*math.cos(a);y=cy+14*math.sin(a)
   ellipse(d,(x-5,y-5,x+5,y+5),'#2a4555')
  ellipse(d,(cx-4,cy-4,cx+4,cy+4),'#bdcbd0')
 poly(d,[(66,103),(38,94),(25,100),(25,141),(38,149),(66,134)],'#233b49')
 line(d,[(38,97),(38,146)],'#7e99a5',2)
 ellipse(d,(17,99,35,142),'#101f2b')
 ellipse(d,(21,105,30,136),'#3c6e85')
 d.rounded_rectangle((90*S,107*S,143*S,138*S),radius=4*S,fill='#344d5b',outline='#7f98a4',width=S)
 ellipse(d,(150,115,155,120),'#5bc9ba')
 return im

def font(size,bold=False):return ImageFont.truetype('/usr/share/fonts/truetype/lato/Lato-'+('Bold' if bold else 'Regular')+'.ttf',size*S)
def text(d,x,y,s,size=22,color=INK,bold=False,anchor=None):d.text((x*S,y*S),s,font=font(size,bold),fill=color,anchor=anchor)
def paste_fit(out,asset,cx,top,width):
 # Tight alpha crop gives all candidates equal apparent width.
 asset=asset.crop(asset.getbbox());h=round(asset.height*width*S/asset.width)
 out.alpha_composite(asset.resize((round(width*S),h),Image.Resampling.LANCZOS),(round((cx-width/2)*S),round(top*S)))
 return h/S

def main():
 out=Image.new('RGBA',(1440*S,710*S),BG);d=ImageDraw.Draw(out)
 text(d,50,28,'CAMERA DESIGN STUDY',18,TEAL,True)
 text(d,50,64,'Three visual directions',37,INK,True)
 text(d,50,115,'Same palette. Enlarged above; actual slide size below.',21,MUTED)
 for x in [486,954]:line(d,[(x,180),(x,675)],'#d9e1e5',1)
 items=[('A','Compact camera','Clean, modern, close to the current design.',compact()),('B','Machine-vision camera','Technical, closer to a laboratory sensor.',industrial()),('C','Cinema camera','Instantly recognizable, more theatrical.',cinema())]
 for i,(letter,title,desc,asset) in enumerate(items):
  cx=252+i*468
  asset.save(P/f'camera_{letter}.png')
  text(d,cx,179,f'{letter}  {title}',24,TEAL,True,'mt')
  paste_fit(out,asset,cx,239,270)
  text(d,cx,470,'At slide size',16,MUTED,False,'mt')
  paste_fit(out,asset,cx,503,86)
  text(d,cx,610,desc,18,MUTED,False,'mt')
  if letter=='A':text(d,cx,647,'Recommended',18,TEAL,True,'mt')
 out.convert('RGB').resize((1440,710),Image.Resampling.LANCZOS).save(P/'camera_design_comparison.png')
 print(P/'camera_design_comparison.png')
if __name__=='__main__':main()
