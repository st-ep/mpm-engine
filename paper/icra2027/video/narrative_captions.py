"""Conference-style narration, timed against the approved three-section reveal.

Text is shared by burned-in subtitles, SRT export, and the editable transcript.
No synthesized narration track is generated.
"""
from functools import lru_cache
from PIL import ImageDraw, ImageFont

SECTIONS=[('opening',13),('observe',24),('balance',28)]
ALL_SECTIONS=SECTIONS+[
    ('insertion',15),('golf',15),('simshape',23),
    ('hardware',30),('pouring',36),('takeaway',10),
]
CUES={
    'opening':[
        (0,3,'The same action can produce very different outcomes.'),
        (3,6,'We start by observing one interaction.'),
        (6,10,'FORM identifies a material law from motion and contact forces.'),
        (10,13,'Then we use that law to plan new actions.'),
    ],
    'observe':[
        (0,4,'First, we record the material as it deforms.'),
        (4,8,'For RGB-D, we reconstruct the 3D shape, assuming symmetry.'),
        (8,12,'A volume-preserving flow model then moves particles through its interior.'),
        (12,16,'Stereo matches texture across two views to recover 3D surface motion.'),
        (16,20,'We fit a smooth motion model to infer how the interior moves.'),
        (20,24,'Finally, we combine the inferred motion with measured contact forces.'),
    ],
    'balance':[
        (0,4,'Which material law explains the motion we just observed?'),
        (4,8,'We weight Newton’s law and integrate over space and time.'),
        (8,12,'The resulting weak balance links stress to motion and contact forces.'),
        (12,16,'We express stress using known responses and unknown coefficients.'),
        (16,20,'Then we recover those coefficients with linear least squares.'),
        (20,24,'Keeping the law fixed, we plan new robot actions in simulation.'),
        (24,28,'Finally, we execute that plan on the real material.'),
    ],
    'insertion':[
        (0,4,'From bending and force, we identify each material’s stiffness.'),
        (4,8,'Our task is rod insertion: a new action and geometry.'),
        (8,12,'Matched ID uses the rod’s own model; swapped ID uses the other material’s.'),
        (12,15,'Matched plans insert; swapped plans hit the wall.'),
    ],
    'golf':[
        (0,4,'We reuse the material laws identified from bending to plan putting.'),
        (4,7.5,'Our goal is to stop the ball inside the target.'),
        (7.5,12.8,'We plan forward-stroke duration and aim angle, keeping the backswing fixed.'),
        (12.8,15,'Matched plans succeed; swapped plans miss.'),
    ],
    'simshape':[
        (0,4,'From pressing, we identify stiffness and yield stress for each material.'),
        (4,8,'The stiffnesses are similar, but the yield stresses differ roughly tenfold.'),
        (8,12,'Our task is to shape a larger block into an X with six planned pinches.'),
        (12,16,'We plan the pinch openings and a shared lateral offset.'),
        (16,20,'The identified material law stays fixed throughout planning.'),
        (20,23,'Both matched plans give lower surface error.'),
    ],
    'hardware':[
        (0,6,'For each material, one press identifies stiffness and yield stress.'),
        (6,9,'We use those laws to predict a separate, lower-force press.'),
        (9,12,'The models capture compression, but spreading differs.'),
        (12,16,'Next, we plan four pinches to shape fresh specimens into an X.'),
        (16,21,'Here, the robot executes the Play-Doh plan without online correction.'),
        (21,26,'The identified laws stay fixed throughout planning and execution.'),
        (26,30,'After rigid alignment, footprint overlap is 72 to 78 percent.'),
    ],
    'pouring':[
        (0,4,'We begin with one 60-degree glycerol pour.'),
        (4,8,'A reduced weak balance identifies an effective viscosity of 3.44 pascal-seconds.'),
        (8,12,'The identified viscosity stays fixed as we plan new target volumes.'),
        (12,16,'For each target volume, MPM selects the cup’s tilt angle.'),
        (16,20,'We send that angle to the robot and repeat the pour five times.'),
        (20,25,'Each point shows the measured mean, with its standard deviation.'),
        (25,31,'Across all six targets, the largest mean error is 3.8 milliliters.'),
        (31,36,'Water uses the same commands, with one trial per target, and overpours.'),
    ],
    'takeaway':[
        (0,5,'FORM turns an observed interaction into an explicit material law.'),
        (5,10,'That same law guides new robot actions, without refitting for each task.'),
    ],
}


def caption_at(section,t):
    for start,end,caption in CUES[section]:
        if start<=t<end:return caption
    return CUES[section][-1][2] if t>=CUES[section][-1][1] else ''


@lru_cache(None)
def subtitle_font(size):
    return ImageFont.truetype('/usr/share/fonts/truetype/lato/Lato-Regular.ttf',size)


def paint_caption(image,caption):
    """Readable one-line subtitles in reserved space below the diagrams."""
    if not caption:return image
    scale=image.width/1280
    draw=ImageDraw.Draw(image)
    font=subtitle_font(round(24*scale))
    width=draw.textlength(caption,font=font)
    assert width<=1192*scale,caption
    center=image.width/2
    draw.rounded_rectangle((center-width/2-14*scale,665*scale,center+width/2+14*scale,710*scale),radius=5*scale,fill='#26353e')
    draw.text((center,687.5*scale),caption,font=font,fill='#ffffff',anchor='mm')
    return image


def timestamp(t):
    ms=round(t*1000)
    return f'{ms//3600000:02d}:{ms//60000%60:02d}:{ms//1000%60:02d},{ms%1000:03d}'


def export_captions(folder,sections=SECTIONS,stem='first_three_slides'):
    offset=0;subtitles=[];transcript=[];index=1
    for section,duration in sections:
        transcript.append(section.upper())
        for start,end,caption in CUES[section]:
            subtitles.append(f'{index}\n{timestamp(offset+start)} --> {timestamp(offset+end)}\n{caption}\n')
            transcript.append(f'{offset+start:05.1f}–{offset+end:05.1f}  {caption}')
            index+=1
        offset+=duration;transcript.append('')
    (folder/f'{stem}.srt').write_text('\n'.join(subtitles),encoding='utf-8')
    (folder/f'{stem}_transcript.txt').write_text('\n'.join(transcript)+'\n',encoding='utf-8')
