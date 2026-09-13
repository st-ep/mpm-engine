"""Full-height side-camera comparison, side by side or stacked for a Story.

Reuse all captured particle states and their documented private-video release.
No simulation, model change, or temporal interpolation is involved.
"""
from pathlib import Path
import argparse
import json
import subprocess

from PIL import Image, ImageDraw, ImageFont
from experiments.pour import pour_share_video_compose as original

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out/pour_share_video_side_by_side_20260911"
original.OUT = OUT


def make_frame(index, story=False):
    size = (1080,1920) if story else (1536,934)
    panel = (800,958) if story else (720,862)
    # Positions are ordered Real, Simulation; the Story places Simulation first.
    positions = [(140,958),(140,0)] if story else [(36,36),(780,36)]
    background = (247,248,250)
    canvas = Image.new("RGB",size,background)
    with Image.open(OUT/f"hardware/frame_{index:04d}.png") as real:
        assert real.size == (480,600)
        image = real.convert("RGB").crop((0,0,480,575))
        canvas.paste(image.resize(panel,Image.Resampling.LANCZOS),positions[0])
    with Image.open(OUT/f"rendered/frame_{index:04d}.png") as sim:
        upright = sim.convert("RGB").transpose(Image.Transpose.ROTATE_90)
        assert abs(upright.width/upright.height-.8)<.002,upright.size
        upright = upright.crop((0,0,upright.width,round(upright.height*575/600)))
        canvas.paste(upright.resize(panel,Image.Resampling.LANCZOS),positions[1])
    overlay = Image.new("RGBA",size,(0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.truetype(original.BOLD,28 if story else 25)
    for (x,y),label in zip(positions,["Real","Simulation"],strict=True):
        x, y = x+16, y+(84 if story else 16)
        bounds = draw.textbbox((0,0),label,font=font)
        width, height = bounds[2]-bounds[0], bounds[3]-bounds[1]
        draw.rounded_rectangle((x,y,x+width+22,y+height+18),radius=7,fill=(17,23,30,178))
        draw.text((x+11-bounds[0],y+9-bounds[1]),label,font=font,fill=(255,255,255,255))
    return Image.alpha_composite(canvas.convert("RGBA"),overlay).convert("RGB")


def compose(story=False):
    verification = OUT/"capture/verification.json"
    if not json.loads(verification.read_text())["accepted"]:
        release = json.loads((OUT/"capture/display_release.json").read_text())
        assert release["accepted_for_private_visualization"]
        assert release["verification_sha256"] == original.digest(verification)
    scene = json.loads((OUT/"scene.json").read_text())
    assert json.loads((OUT/"render_complete.json").read_text())["frames"] == scene["frame_count"]
    size = (1080,1920) if story else (1536,934)
    filename = "glycerol_60deg_story_full_view.mp4" if story else "glycerol_60deg_side_by_side.mp4"
    destination = OUT/filename
    command = ["ffmpeg","-v","error","-y","-f","rawvideo","-pix_fmt","rgb24",
               "-s",f"{size[0]}x{size[1]}","-r","30","-i","-","-an","-c:v","libx264",
               "-preset","medium","-crf","18","-pix_fmt","yuv420p","-movflags","+faststart",
               "-metadata","title=Glycerol pouring: real experiment and MPM",str(destination)]
    process = subprocess.Popen(command,stdin=subprocess.PIPE)
    try:
        for index in range(scene["frame_count"]):
            process.stdin.write(make_frame(index,story).tobytes())
    finally:
        process.stdin.close()
    assert process.wait() == 0
    make_frame(249,story).save(OUT/("story_poster.png" if story else "poster.png"))
    info = dict(output=str(destination.relative_to(ROOT)),sha256=original.digest(destination),
        width=size[0],height=size[1],fps=30,frames=scene["frame_count"],duration_s=scene["frame_count"]/30,
        layout="Simulation above; Real below" if story else "Real left; Simulation right",
        display_panel_size=[800,958] if story else [720,862],
        panel_positions={"Simulation":[140,0],"Real":[140,958]} if story else {"Real":[36,36],"Simulation":[780,36]},
        labels=["Simulation","Real"] if story else ["Real","Simulation"],label_position="Small translucent badges in the upper-left area of each view, clear of the arm and cups",
        camera="Original recorded side camera",upright_crop_xyxy=[0,0,480,575],
        render_crop_xyxy=scene["upright_crop_xyxy"],
        framing="Full recorded top and width; bottom trimmed just below the receiver at y=575",
        capture="out/pour_share_video_20260911/capture",new_simulation=False,
        particle_positions_changed=False,liquid_meshes_changed=False,playback_speed=1.,
        synchronization="synchronization.json",strict_replay_check_passed=False,
        private_visualization_release="capture/display_release.json",
        scripts={str(p.relative_to(ROOT)):original.digest(p) for p in [Path(__file__),
            ROOT/"experiments/pour/pour_share_video_blender.py",ROOT/"experiments/pour/pour_share_video_compose.py"]})
    (OUT/("story_provenance.json" if story else "video_provenance.json")).write_text(json.dumps(info,indent=2)+"\n")
    print(destination,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-hardware",action="store_true")
    parser.add_argument("--preview",type=int)
    parser.add_argument("--story",action="store_true")
    args = parser.parse_args()
    if args.prepare_hardware:
        original.prepare_hardware()
    elif args.preview is not None:
        make_frame(args.preview,args.story).save(OUT/f"{'story_' if args.story else ''}preview_{args.preview:04d}.png")
    else:
        compose(args.story)
