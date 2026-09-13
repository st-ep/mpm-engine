"""Synchronize recorded camera frames and assemble a private landscape MP4."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out/pour_share_video_20260911"
RAW = ROOT / "pouring_real_data/09-04-60-2s"
SIZE = (1536, 1024)
PANEL = (720, 916)
FPS = 30
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_hardware():
    scene = json.loads((OUT / "scene.json").read_text())
    camera = scene["camera_name"]
    timestamp_path = RAW / f"frames_{camera}.jsonl"
    video_path = RAW / f"{camera}_rgb.mp4"
    rows = [json.loads(line) for line in timestamp_path.read_text().splitlines()]
    times = np.array([r["t_host"] for r in rows])
    paired = []
    for index in range(scene["frame_count"]):
        t = (index+1)/FPS
        host = scene["host_start_s"]+t
        nearest = int(np.argmin(abs(times-host)))
        row = rows[nearest]
        delta = float(row["t_host"]-host)
        assert abs(delta) <= 1/60, (index, delta)
        paired.append(dict(index=index, simulation_t_s=t, simulation_host_s=host,
            hardware_frame=int(row["frame_idx"]), hardware_host_s=row["t_host"],
            hardware_minus_simulation_s=delta))
    assert all(b["hardware_frame"] > a["hardware_frame"] for a, b in zip(paired[:-1], paired[1:]))
    with (OUT / "frame_pairing.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    target = OUT / "hardware"
    target.mkdir(exist_ok=True)
    selection = "+".join(f"eq(n\\,{r['hardware_frame']})" for r in paired)
    x0, y0, x1, y1 = scene["upright_crop_xyxy"]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video_path),
        "-vf", f"select={selection},transpose=2,crop={x1-x0}:{y1-y0}:{x0}:{y0}", "-vsync", "0",
        "-start_number", "0", str(target / "frame_%04d.png")], check=True)
    assert len(list(target.glob("frame_*.png"))) == scene["frame_count"]
    (OUT / "synchronization.json").write_text(json.dumps(dict(
        camera=camera, camera_mount="static", episode=RAW.name,
        hardware_video_sha256=digest(video_path),
        timestamps_sha256=digest(timestamp_path),
        pairing_sha256=digest(OUT / "frame_pairing.csv"), frames=len(paired),
        maximum_absolute_time_difference_s=max(abs(r["hardware_minus_simulation_s"]) for r in paired),
        synchronization="Nearest recorded host timestamp for each 30 Hz simulation display state",
        orientation="90 degrees counterclockwise", upright_crop_xyxy=scene["upright_crop_xyxy"],
        playback_speed=1., temporal_interpolation=False), indent=2)+"\n")


def background():
    canvas = Image.new("RGB", SIZE, (247, 248, 250))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(BOLD, 25)
    small = ImageFont.truetype(FONT, 19)
    draw.text((36, 25), "Real experiment", fill=(45, 53, 61), font=font)
    draw.text((780, 25), "MPM simulation", fill=(29, 101, 148), font=font)
    draw.text((768, 1006), "Glycerol · 60° pour", fill=(86, 96, 108), font=small, anchor="mm")
    return canvas


def frame(index):
    canvas = background()
    with Image.open(OUT / f"hardware/frame_{index:04d}.png") as hardware:
        canvas.paste(hardware.convert("RGB").resize(PANEL, Image.Resampling.LANCZOS), (36,72))
    with Image.open(OUT / f"rendered/frame_{index:04d}.png") as simulation:
        upright = simulation.convert("RGB").transpose(Image.Transpose.ROTATE_90)
        assert abs(upright.width/upright.height-440/560) < .002, upright.size
        canvas.paste(upright.resize(PANEL, Image.Resampling.LANCZOS), (780,72))
    return canvas


def story_frame(index):
    """Phone layout: stack the cups, with space for Instagram's UI at each end."""
    canvas = Image.new("RGB", (1080,1920), (247,248,250))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(BOLD, 29)
    small = ImageFont.truetype(FONT, 25)
    panel = (740,732)
    draw.text((170,157), "Real experiment", fill=(45,53,61), font=font)
    draw.text((170,967), "MPM simulation", fill=(29,101,148), font=font)
    draw.text((540,1810), "Glycerol · 60° pour", fill=(86,96,108), font=small, anchor="mm")
    with Image.open(OUT / f"hardware/frame_{index:04d}.png") as hardware:
        # The story prioritizes the cups; the landscape version retains more arm.
        image = hardware.convert("RGB").crop((0,125,440,560))
        canvas.paste(image.resize(panel,Image.Resampling.LANCZOS),(170,200))
    with Image.open(OUT / f"rendered/frame_{index:04d}.png") as simulation:
        image = simulation.convert("RGB").transpose(Image.Transpose.ROTATE_90)
        image = image.crop((0,round(image.height*125/560),image.width,image.height))
        canvas.paste(image.resize(panel,Image.Resampling.LANCZOS),(170,1010))
    return canvas


def compose(story=False):
    verification = OUT / "capture/verification.json"
    if not json.loads(verification.read_text())["accepted"]:
        # A failed numerical screen stays failed. A separate, explicit review
        # may release this unchanged replay solely for the private visual clip.
        release = json.loads((OUT / "capture/display_release.json").read_text())
        assert release["accepted_for_private_visualization"]
        assert release["verification_sha256"] == digest(verification)
    scene = json.loads((OUT / "scene.json").read_text())
    assert json.loads((OUT / "render_complete.json").read_text())["frames"] == scene["frame_count"]
    destination = OUT / ("glycerol_60deg_instagram_story.mp4" if story else "glycerol_60deg_real_and_mpm.mp4")
    size = (1080,1920) if story else SIZE
    make_frame = story_frame if story else frame
    command = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{size[0]}x{size[1]}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-metadata", "title=Glycerol pouring: real experiment and MPM", str(destination)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for index in range(scene["frame_count"]):
            process.stdin.write(make_frame(index).tobytes())
    finally:
        process.stdin.close()
    assert process.wait() == 0
    make_frame(249).save(OUT / ("story_poster.png" if story else "poster.png"))
    (OUT / ("story_provenance.json" if story else "video_provenance.json")).write_text(json.dumps(dict(
        output=str(destination.relative_to(ROOT)), video_sha256=digest(destination),
        width=size[0], height=size[1], fps=FPS, frames=scene["frame_count"],
        layout="Vertical 9:16; hardware above MPM" if story else "Landscape; hardware left, MPM right",
        upright_crop_xyxy=[20,150,460,585] if story else scene["upright_crop_xyxy"],
        duration_s=scene["frame_count"]/FPS, audio=False, playback_speed=1.,
        purpose="Private video for the user to share; not a manuscript asset",
        recording=RAW.name, commanded_angle_deg=60, camera="side (static)",
        model="Frozen September 8 MPM replay used for the paper identification-pour still",
        renderer="Blender Cycles, original paper lighting/materials, 64 samples, blue display liquid",
        scientific_scope="Same identification/calibration recording; not held-out validation",
        simulation_verification="capture/verification.json", synchronization="synchronization.json",
        private_visualization_review="capture/display_release.json",
        surface_audits="frames/frame_*.json", scripts={p.name:digest(p) for p in
            sorted((ROOT / "experiments/pour").glob("pour_share_video_*.py"))}), indent=2)+"\n")
    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-hardware", action="store_true")
    parser.add_argument("--preview", type=int)
    parser.add_argument("--story", action="store_true")
    args = parser.parse_args()
    if args.prepare_hardware:
        prepare_hardware()
    elif args.preview is not None:
        make_frame = story_frame if args.story else frame
        make_frame(args.preview).save(OUT / f"{'story_' if args.story else ''}preview_{args.preview:04d}.png")
    else:
        compose(args.story)
