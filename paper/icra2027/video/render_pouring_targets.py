"""Render frozen target-pour states with the existing paper scene's style.

Run in Blender with --background --python; generated meshes arrive from the
independent preparation process. The original .blend file is never saved over.
"""
from pathlib import Path
import argparse
import json
import sys
import time

import bpy
from mathutils import Matrix, Vector
from bpy_extras.object_utils import world_to_camera_view

ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--indices", type=int, nargs="+")
parser.add_argument("--device-index", type=int, default=1)
args = parser.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])
OUT = args.output.resolve()
INFO = json.loads((OUT / "scene.json").read_text())
bpy.ops.wm.open_mainfile(filepath=INFO["paper_scene"])
scene = bpy.context.scene
camera = scene.camera
T = Matrix(INFO["extrinsics"])
camera.matrix_world = T @ Matrix.Diagonal((1., -1., -1., 1.))
intrinsics = INFO["camera"]["intrinsics"]
width, height = INFO["camera"]["width"], INFO["camera"]["height"]
camera.data.sensor_fit = "HORIZONTAL"
camera.data.sensor_width = 36.
camera.data.lens = intrinsics["fx"]*36/width
camera.data.shift_x = (width/2-intrinsics["ppx"])/width
camera.data.shift_y = (intrinsics["ppy"]-height/2)/width*intrinsics["fx"]/intrinsics["fy"]
scene.render.resolution_x = round(width*INFO["render_scale"])
scene.render.resolution_y = round(height*INFO["render_scale"])
scene.render.resolution_percentage = 100
scene.render.pixel_aspect_x = intrinsics["fy"]/intrinsics["fx"]
scene.render.pixel_aspect_y = 1.
errors = []
for point in [(.34, .01, -.07), (.42, 0., .06), (.4, .02, .14), (0., 0., .2)]:
    cv = T.inverted() @ Vector((*point, 1.))
    expected = Vector((intrinsics["fx"]*cv.x/cv.z+intrinsics["ppx"],
                       intrinsics["fy"]*cv.y/cv.z+intrinsics["ppy"]))
    uv = world_to_camera_view(scene, camera, Vector(point))
    actual = Vector((uv.x*width, (1-uv.y)*height))
    errors.append((expected-actual).length)
assert max(errors) < .001, errors
# Crop empty tabletop below the receiver. This is the same fixed crop as the
# hardware panel, expressed in the unrotated recorded camera coordinates.
x0, y0, x1, y1 = INFO["upright_crop_xyxy"]
scene.render.use_border = True
scene.render.use_crop_to_border = True
scene.render.border_min_x = (width-y1)/width
scene.render.border_max_x = (width-y0)/width
scene.render.border_min_y = 1-x1/height
scene.render.border_max_y = 1-x0/height
scene.render.engine = "CYCLES"
preferences = bpy.context.preferences.addons["cycles"].preferences
preferences.compute_device_type = "OPTIX"
preferences.get_devices()
devices = [d for d in preferences.devices if d.type == "OPTIX"]
assert devices, "OptiX device required for this animation render"
selected = devices[min(args.device_index, len(devices)-1)]
for device in preferences.devices:
    device.use = device == selected
scene.cycles.device = "GPU"
scene.cycles.samples = 64
scene.cycles.use_denoising = True
scene.cycles.denoiser = "OPTIX"
scene.cycles.seed = 0
scene.cycles.use_animated_seed = False
scene.render.use_persistent_data = True
scene.render.threads_mode = "FIXED"
scene.render.threads = 8
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
scene.render.image_settings.color_depth = "8"
liquid_material = bpy.data.objects["liquid"].data.materials[0]
(OUT / "render_settings.json").write_text(json.dumps(dict(
    engine="Blender Cycles", version=bpy.app.version_string, samples=64,
    device=selected.name, device_id=selected.id, camera="Recorded static side camera",
    maximum_projection_error_native_pixels=max(errors),
    upright_crop_xyxy=INFO["upright_crop_xyxy"], render_scale=INFO["render_scale"],
    appearance="Existing paper scene materials, lighting, cup normals, and blue liquid",
    particle_positions_modified=False), indent=2)+"\n")

for index in args.indices if args.indices is not None else range(INFO["frame_count"]):
    destination = OUT / f"rendered/frame_{index:04d}.png"
    if destination.exists():
        continue
    manifest = OUT / f"frames/frame_{index:04d}.json"
    while not manifest.exists():
        time.sleep(.5)
    info = json.loads(manifest.read_text())
    started = time.monotonic()
    for name, matrix in info["poses"].items():
        bpy.data.objects[name].matrix_world = Matrix(matrix)
    previous = bpy.data.objects.get("liquid")
    if previous is not None:
        mesh = previous.data
        bpy.data.objects.remove(previous, do_unlink=True)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    bpy.ops.wm.ply_import(filepath=info["mesh"])
    liquid = bpy.context.object
    liquid.name = "liquid"
    liquid.data.materials.append(liquid_material)
    for face in liquid.data.polygons:
        face.use_smooth = True
    temporary = destination.with_suffix(".pending.png")
    scene.render.filepath = str(temporary)
    bpy.ops.render.render(write_still=True)
    temporary.rename(destination)
    print(f"VIDEO_FRAME {index+1}/{INFO['frame_count']} seconds={time.monotonic()-started:.3f}", flush=True)

if all((OUT / f"rendered/frame_{i:04d}.png").exists() for i in range(INFO["frame_count"])):
    (OUT / "render_complete.json").write_text(json.dumps(dict(status="complete", frames=INFO["frame_count"]), indent=2)+"\n")
