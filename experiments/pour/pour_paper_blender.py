"""Render the prepared MPM scene; invoke with Blender --background --python."""
from pathlib import Path
import json
import hashlib
import math
import os

import bpy
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out/pour_figure_final_20260911"
ASSETS = OUT / "render_assets"
INFO = json.loads((ASSETS / "scene.json").read_text())
SCENE_SHA = hashlib.sha256((ASSETS / "scene.json").read_bytes()).hexdigest()

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)


def material(name, color, roughness=.3, transmission=0., ior=1.45):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color[:3], 1.)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Transmission Weight"].default_value = transmission
    bsdf.inputs["IOR"].default_value = ior
    return mat


cup_mat = material("Clear measuring cup", (.92, .96, 1.), .10, 1., 1.49)
liquid_mat = material("MPM liquid display blue", (.025, .16, .29), .25, 0., 1.45)
for item in INFO["objects"]:
    bpy.ops.wm.ply_import(filepath=str(ASSETS / item["file"]))
    obj = bpy.context.object
    obj.name = item["name"]
    for face in obj.data.polygons:
        face.use_smooth = True
    if item["material"] == "cup":
        # Smooth the curved walls while preserving actual sharp floor/rim
        # boundaries. Averaging normals across these edges distorted refraction.
        split = obj.modifiers.new("Measured cup sharp edges", "EDGE_SPLIT")
        split.split_angle = math.radians(30.)
        split.use_edge_angle = True
        split.use_edge_sharp = False
    mat = (cup_mat if item["material"] == "cup" else liquid_mat if item["material"] == "liquid"
           else material(item["name"], item["rgba"], .32))
    obj.data.materials.append(mat)

# The neutral tabletop is at the frozen physical table height.
bpy.ops.mesh.primitive_plane_add(size=200, location=(.35, 0, INFO["table_z"]-.0001))
bpy.context.object.name = "Table"
bpy.context.object.data.materials.append(material("Table", (.19, .215, .23), .58))

cam_data = bpy.data.cameras.new("Recorded side camera")
cam = bpy.data.objects.new("Recorded side camera", cam_data)
bpy.context.collection.objects.link(cam)
T = Matrix(INFO["extrinsics"])
cam.matrix_world = T @ Matrix.Diagonal((1., -1., -1., 1.))
k = INFO["camera"]["intrinsics"]
w, h = INFO["camera"]["width"], INFO["camera"]["height"]
cam_data.type = "PERSP"
cam_data.sensor_fit = "HORIZONTAL"
cam_data.sensor_width = 36.
cam_data.lens = k["fx"] * cam_data.sensor_width / w
cam_data.shift_x = (w/2-k["ppx"])/w
cam_data.shift_y = (k["ppy"]-h/2)/w * k["fx"]/k["fy"]
cam_data.clip_start = .005
cam_data.clip_end = 200
scene = bpy.context.scene
scene.camera = cam
scene.render.resolution_x = w*INFO["render_scale"]
scene.render.resolution_y = h*INFO["render_scale"]
scene.render.resolution_percentage = 100
scene.render.pixel_aspect_x = k["fy"]/k["fx"]
scene.render.pixel_aspect_y = 1.

# Check the camera convention against direct OpenCV projection at scene points.
from bpy_extras.object_utils import world_to_camera_view
projection_errors = []
for point in [( .34, .01, -.07), (.42, 0., .06), (.40, .02, .14), (.30, -.02, .02)]:
    cv = T.inverted() @ Vector((*point, 1.))
    expected = Vector((k["fx"]*cv.x/cv.z+k["ppx"], k["fy"]*cv.y/cv.z+k["ppy"]))
    uv = world_to_camera_view(scene, cam, Vector(point))
    actual = Vector((uv.x*w, (1-uv.y)*h))
    projection_errors.append((expected-actual).length)
assert max(projection_errors) < .001, projection_errors
(OUT / "camera_verification.json").write_text(json.dumps(dict(
    maximum_projection_difference_native_pixels=max(projection_errors),
    checks=len(projection_errors), tolerance_pixels=.001,
    camera="Recorded side-camera intrinsics and full extrinsic rotation, including roll",
    passed=True), indent=2)+"\n")


def light(name, location, power, size, target=(.37, .01, .035)):
    data = bpy.data.lights.new(name, "AREA")
    data.energy = power
    data.shape = "DISK"
    data.size = size
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    obj.rotation_euler = (Vector(target)-obj.location).to_track_quat("-Z", "Y").to_euler()


light("Key softbox", (.05, -.28, .55), 18, .45)
light("Rim softbox", (.65, .2, .4), 24, .35)
light("Front fill", (.42, -.55, .18), 5, .3)
scene.world.use_nodes = True
scene.world.node_tree.nodes["Background"].inputs[0].default_value = (.80, .84, .89, 1.)
scene.world.node_tree.nodes["Background"].inputs[1].default_value = .55
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = int(os.environ.get("POUR_RENDER_SAMPLES", "96"))
scene.cycles.use_denoising = True
scene.cycles.max_bounces = 16
scene.cycles.transmission_bounces = 12
scene.cycles.transparent_max_bounces = 16
scene.render.threads_mode = "FIXED"
scene.render.threads = 16
scene.view_settings.view_transform = "AgX"
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = str(OUT / "mpm_raw.png")
bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "mpm_scene.blend"))
bpy.ops.render.render(write_still=True)
(OUT / "render_provenance.json").write_text(json.dumps(dict(
    scene_sha256=SCENE_SHA,
    snapshot_sha256=INFO.get("surface_reconstruction", {}).get("snapshot_sha256"),
    rendered_image_sha256=hashlib.sha256((OUT / "mpm_raw.png").read_bytes()).hexdigest(),
    renderer_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    engine="Blender Cycles", version=bpy.app.version_string, samples=scene.cycles.samples,
    appearance="Blue simulated liquid for visibility and consistency with the MPM curve; fixed neutral lighting",
    cup_normals="Outward winding; smooth walls with edges split above 30 degrees",
    physics_or_particle_positions_modified=False), indent=2)+"\n")
