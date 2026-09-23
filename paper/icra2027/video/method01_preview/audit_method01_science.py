"""Read-only scientific audit of the current method-1 source and encoded clip.

Writes only science_audit/. Does not rebuild the shared film or run physics.
"""
from pathlib import Path
import hashlib, json, subprocess, sys
import cv2
import numpy as np
from PIL import Image

P = Path(__file__).resolve().parent
ROOT = P.parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(P.parent)]
import render_method01 as m
from narrative_captions import CUES
from experiments.robotics.plate_observable_field import reconstruct

OUT = P / 'science_audit'
OUT.mkdir(exist_ok=True)


def sha(path):
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    run = ROOT / 'out/press_separated_20260913/monotonic320'
    ep = ROOT / 'press_real_data/ep0001'
    clip = P.parent / 'sections/observe.mp4'
    paths = [P / 'render_method01.py', P.parent / 'narrative_captions.py',
             P.parent / 'method_slide_frame.py', clip,
             P.parent / 'references/2027_ICRA_FORM.pdf',
             ROOT / 'press_real_data/handoff_ep0-7/README.md',
             ROOT / 'press_real_data/handoff_ep0-7/ep0001/particles.npz',
             ROOT / 'experiments/robotics/plate_observable_field.py',
             ROOT / 'experiments/robotics/plate_observable_identify.py',
             ROOT / 'experiments/robotics/press_hardware_observe.py',
             run / 'fit_A/tracks.npz', run / 'field_selection.json',
             run / 'inputs_A/known.json', run / 'inputs_A/calibration.json',
             P / 'animation_stereo_kinematics.npz', P / 'animation_hardware_rgb.npz',
             ROOT / 'out/press_observation_assessment_20260913/ep0001/robot_force.npz',
             ep / 'states.jsonl', ep / 'actions.jsonl', ep / 'meta.json',
             ep / 'frames_hand.jsonl', ep / 'hand_rgb.mp4']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    D = m.data()
    report = {'scope': 'Read-only source/data audit; no material fit, physics, or main-movie write.'}

    # Recreate the plotted force directly from the original robot log.
    records = [json.loads(line) for line in (ep / 'states.jsonl').read_text().splitlines()]
    states = [r['state'] for r in records]
    host = np.array([r['t'] for r in records])
    baseline_index = next(i for i,s in enumerate(states)
                          if s['force_control']['active'] and s['force_control']['phase'] == 'baseline')
    action = next(json.loads(line) for line in (ep / 'actions.jsonl').read_text().splitlines()
                  if json.loads(line)['type'] == 'press')
    force = read(ep / 'meta.json')['press']['force_sign'] * (
        np.array([s['ext_force_base'][2] for s in states]) - action['baseline_raw_n'])
    np.testing.assert_array_equal(force, D['force']['incremental_normal_force'])
    np.testing.assert_array_equal(host - host[baseline_index], D['force']['time'])
    visible = (D['force']['time'] >= 0) & (D['force']['time'] <= 2)
    report['force'] = {'raw_log_match': True, 'units': 'N', 'source_interval_s': [0, 2],
                       'min_max_N': [float(force[visible].min()), float(force[visible].max())],
                       'scope': 'Recorded normal force relative to recorded contact baseline, not absolute calibrated force.'}

    # Both cameras, not a decorative one-view extrusion, supply the 3D points.
    cams = read(run / 'inputs_A/calibration.json')
    tr = D['tr']; errors = []; reprojection = []
    for frame in range(10, 191, 10):
        ids = np.flatnonzero(tr['valid'][frame])
        pixels = [tr['pixels'][ci,frame,ids].T for ci in (0, 1)]
        homogeneous = cv2.triangulatePoints(np.array(cams[0]['P']), np.array(cams[1]['P']), *pixels)
        xyz = (homogeneous[:3] / homogeneous[3]).T
        errors.append(float(np.max(abs(xyz - tr['world'][frame,ids]))))
        for ci, cam in enumerate(cams):
            q = np.c_[xyz, np.ones(len(xyz))] @ np.array(cam['P']).T
            reprojection.extend(np.linalg.norm(q[:,:2] / q[:,2,None] - pixels[ci].T, axis=1))
    assert max(errors) < 1e-10
    assert max(reprojection) < .4
    assert (D['patch_weights'] >= -1e-10).all()
    np.testing.assert_allclose(D['patch_weights'].sum(1), 1, atol=1e-12)
    report['stereo'] = {'triangulation_max_difference_m': max(errors),
                        'max_reprojection_error_px': float(max(reprojection)),
                        'surface_interpolation': 'Convex barycentric interpolation of valid tracks; no extrapolation.',
                        'depth_cue': '2 mm schematic extrusion, not a measured interior thickness.'}

    # Re-evaluate the original selected displacement basis, with no material fit.
    known = read(run / 'inputs_A/known.json')
    selected = read(run / 'field_selection.json')['selected']
    known['cross_section_degree'] = selected['cross_section_degree']
    X,V,F,q,error = reconstruct(tr, known, selected['spacing'],
                              quadrature=D['kin']['reference'], smoothing=selected['smoothing'])
    np.testing.assert_allclose(X, D['kin']['X'], rtol=0, atol=1e-10)
    np.testing.assert_allclose(V, D['kin']['V'], rtol=0, atol=1e-10)
    frozen = np.load(ROOT / 'out/method_shaping_separated_20260913/reconstructed_display.npz')
    np.testing.assert_allclose(X[172], frozen['x'], rtol=0, atol=1e-10)
    assert np.isfinite(F).all() and np.linalg.det(F).min() > 0
    report['interior_formula'] = {'formula': 'u(X,t) = sum_j c_j(t) phi_j(X)',
                                  'meaning': 'Displacement, with x = X + u; square-symmetric spatial basis fitted to surface tracks.',
                                  'cached_positions_max_error_m': float(np.max(abs(X-D['kin']['X']))),
                                  'cached_velocities_max_error_m_s': float(np.max(abs(V-D['kin']['V']))),
                                  'min_deformation_J': float(np.linalg.det(F).min()),
                                  'deformation_output': 'F = I + grad_X u is computed by the same reconstruction.'}

    # RGB-D footage and motion share the timestamp convention and saved samples.
    rgb = D['rgb']; cap = cv2.VideoCapture(str(ep / 'hand_rgb.mp4'))
    for j in (0, len(rgb['time'])//2, len(rgb['time'])-1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(rgb['frame_idx'][j]))
        ok, image = cap.read(); assert ok
        raw = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[125:286,330:590]
        np.testing.assert_array_equal(raw, rgb['rgb'][j])
    cap.release()
    time_error = max(np.min(abs(rgb['time']-t)) for t in np.linspace(.05,2,401))
    assert time_error < .018
    hp = D['hp']; centered = D['display_pos']
    np.testing.assert_allclose(np.ptp(centered,axis=1), np.ptp(hp['pos'].astype(float),axis=1), atol=1e-12)
    np.testing.assert_array_equal(centered[:,:,2], hp['pos'][:,:,2])
    slip = {}
    select = (hp['times'][:-1]>=.05) & (hp['times'][1:]<=2)
    for contact in (1,2):
        both = (hp['stuck'][:-1]==contact) & (hp['stuck'][1:]==contact) & select[:,None]
        tangential = np.linalg.norm(np.diff(centered,axis=0)[both,:2],axis=1)*1000
        slip[str(contact)] = float(tangential.max())
        assert tangential.max() < .005
    report['rgbd'] = {'rgb_cache_exact_match': True, 'max_rgb_time_error_s': float(time_error),
                      'max_contact_tangential_increment_mm_centered_frame': slip,
                      'assumptions': ['axisymmetry','incompressible reconstruction','no slip relative to contacts'],
                      'not_directly_observed': 'Interior trajectories; the upstream handoff supplies them under these assumptions.',
                      'upstream_limit': 'Original RGB-D flow solver is not in the handoff; its model is documented in README, and saved trajectories were checked.'}
    # Display shells approximate the particle envelope, not an exact volume mesh.
    volumes = []; initial = None; maximum_displacement = 0
    for t in np.linspace(.05,2,81):
        points,_,_ = m.lerp_frame(hp['pos'],hp['times'],t)
        mesh = m.profile(m.centered_positions(points,D['reference']))
        center = (mesh[:,0,:2]+mesh[:,32,:2])/2
        radii = np.linalg.norm(mesh[:,0,:2]-center,axis=1)
        volumes.append(float(np.trapezoid(np.pi*radii*radii,mesh[:,0,2])))
        maximum_displacement = max(maximum_displacement, float(np.linalg.norm(points-D['reference'],axis=1).max()))
        projections = [p(mesh.reshape(-1,3))-p.center for p in D['hproj']]
        for other in projections[1:]: np.testing.assert_allclose(projections[0],other,atol=1e-10)
    assert maximum_displacement <= D['disp_max_m']
    np.testing.assert_allclose(m.lerp_frame(hp['pos'],hp['times'],.05)[0],D['reference'],atol=0)
    report['display'] = {'all_four_hardware_projections_same_geometry_and_scale': True,
                          'shell_volume_ratio_range': [min(volumes)/volumes[0],max(volumes)/volumes[0]],
                          'shell_qualification': 'Smoothed discrete particle envelope: qualitative surface illustration, not an exact incompressible volume mesh.',
                          'color': 'Norm of original-frame displacement since 0.05 s, not initial height, speed, or amplified displacement.',
                          'max_displacement_mm': maximum_displacement*1000,
                          'color_max_mm': D['disp_max_m']*1000}
    report['advection_formula'] = {'formula': 'xdot_p(t) = v(x_p(t),t)',
                                   'meaning': 'Particle trajectory derivative equals the inferred field evaluated at its current position and time.',
                                   'not_a_fit': 'This is the advection rule, not a new material-identification equation.'}

    previous = 0
    for start,end,caption in CUES['observe']:
        assert start == previous and end > start and len(caption)/(end-start) <= 20
        previous = end
    assert previous == m.DURATION
    for phase in (0.,1.2,3.8):
        reference = np.asarray(m.draw_method01(phase,reveal=False))[75:650]
        for cycle in (1,3,7):
            assert m.playback_times(phase) == m.playback_times(phase+4*cycle)
            np.testing.assert_array_equal(reference,np.asarray(m.draw_method01(phase+4*cycle,reveal=False))[75:650])
    cap = cv2.VideoCapture(str(clip))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == m.DURATION*m.FPS
    assert cap.get(cv2.CAP_PROP_FPS) == m.FPS
    sheet = Image.new('RGB',(1280,1440),'white')
    for i,(start,end,_) in enumerate(CUES['observe']):
        cap.set(cv2.CAP_PROP_POS_MSEC,(start+end)*500)
        ok, frame = cap.read(); assert ok
        image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        image.save(OUT/f'caption_{i}.png')
        sheet.paste(image.resize((640,360)),((i%2)*640,(i//2)*360))
    cap.release(); sheet.save(OUT/'encoded_review.jpg',quality=94)
    subprocess.run(['ffmpeg','-v','error','-i',str(clip),'-f','null','-'],check=True)
    assert all(sha(ROOT/path)==digest for path,digest in hashes.items()), 'Source changed during audit'
    report['presentation'] = {'duration_s': m.DURATION, 'frames': m.DURATION*m.FPS,
                               'source_loop_period_s': 4, 'source_speed': .5,
                               'caption_reading_rates': 'All <=20 characters/s',
                               'loop_geometry_identical': True, 'full_decode': 'PASS'}
    report['source_sha256'] = hashes
    report['checks_passed'] = True
    (OUT/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))


if __name__ == '__main__':
    main()
