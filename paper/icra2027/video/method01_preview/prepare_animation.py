"""Cache the existing stereo reconstruction for display; no material fitting."""
from pathlib import Path
import sys,json,numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from experiments.robotics.plate_observable_field import reconstruct
run=ROOT/'out/press_separated_20260913/monotonic320'
known=json.loads((run/'inputs_A/known.json').read_text());sel=json.loads((run/'field_selection.json').read_text())['selected'];known['cross_section_degree']=sel['cross_section_degree']
tr=np.load(run/'fit_A/tracks.npz');ref=np.load(ROOT/'out/method_shaping_separated_20260913/reconstructed_display.npz')
X,V,F,q,error=reconstruct(tr,known,sel['spacing'],quadrature=ref['reference'],smoothing=sel['smoothing'])
assert np.allclose(X[172],ref['x'],atol=1e-10)
np.savez_compressed(P/'animation_stereo_kinematics.npz',X=X,V=V,reference=q,time=tr['time'])
(P/'animation_cache_checks.json').write_text(json.dumps({'scope':'Re-evaluation of the previously selected reconstruction for animation, using unchanged tracks, geometry, spatial basis, smoothing and sample locations. No new material fit or forward simulation.','frozen_frame172_max_difference_m':float(np.max(abs(X[172]-ref['x']))),'selected_field':sel},indent=2)+'\n')
print('Cached existing field at all times; frame 172 matches frozen figure.',flush=True)
