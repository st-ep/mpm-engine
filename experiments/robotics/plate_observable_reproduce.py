"""Reproduce the plate pilot in a fresh directory; never overwrite an old run."""
import argparse
import os
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--device',default='cuda:0');a=ap.parse_args()
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(name,'1')
    from experiments.robotics.plate_observable_probe import PROTOCOL,record
    from experiments.robotics.plate_observable_camera import render
    from experiments.robotics.plate_observable_identify import track,fit as uniform
    from experiments.robotics.plate_observable_field import select,fit as pressure_sensitive
    from experiments.robotics.plate_observable_divfree import fit
    from experiments.robotics.plate_observable_audit import isolate,ideal,evaluate
    from experiments.robotics.plate_observable_report import media,write_report,provenance
    from experiments.elastic.strip_camera_observe import save
    a.out.mkdir(parents=True,exist_ok=False);save(a.out/'protocol.json',PROTOCOL)
    for k in 'AB':
        record(a.out,k,'truth',a.device);render(a.out,k)
        track(a.out/f'inputs_{k}',a.out/f'fit_{k}');uniform(a.out/f'inputs_{k}',a.out/f'fit_{k}')
    select(a.out)
    for k in 'AB':
        pressure_sensitive(a.out,k);fit(a.out,k)
        for kind in ['prediction','fine','fine_prediction']:record(a.out,k,kind,a.device)
    isolate(a.out);ideal(a.out);evaluate(a.out);media(a.out);write_report(a.out);provenance(a.out)


if __name__=='__main__':main()
