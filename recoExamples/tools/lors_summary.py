#!/usr/bin/env python3
"""Summarize a PTCRYSP listmode coincidence file (lors_det.h5).

Reports the event count and the ground-truth class breakdown used throughout the
MC-data studies -- true / scatter (split into single SS and multiple MS) / random
-- plus the per-event scatter-order histogram. Run this on every new dataset
before building a scenario, instead of ad-hoc one-off snippets.

  python3 recoExamples/tools/lors_summary.py <path/to/lors_det.h5>

Classification (keying randoms FIRST):
  random   : truth == 2
  true     : truth == 0
  scatter  : truth == 1, split by nscat1+nscat2 == 1 (single) vs >= 2 (multiple)
"""
import sys
import h5py
import numpy as np


def summarize(path):
    with h5py.File(path, "r") as f:
        t = f["truth"][()]
        n1 = f["nscat1"][()].astype(int)
        n2 = f["nscat2"][()].astype(int)
        attrs = dict(f.attrs)
    N = len(t)
    order = n1 + n2
    true = t == 0
    scat = t == 1
    rand = t == 2
    ss = scat & (order == 1)
    ms = scat & (order >= 2)

    def line(label, mask):
        c = int(mask.sum())
        print(f"  {label:<14} {c:>12,}  ({100 * c / N:6.3f}%)")

    print(f"file = {path}")
    print(f"N    = {N:,}")
    line("true (t=0)", true)
    line("scatter(t=1)", scat)
    line("  single SS", ss)
    line("  multiple MS", ms)
    line("random (t=2)", rand)
    hist = [int((order == k).sum()) for k in range(6)]
    print(f"  nscat-order hist (0..5): {hist}  max {int(order.max())}")
    if attrs:
        keys = ("sigma_xyz_mm", "tau_ns", "emin_keV", "xyz_scale_mm", "e_scale_keV")
        shown = {k: attrs[k] for k in keys if k in attrs}
        if shown:
            print(f"  attrs: {shown}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    summarize(sys.argv[1])
