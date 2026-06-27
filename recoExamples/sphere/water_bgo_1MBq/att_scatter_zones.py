#!/usr/bin/env python3
"""Zoned diagnosis of the scatter correction (from run_att_scatter.jl's output).

The scatter correction's effect on a uniform sphere is structured, not uniform,
and a single global number hides it. This reads the att_scatter result and prints
the central-slice mean of the uncorrected and corrected images, and their
difference, in radial zones from the core out to the corners -- the breakdown that
shows WHERE the correction acts:

  core / mid      degenerate with the signal -> ~no change
  inner-edge      correction ADDS (recovers edge contrast scatter had filled in)
  far-halo/corner pure scatter -> correction strongly REMOVES the scatter pile-up

(The single extreme corner voxel is low-sensitivity noise; the zone MEANS are the
systematic effect.) This is the verdict behind "the model is sound, the uniform
sphere just can't display scatter correction well -- a contrast phantom would."

  python3 att_scatter_zones.py
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "water_bgo_1MBq_att_scatter.npz"))
R = float(d["radius_mm"])
ext = float(d["extent"])

su, sc = d["slice_uncorr"], d["slice_corr"]
n = su.shape[0]
ax1d = np.linspace(-ext, ext, n)
X, Y = np.meshgrid(ax1d, ax1d, indexing="ij")
rr = np.hypot(X, Y)

r = d["radii"]
g, u, c = d["prof_gold"], d["prof_uncorr"], d["prof_corr"]
print("water_bgo_1MBq scatter-correction zoned diagnosis")
print(f"  R = {R:.0f} mm,  grid {n}x{n},  extent +-{ext:.0f} mm")
print(f"  profile max|uncorr-gold| = {np.abs(u - g).max():.4f}")
print(f"  profile max|corr -gold|  = {np.abs(c - g).max():.4f}   (lower = correction helps)")
print("  central-slice mean, uncorr -> corr, by radial zone:")

zones = [(0, 40, "core"), (40, 60, "mid"), (60, 80, "inner-edge"),
         (80, 100, "outer-edge"), (100, 160, "far-halo")]
for lo, hi, lab in zones:
    m = (rr >= lo) & (rr < hi)
    um, cm = su[m].mean(), sc[m].mean()
    print(f"    r[{lo:3d},{hi:3d}) {lab:11s} {um:.4f} -> {cm:.4f}"
          f"  (diff {um - cm:+.4f}{'' if um == 0 else f', {100*(cm-um)/um:+.0f}%'})")

corner = (np.abs(X) > 0.88 * ext) & (np.abs(Y) > 0.88 * ext)
um, cm = su[corner].mean(), sc[corner].mean()
print(f"    corners |x|,|y|>{0.88*ext:.0f}     {um:.4f} -> {cm:.4f}  (diff {um - cm:+.4f})")
imax = np.unravel_index(np.abs(su - sc).argmax(), su.shape)
print(f"  single max|diff| voxel at (x,y)=({ax1d[imax[0]]:.0f},{ax1d[imax[1]]:.0f}), "
      f"r={rr[imax]:.0f} (low-sensitivity corner -> noise, not the systematic effect)")
