#!/usr/bin/env python3
"""Noise (and CRC) vs voxel size (from scan_voxel.jl), at fixed physical
regularization (MLEM niter + fixed-mm post-filter). Left: background CoV vs voxel
-- flat/slightly-rising means voxel size is not a noise-reduction lever; the sens
CoV (fixed-LOR sampling) is shown for reference. Right: CRC vs voxel (the trend is
partly an ROI-discretization artifact -- coarse VOIs bias toward the bright
centre, so read it with care).

  python3 scan_voxel_plot.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "out", "nema_la_air_bgo_voxel_scan.npz"))
vx = d["voxel_mm"]
diam = d["diam_mm"]
small = np.argsort(diam)[:3]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))
axL.plot(vx, d["cov"], "-o", ms=7, color="#1f77b4", label="background CoV")
axL.plot(vx, d["sens_cov"], "-s", ms=5, color="#888888", label="sensitivity CoV (fixed nsens)")
axL.set_xlabel("voxel size (mm)"); axL.set_ylabel("CoV")
axL.set_title(f"noise vs voxel (niter={int(d['niter'])} + {float(d['fwhm_mm']):.0f}mm filter)\n"
              "background floor ~flat -> voxel is not a noise lever")
axL.set_ylim(0, max(d["cov"]) * 1.2); axL.grid(alpha=0.3); axL.legend(frameon=False)

crc = d["crc"]
axR.plot(vx, crc[:, small].mean(axis=1), "-^", ms=6, color="#d62728", label="mean CRC 10/13/17 mm")
axR.plot(vx, crc[:, int(np.argmin(diam))], "-o", ms=5, color="#ff9896", label="CRC 10 mm")
axR.set_xlabel("voxel size (mm)"); axR.set_ylabel("CRC (%)")
axR.set_title("CRC vs voxel (partly ROI-discretization artifact)")
axR.grid(alpha=0.3); axR.legend(frameon=False)

os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
out = os.path.join(HERE, "figures", "nema_la_air_bgo_voxel_scan.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
