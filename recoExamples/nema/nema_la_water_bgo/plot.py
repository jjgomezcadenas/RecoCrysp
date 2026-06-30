#!/usr/bin/env python3
"""NEMA water three-way reconstruction for ONE variant, with NEMA metrics. Top:
gold/uncorr/corr central slices (the spheres), each titled with the 10 mm CRC and
NEMA background variability. Bottom: CRC and NEMA BV vs sphere diameter, against the
clinical reference (Discovery MI TOF-OSEM, Vandendriessche 2019).

  python3 plot.py [tag]   (default mlem; reads out/nema_la_water_bgo_<tag>.npz)
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
tag = sys.argv[1] if len(sys.argv) > 1 else "mlem"
d = np.load(os.path.join(HERE, "out", f"nema_la_water_bgo_{tag}.npz"))
ext = float(d["extent_xy"]); extent = [-ext, ext, -ext, ext]
diam = d["diam_mm"]; order = np.argsort(diam)
vmax = np.percentile(d["slice_gold"], 99.7)

# clinical reference: GE Discovery MI 3-ring, NEMA NU2-2012 IQ, TOF-OSEM (no PSF)
CLIN_DIAM = np.array([10., 13., 17., 22., 28., 37.])
CLIN_CR = np.array([47.4, 59.3, 67.0, 77.0, 82.5, 85.1])
CLIN_BV = np.array([16.4, 12.1, 9.1, 6.6, 5.1, 3.8])

fig = plt.figure(figsize=(13, 9))
fig.suptitle(f"nema_la_water  [{tag}]", fontsize=12)
for i, (k, t) in enumerate([("gold", "gold (trues, AC)"),
                            ("uncorr", "uncorrected (AC only)"),
                            ("corr", "corrected (AC+scat+rand)")]):
    ax = fig.add_subplot(2, 3, i + 1)
    ax.imshow(d[f"slice_{k}"].T, origin="lower", extent=extent, cmap="magma", vmin=0, vmax=vmax)
    ax.set_xlim(-110, 110); ax.set_ylim(-110, 110)
    crc10 = np.asarray(d[f"crc_{k}"])[order][0]; bv10 = np.asarray(d[f"bv_{k}"])[order][0]
    ax.set_title(f"{t}\n10mm: CRC {crc10:.0f}%   BV {bv10:.1f}%", fontsize=9)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")

curves = (("gold", "gold", "#2ca02c", "-o"), ("corr", "corrected", "#1f77b4", "-s"),
          ("uncorr", "uncorrected", "#d62728", "-^"))
axc = fig.add_subplot(2, 2, 3)
for k, lbl, col, mk in curves:
    axc.plot(diam[order], np.asarray(d[f"crc_{k}"])[order], mk, ms=5, color=col, label=lbl)
axc.plot(CLIN_DIAM, CLIN_CR, "k-D", ms=4, lw=1, label="clinical [ref]")
axc.set_xlabel("sphere diameter (mm)"); axc.set_ylabel("CRC (%)")
axc.set_title("contrast recovery"); axc.grid(alpha=0.3); axc.legend(frameon=False, fontsize=8)

axb = fig.add_subplot(2, 2, 4)
for k, lbl, col, mk in curves:
    axb.plot(diam[order], np.asarray(d[f"bv_{k}"])[order], mk, ms=5, color=col, label=lbl)
axb.plot(CLIN_DIAM, CLIN_BV, "k-D", ms=4, lw=1, label="clinical [ref]")
axb.set_xlabel("sphere diameter (mm)"); axb.set_ylabel("NEMA background variability (%)")
axb.set_title("background variability"); axb.grid(alpha=0.3); axb.legend(frameon=False, fontsize=8)

figdir = os.path.join(HERE, "figures"); os.makedirs(figdir, exist_ok=True)
out = os.path.join(figdir, f"nema_la_water_bgo_{tag}.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
