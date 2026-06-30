#!/usr/bin/env python3
"""NEMA water (full physics) three-way reconstruction (from run.jl): gold (trues, AC) /
uncorrected (prompts, AC only) / corrected (AC + scatter + randoms). Top: central
slices on a common scale. Bottom: per-sphere CRC vs diameter for the three -- here the
curves SEPARATE (uncorrected contrast washed by scatter+randoms filling the cold
background; corrected recovers toward gold), unlike the degenerate uniform sphere.

  python3 plot.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "out", "nema_la_water_bgo.npz"))
ext = float(d["extent_xy"]); extent = [-ext, ext, -ext, ext]
diam = d["diam_mm"]; order = np.argsort(diam)
vmax = np.percentile(d["slice_gold"], 99.7)

fig = plt.figure(figsize=(13, 8.5))
titles = [("slice_gold", "cov_gold", "gold (trues, AC)"),
          ("slice_uncorr", "cov_uncorr", "uncorrected (AC only)"),
          ("slice_corr", "cov_corr", "corrected (AC+scatter+randoms)")]
for i, (sk, ck, t) in enumerate(titles):
    ax = fig.add_subplot(2, 3, i + 1)
    ax.imshow(d[sk].T, origin="lower", extent=extent, cmap="magma", vmin=0, vmax=vmax)
    ax.set_xlim(-110, 110); ax.set_ylim(-110, 110)
    ax.set_title(f"{t}\nbg CoV {float(d[ck]):.2f}")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")

axc = fig.add_subplot(2, 1, 2)
for key, lbl, col, mk in (("crc_gold", "gold (trues)", "#2ca02c", "-o"),
                          ("crc_corr", "corrected", "#1f77b4", "-s"),
                          ("crc_uncorr", "uncorrected", "#d62728", "-^")):
    axc.plot(diam[order], np.asarray(d[key])[order], mk, ms=6, color=col, label=lbl)
axc.axhline(100, color="gray", lw=0.8, ls=":")
axc.set_xlabel("sphere diameter (mm)"); axc.set_ylabel("CRC (%)")
axc.set_title("contrast recovery vs sphere size (4:1)")
axc.grid(alpha=0.3); axc.legend(frameon=False)

figdir = os.path.join(HERE, "figures"); os.makedirs(figdir, exist_ok=True)
out = os.path.join(figdir, "nema_la_water_bgo.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
