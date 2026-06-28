#!/usr/bin/env python3
"""NEMA gold reconstructed with the sensitivity sampled at nsens=200M (from
spheres_200M.jl): the central slice (six spheres, clean background) and the CRC vs
sphere size. Background CoV and nsens are in the title.

  python3 spheres_200M_plot.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "out", "nema_la_air_bgo_spheres_200M.npz"))
ext = float(d["extent_xy"]); extent = [-ext, ext, -ext, ext]
diam = d["diam_mm"]; crc = d["crc"]; order = np.argsort(diam)

fig, (axi, axc) = plt.subplots(1, 2, figsize=(13, 5.6))
im = axi.imshow(d["slice"].T, origin="lower", extent=extent, cmap="magma",
                vmin=0, vmax=np.percentile(d["slice"], 99.7))
axi.set_xlim(-95, 95); axi.set_ylim(-95, 95)
axi.set_title(f"gold, nsens=200M  (bg CoV {float(d['cov']):.2f})")
axi.set_xlabel("x (mm)"); axi.set_ylabel("y (mm)")

axc.plot(diam[order], np.asarray(crc)[order], "-o", ms=6, color="#1f77b4")
axc.axhline(100, color="gray", lw=0.8, ls=":")
axc.set_xlabel("sphere diameter (mm)"); axc.set_ylabel("CRC (%)")
axc.set_title("contrast recovery vs sphere size"); axc.grid(alpha=0.3)

os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
out = os.path.join(HERE, "figures", "nema_la_air_bgo_spheres_200M.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
