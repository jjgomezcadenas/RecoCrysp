#!/usr/bin/env python3
"""NEMA gold reconstructed with the sensitivity sampled at a given nsens (from
spheres_nsens.jl): the central slice (six spheres) and the CRC vs sphere size.
Background CoV and nsens are in the title.

  python3 spheres_nsens_plot.py <out/nema_la_air_bgo_spheres_<X>M.npz>
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    sys.exit("usage: spheres_nsens_plot.py <out/..._spheres_<X>M.npz>")
NPZ = os.path.abspath(sys.argv[1])
d = np.load(NPZ)
ext = float(d["extent_xy"]); extent = [-ext, ext, -ext, ext]
diam = d["diam_mm"]; crc = d["crc"]; order = np.argsort(diam)
mtag = f"{int(d['nsens']) // 1_000_000}M"

fig, (axi, axc) = plt.subplots(1, 2, figsize=(13, 5.6))
im = axi.imshow(d["slice"].T, origin="lower", extent=extent, cmap="magma",
                vmin=0, vmax=np.percentile(d["slice"], 99.7))
axi.set_xlim(-95, 95); axi.set_ylim(-95, 95)
axi.set_title(f"gold, nsens={mtag}  (bg CoV {float(d['cov']):.2f})")
axi.set_xlabel("x (mm)"); axi.set_ylabel("y (mm)")

axc.plot(diam[order], np.asarray(crc)[order], "-o", ms=6, color="#1f77b4")
axc.axhline(100, color="gray", lw=0.8, ls=":")
axc.set_xlabel("sphere diameter (mm)"); axc.set_ylabel("CRC (%)")
axc.set_title("contrast recovery vs sphere size"); axc.grid(alpha=0.3)

figdir = os.path.join(os.path.dirname(os.path.dirname(NPZ)), "figures")
os.makedirs(figdir, exist_ok=True)
out = os.path.join(figdir, os.path.basename(NPZ).replace(".npz", ".png"))
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
