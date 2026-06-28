#!/usr/bin/env python3
"""NEMA gold reconstructed three ways (from compare_methods.jl): MLEM + Gaussian
post-filter, quadratic-smoothness prior, and Huber prior, at comparable background
noise. Top: the central slices. Bottom: contrast recovery vs sphere size for the
three. The background CoV of each is in its title.

  python3 compare_methods_plot.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "out", "nema_la_air_bgo_methods.npz"))
diam = d["diam_mm"]
ext = float(d["extent_xy"])
extent = [-ext, ext, -ext, ext]
labels = ["MLEM + 5mm filter", "quadratic β=1000", "Huber β=1000 δ=0.05"]
cov = d["cov"]
keys = ["slice_gauss", "slice_quad", "slice_huber"]
crcs = [d["crc_gauss"], d["crc_quad"], d["crc_huber"]]
order = np.argsort(diam)

fig = plt.figure(figsize=(15, 8.4))
gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1.0], hspace=0.3, wspace=0.25)

vmax = np.percentile(d["slice_gauss"], 99.7)
for j, (key, lab) in enumerate(zip(keys, labels)):
    ax = fig.add_subplot(gs[0, j])
    ax.imshow(d[key].T, origin="lower", extent=extent, cmap="magma", vmin=0, vmax=vmax)
    ax.set_title(f"{lab}\nbg CoV = {cov[j]:.2f}")
    ax.set_xlabel("x (mm)")
    if j == 0:
        ax.set_ylabel("y (mm)")

axc = fig.add_subplot(gs[1, :])
colors = ["#2ca02c", "#1f77b4", "#d62728"]
markers = ["-D", "-s", "-^"]
for crc, lab, col, mk in zip(crcs, labels, colors, markers):
    axc.plot(diam[order], np.asarray(crc)[order], mk, ms=5, color=col, label=lab)
axc.axhline(100, color="gray", lw=0.8, ls=":")
axc.set_xlabel("sphere diameter (mm)"); axc.set_ylabel("contrast recovery CRC (%)")
axc.set_title("contrast recovery vs sphere size, three methods at matched noise")
axc.legend(frameon=False); axc.grid(alpha=0.3)

os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
out = os.path.join(HERE, "figures", "nema_la_air_bgo_methods.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
