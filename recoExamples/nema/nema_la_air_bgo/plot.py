#!/usr/bin/env python3
"""NEMA low-activity (vacuum, 2.9% randoms) -- contrast phantom, randoms study.
Top row: central transverse slice of the reconstructed phantom (six hot spheres)
for gold (trues), uncorrected (prompts), and randoms-corrected. Bottom: the
per-sphere contrast-recovery coefficient (CRC) versus sphere diameter for the
three. Randoms add a roughly uniform background that lifts the cold-region level
and so washes out contrast (lower CRC, uncorrected); the correction should pull
the CRC back toward the gold curve.

  CRC = (sphere_mean / background_mean - 1) / (hot_ratio - 1) * 100%

Diagnostics (CRC per sphere) printed every run.

  python3 plot.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "out", "nema_la_air_bgo.npz"))
diam = d["diam_mm"]
ratio = float(d["hot_ratio"])
ext = float(d["extent_xy"])
extent = [-ext, ext, -ext, ext]
order = np.argsort(diam)


def crc(sph, bg):
    return (np.asarray(sph) / bg - 1.0) / (ratio - 1.0) * 100.0


crc_g = crc(d["sph_gold"], float(d["bg_gold"]))
crc_u = crc(d["sph_uncorr"], float(d["bg_uncorr"]))
crc_c = crc(d["sph_corr"], float(d["bg_corr"]))


def _report():
    print(f"nema_la_air_bgo contrast-recovery diagnostics (CRC %, hot:bg = {ratio:.0f}:1)")
    print(f"  background mean (gold/uncorr/corr) = "
          f"{float(d['bg_gold']):.4f} / {float(d['bg_uncorr']):.4f} / {float(d['bg_corr']):.4f}")
    print("  diam(mm)   gold   uncorr   corr")
    for i in order[::-1]:
        print(f"    {diam[i]:5.0f}   {crc_g[i]:5.1f}   {crc_u[i]:5.1f}   {crc_c[i]:5.1f}")


_report()

fig = plt.figure(figsize=(15, 8.4))
gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1.0], hspace=0.28, wspace=0.25)

vmax = np.percentile(d["slice_gold"], 99.7)
for j, (key, title) in enumerate((("slice_gold", "gold (trues)"),
                                  ("slice_uncorr", "uncorrected (prompts)"),
                                  ("slice_corr", "randoms-corrected"))):
    ax = fig.add_subplot(gs[0, j])
    ax.imshow(d[key].T, origin="lower", extent=extent, cmap="magma", vmin=0, vmax=vmax)
    ax.set_title(title); ax.set_xlabel("x (mm)")
    if j == 0:
        ax.set_ylabel("y (mm)")

axc = fig.add_subplot(gs[1, :])
axc.plot(diam[order], crc_g[order], "-o", ms=5, color="k", label="gold (trues)")
axc.plot(diam[order], crc_u[order], "-s", ms=5, color="#d62728", label="uncorrected")
axc.plot(diam[order], crc_c[order], "-^", ms=5, color="#1f77b4", label="randoms-corrected")
axc.axhline(100, color="gray", lw=0.8, ls=":")
axc.set_xlabel("sphere diameter (mm)"); axc.set_ylabel("contrast recovery CRC (%)")
axc.set_title("contrast recovery vs sphere size")
axc.legend(frameon=False); axc.grid(alpha=0.3)

os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
out = os.path.join(HERE, "figures", "nema_la_air_bgo.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
