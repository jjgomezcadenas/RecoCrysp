#!/usr/bin/env python3
"""Water, attenuation + scatter. Top row: central slice of the reconstructed
water sphere (all attenuation-corrected) for gold (trues only), uncorrected
(trues+scatter, no scatter correction), and scatter-corrected. Bottom: radial
profiles of the three (each normalized to its own interior mean) and the
uncorrected-minus-corrected difference (the scatter background).

Scatter is a smooth additive background, so -- like randoms -- it is partly
degenerate with a uniform sphere's signal; the clearest signature is expected
outside the sphere (a halo) and as a mild interior change. The diagnostics
printed every run quantify it.

  python3 att_scatter_plot.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "water_bgo_1MBq_att_scatter.npz"))
R = float(d["radius_mm"])
ext = float(d["extent"])
extent = [-ext, ext, -ext, ext]


def _report():
    r = d["radii"]
    g, u, c = d["prof_gold"], d["prof_uncorr"], d["prof_corr"]
    out = r > R + 8
    su, sc = d["slice_uncorr"], d["slice_corr"]
    print("water_bgo_1MBq attenuation+scatter diagnostics")
    print(f"  profile max|uncorr-gold| = {np.abs(u - g).max():.4f}")
    print(f"  profile max|corr -gold|  = {np.abs(c - g).max():.4f}")
    print(f"  profile max|uncorr-corr| = {np.abs(u - c).max():.4f}")
    print(f"  slice  mean|uncorr-corr| = {np.abs(su - sc).mean():.4e} "
          f"(rel {np.abs(su - sc).mean() / sc.mean():.3%})")
    print(f"  outside-sphere halo (gold/uncorr/corr) = "
          f"{g[out].mean():.4f} / {u[out].mean():.4f} / {c[out].mean():.4f}")


_report()

fig = plt.figure(figsize=(15, 8.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.28, wspace=0.25)

vmax = np.percentile(d["slice_corr"], 99.5)
for j, (key, title) in enumerate((("slice_gold", "gold (trues only)"),
                                  ("slice_uncorr", "uncorrected (trues+scatter)"),
                                  ("slice_corr", "scatter-corrected"))):
    ax = fig.add_subplot(gs[0, j])
    ax.imshow(d[key].T, origin="lower", extent=extent, cmap="magma", vmin=0, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("x (mm)")
    if j == 0:
        ax.set_ylabel("y (mm)")

axp = fig.add_subplot(gs[1, 0:2])
axp.plot(d["radii"], d["prof_gold"], "-o", ms=3, color="k", label="gold (trues only)")
axp.plot(d["radii"], d["prof_uncorr"], "-s", ms=3, color="#d62728", label="uncorrected")
axp.plot(d["radii"], d["prof_corr"], "-^", ms=3, color="#1f77b4", label="scatter-corrected")
axp.axvline(R, color="gray", lw=0.8, ls=":")
axp.set_xlabel("radius (mm)"); axp.set_ylabel("normalized activity")
axp.set_title("radial profiles")
axp.legend(frameon=False); axp.grid(alpha=0.3)

axd = fig.add_subplot(gs[1, 2])
im = axd.imshow((d["slice_uncorr"] - d["slice_corr"]).T, origin="lower", extent=extent,
                cmap="RdBu_r", vmin=-0.25 * vmax, vmax=0.25 * vmax)
axd.set_title("uncorrected - corrected")
axd.set_xlabel("x (mm)"); axd.set_ylabel("y (mm)")
fig.colorbar(im, ax=axd, fraction=0.046)

out = os.path.join(HERE, "water_bgo_1MBq_att_scatter.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
