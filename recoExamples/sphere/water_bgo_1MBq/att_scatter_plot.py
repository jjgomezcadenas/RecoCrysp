#!/usr/bin/env python3
"""Water, attenuation + scatter. Top row: central slice of the reconstructed
water sphere (all attenuation-corrected) for gold (trues only), uncorrected
(trues+scatter, no scatter correction), and scatter-corrected. Bottom row:
radial profiles of the three; the uncorrected-minus-corrected difference map
(robust color scale, so a single low-sensitivity corner voxel doesn't dominate);
and the radial difference-from-gold of uncorr and corr.

The correction's effect is structured, not uniform: near-zero in the degenerate
core, a small POSITIVE ring just inside the edge (scatter had filled in the edge
contrast; the correction restores it), and a strong REMOVAL in the far halo /
corners (pure-scatter voxels). The zoned diagnostics below quantify it.

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

su, sc = d["slice_uncorr"], d["slice_corr"]
diff = su - sc
n = su.shape[0]
ax1d = np.linspace(-ext, ext, n)
X, Y = np.meshgrid(ax1d, ax1d, indexing="ij")
rr = np.hypot(X, Y)


def _report():
    g, u, c = d["prof_gold"], d["prof_uncorr"], d["prof_corr"]
    print("water_bgo_1MBq attenuation+scatter diagnostics")
    print(f"  profile max|uncorr-gold| = {np.abs(u - g).max():.4f}")
    print(f"  profile max|corr -gold|  = {np.abs(c - g).max():.4f}  (lower = correction helps)")
    print("  (radial-zone breakdown of the correction: att_scatter_zones.py)")


_report()

fig = plt.figure(figsize=(15, 8.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.28, wspace=0.28)

vmax = np.percentile(d["slice_corr"], 99.5)
for j, (key, title) in enumerate((("slice_gold", "gold (trues only)"),
                                  ("slice_uncorr", "uncorrected (trues+scatter)"),
                                  ("slice_corr", "scatter-corrected"))):
    ax = fig.add_subplot(gs[0, j])
    ax.imshow(d[key].T, origin="lower", extent=extent, cmap="magma", vmin=0, vmax=vmax)
    ax.set_title(title); ax.set_xlabel("x (mm)")
    if j == 0:
        ax.set_ylabel("y (mm)")

# radial profiles
axp = fig.add_subplot(gs[1, 0])
axp.plot(d["radii"], d["prof_gold"], "-o", ms=3, color="k", label="gold")
axp.plot(d["radii"], d["prof_uncorr"], "-s", ms=3, color="#d62728", label="uncorrected")
axp.plot(d["radii"], d["prof_corr"], "-^", ms=3, color="#1f77b4", label="corrected")
axp.axvline(R, color="gray", lw=0.8, ls=":")
axp.set_xlabel("radius (mm)"); axp.set_ylabel("normalized activity")
axp.set_title("radial profiles"); axp.legend(frameon=False); axp.grid(alpha=0.3)

# difference map, robust symmetric scale (ignore the single-voxel corner spike)
axd = fig.add_subplot(gs[1, 1])
lim = np.percentile(np.abs(diff), 99)
im = axd.imshow(diff.T, origin="lower", extent=extent, cmap="RdBu_r", vmin=-lim, vmax=lim)
axd.add_patch(plt.Circle((0, 0), R, fill=False, color="k", lw=0.6, ls=":"))
axd.set_title("uncorrected - corrected"); axd.set_xlabel("x (mm)"); axd.set_ylabel("y (mm)")
fig.colorbar(im, ax=axd, fraction=0.046)

# radial difference from gold: how far each is from gold vs radius
axr = fig.add_subplot(gs[1, 2])
axr.plot(d["radii"], d["prof_uncorr"] - d["prof_gold"], "-s", ms=3, color="#d62728", label="uncorr - gold")
axr.plot(d["radii"], d["prof_corr"] - d["prof_gold"], "-^", ms=3, color="#1f77b4", label="corr - gold")
axr.axhline(0, color="gray", lw=0.6); axr.axvline(R, color="gray", lw=0.8, ls=":")
axr.set_xlabel("radius (mm)"); axr.set_ylabel("profile - gold")
axr.set_title("difference from gold"); axr.legend(frameon=False); axr.grid(alpha=0.3)

out = os.path.join(HERE, "water_bgo_1MBq_att_scatter.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
