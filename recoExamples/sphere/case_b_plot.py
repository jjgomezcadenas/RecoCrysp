#!/usr/bin/env python3
"""Case (b) randoms. Top row: the reconstructed sphere (central slice) for gold
(trues only), uncorrected (all prompts), and randoms-corrected. Bottom row: the
radial profiles of the three, and the uncorrected-minus-corrected difference
(the randoms background, on a magnified scale). At low randoms fractions the
three are nearly identical -- the correction is validated, the effect is small.

  python3 case_b_plot.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "case_b_results.npz"))
R = float(d["radius_mm"])
ext = float(d["extent"])
extent = [-ext, ext, -ext, ext]

fig = plt.figure(figsize=(15, 8.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.28, wspace=0.25)

# --- top: the reconstructed sphere, three ways ---------------------------------
vmax = np.percentile(d["slice_corr"], 99.5)
for j, (key, title) in enumerate((("slice_gold", "gold (trues only)"),
                                  ("slice_uncorr", "uncorrected (prompts)"),
                                  ("slice_corr", "randoms-corrected"))):
    ax = fig.add_subplot(gs[0, j])
    ax.imshow(d[key].T, origin="lower", extent=extent, cmap="magma", vmin=0, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("x (mm)")
    if j == 0:
        ax.set_ylabel("y (mm)")

# --- bottom-left: radial profiles ----------------------------------------------
axp = fig.add_subplot(gs[1, 0:2])
axp.plot(d["radii"], d["prof_gold"], "-o", ms=3, color="k", label="gold (trues only)")
axp.plot(d["radii"], d["prof_uncorr"], "-s", ms=3, color="#d62728", label="uncorrected")
axp.plot(d["radii"], d["prof_corr"], "-^", ms=3, color="#1f77b4", label="randoms-corrected")
axp.axvline(R, color="gray", lw=0.8, ls=":")
axp.set_xlabel("radius (mm)"); axp.set_ylabel("normalized activity")
axp.set_title("radial profiles")
axp.legend(frameon=False); axp.grid(alpha=0.3)

# --- bottom-right: the randoms background (uncorrected - corrected) -------------
axd = fig.add_subplot(gs[1, 2])
im = axd.imshow((d["slice_uncorr"] - d["slice_corr"]).T, origin="lower", extent=extent,
                cmap="RdBu_r", vmin=-0.05 * vmax, vmax=0.05 * vmax)
axd.set_title("uncorrected - corrected  (×20)")
axd.set_xlabel("x (mm)"); axd.set_ylabel("y (mm)")
fig.colorbar(im, ax=axd, fraction=0.046)

out = os.path.join(HERE, "case_b_result.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
