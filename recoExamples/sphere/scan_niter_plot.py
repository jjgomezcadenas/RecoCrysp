#!/usr/bin/env python3
"""Overlay the case-(a) radial profiles at 10/20/40 MLEM iterations.

If the central gradient fills toward 1.0 as iterations increase, the dip is
under-convergence (normalization is fine). If it sticks, it's a sensitivity bias.

  python3 scan_niter_plot.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "niter_scan_results.npz"))
R = float(d["radius_mm"])

fig, ax = plt.subplots(figsize=(7, 5))
for k, c in zip((10, 20, 40), ("#1f77b4", "#ff7f0e", "#2ca02c")):
    ax.plot(d["radii"], d[f"prof_{k}"], "-o", ms=3, color=c, label=f"{k} iterations")
ax.axhline(1.0, color="k", lw=0.8, ls="--")
ax.axvline(R, color="gray", lw=0.8, ls=":")
ax.set_xlabel("radius (mm)")
ax.set_ylabel("normalized activity")
ax.set_title("case (a): radial profile vs MLEM iterations")
ax.set_ylim(0, 1.3)
ax.legend(frameon=False)
ax.grid(alpha=0.3)

fig.tight_layout()
out = os.path.join(HERE, "niter_scan_result.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
