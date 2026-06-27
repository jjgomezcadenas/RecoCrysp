#!/usr/bin/env python3
"""Overlay the case-(a) radial profiles for the two sensitivity models:
surface sampling vs emission+DOI. Flatter = better normalization.

  python3 compare_norm_plot.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "compare_norm_results.npz"))
R = float(d["radius_mm"])

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(d["radii"], d["prof_surface"], "-o", ms=3, color="#d62728",
        label="surface (single radius)")
ax.plot(d["radii"], d["prof_emission"], "-o", ms=3, color="#1f77b4",
        label="surface + DOI")
ax.axhline(1.0, color="k", lw=0.8, ls="--")
ax.axvline(R, color="gray", lw=0.8, ls=":")
ax.set_xlabel("radius (mm)")
ax.set_ylabel("normalized activity")
ax.set_title("case (a): sensitivity: surface vs surface+DOI")
ax.set_ylim(0, 1.3)
ax.legend(frameon=False)
ax.grid(alpha=0.3)

fig.tight_layout()
out = os.path.join(HERE, "compare_norm_result.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
