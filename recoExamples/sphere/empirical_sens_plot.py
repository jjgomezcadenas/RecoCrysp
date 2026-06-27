#!/usr/bin/env python3
"""Empirical vs geometric sensitivity (case a). The ratio (green) is the
efficiency shape the geometry assumes flat: if it slopes ~8%, the residual tilt
is efficiency; if it's flat, it isn't.

  python3 empirical_sens_plot.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "empirical_sens_results.npz"))
R = float(d["radius_mm"])

fig, ax = plt.subplots(1, 2, figsize=(13, 5))

ax[0].plot(d["radii"], d["prof_geom"], "-o", ms=3, color="#d62728", label="geometric  Aᵀ(1)")
ax[0].plot(d["radii"], d["prof_emp"], "-o", ms=3, color="#1f77b4", label="empirical  Aᵀ(eff)")
ax[0].set_title("sensitivity radial profiles (normalized)")
ax[0].set_xlabel("radius (mm)"); ax[0].set_ylabel("normalized sensitivity")
ax[0].legend(frameon=False); ax[0].grid(alpha=0.3)

ax[1].plot(d["radii"], d["prof_ratio"], "-o", ms=3, color="#2ca02c")
ax[1].axhline(1.0, color="k", lw=0.8, ls="--")
ax[1].axvline(R, color="gray", lw=0.8, ls=":")
ax[1].set_title("empirical / geometric  =  missing efficiency shape")
ax[1].set_xlabel("radius (mm)"); ax[1].set_ylabel("ratio (normalized)")
ax[1].set_ylim(0.7, 1.3); ax[1].grid(alpha=0.3)

fig.tight_layout()
out = os.path.join(HERE, "empirical_sens_result.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
