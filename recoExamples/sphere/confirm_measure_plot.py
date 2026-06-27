#!/usr/bin/env python3
"""Does the empirical/geometric ratio survive with efficiency removed from the
inputs? Overlay the ratio from REAL MC trues (efficiency in) and from IDEAL
events (sphere emission, geometry only, efficiency out). If they coincide, the
tilt is the LOR measure, not efficiency.

  python3 confirm_measure_plot.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "confirm_measure_results.npz"))
R = float(d["radius_mm"])

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(d["radii"], d["ratio_ideal"], "-o", ms=3, color="#9467bd",
        label="ideal events (efficiency OUT)")
if len(d["ratio_real"]):
    ax.plot(d["radii"], d["ratio_real"], "-o", ms=3, color="#2ca02c",
            label="real MC trues (efficiency IN)")
ax.axhline(1.0, color="k", lw=0.8, ls="--")
ax.axvline(R, color="gray", lw=0.8, ls=":")
ax.set_xlabel("radius (mm)"); ax.set_ylabel("empirical / geometric (normalized)")
ax.set_title("does the tilt survive with efficiency removed?")
ax.set_ylim(0.7, 1.3); ax.legend(frameon=False); ax.grid(alpha=0.3)

fig.tight_layout()
out = os.path.join(HERE, "confirm_measure_result.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
