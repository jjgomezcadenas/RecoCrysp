#!/usr/bin/env python3
"""Plot the attenuation example results (run.jl -> attenuation_results.npz).

Produces attenuation_result.png: the central slice of the truth and the two
reconstructions (with / without attenuation correction), and a central profile
showing the cupping artifact when attenuation is ignored.

  python3 attenuation_plot.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "attenuation_results.npz"))
x_mm = d["x_mm"]
vmax = float(d["slice_true"].max()) * 1.15

fig, ax = plt.subplots(1, 4, figsize=(14, 3.6), gridspec_kw={"width_ratios": [1, 1, 1, 1.3]})
ext = [x_mm[0], x_mm[-1], x_mm[0], x_mm[-1]]
for a, key, title in [(ax[0], "slice_true", "truth"),
                      (ax[1], "slice_ac", "MLEM, with AC"),
                      (ax[2], "slice_no", "MLEM, no AC")]:
    im = a.imshow(d[key].T, origin="lower", extent=ext, cmap="magma", vmin=0, vmax=vmax)
    a.set_title(title)
    a.set_xlabel("x (mm)")
ax[0].set_ylabel("y (mm)")

ax[3].plot(x_mm, d["prof_true"], "k--", label="truth")
ax[3].plot(x_mm, d["prof_ac"], color="C0", label="with AC")
ax[3].plot(x_mm, d["prof_no"], color="C3", label="no AC")
ax[3].set_title("central profile")
ax[3].set_xlabel("x (mm)")
ax[3].set_ylabel("activity")
ax[3].legend(frameon=False, fontsize=9)

fig.tight_layout()
out = os.path.join(HERE, "attenuation_result.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
