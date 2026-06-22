#!/usr/bin/env python3
"""Plot the resolution example (run.jl -> resolution_results.npz).

Central transverse slice of the Derenzo phantom: the sharp truth, the
resolution-limited image G*x_true (what is recoverable), and the MLEM
reconstruction. Coarse rod sectors resolve; fine ones merge.

  python3 resolution_plot.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "resolution_results.npz"))
ext = float(d["extent"])
extent = [-ext, ext, -ext, ext]

fig, ax = plt.subplots(1, 3, figsize=(12.5, 4.3))
for a, key, title in [(ax[0], "slice_true", "truth (rods)"),
                      (ax[1], "slice_blur", r"resolution-limited  $G\,x$"),
                      (ax[2], "slice_rec", "MLEM (attenuation-corrected)")]:
    im = d[key].T
    vmax = np.percentile(im, 99.7) if im.max() > 0 else 1.0
    a.imshow(im, origin="lower", extent=extent, cmap="magma", vmin=0, vmax=vmax)
    a.set_title(title)
    a.set_xlabel("x (mm)")
ax[0].set_ylabel("y (mm)")

fig.tight_layout()
out = os.path.join(HERE, "resolution_result.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
