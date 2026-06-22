#!/usr/bin/env python3
"""Plot the resolution example (run.jl -> resolution_results.npz).

Four central-slice panels: the sharp truth, the resolution-limited image
G*x_true, the MLEM reconstruction from sharp (G = 1) data, and the MLEM
reconstruction from smeared (G = fwhm) data. The sharp reconstruction recovers
the truth; the smeared one recovers G*x_true -- the difference is purely the
detector resolution.

  python3 resolution_plot.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "resolution_results.npz"))
ext = float(d["extent"])
extent = [-ext, ext, -ext, ext]

panels = [("slice_true", "truth (rods)"),
          ("slice_blur", r"resolution-limited  $G\,x$"),
          ("slice_rec_sharp", "MLEM from sharp data"),
          ("slice_rec_blur", "MLEM from smeared data")]

fig, ax = plt.subplots(1, 4, figsize=(16.5, 4.3))
for a, (key, title) in zip(ax, panels):
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
