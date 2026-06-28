#!/usr/bin/env python3
"""Plot case (a): the uniform-sphere reconstruction and its flatness test.

Left: the true uniform sphere (central slice). Middle: the normalized MLEM
reconstruction (same slice) — for the vacuum dataset it should look flat; for
the water smoke-test it cups toward the centre. Right: the radial profile of the
normalized reconstruction, which should sit at 1.0 inside the sphere if the
normalization is correct.

  python3 plot.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "out", "air_bgo_100kBq.npz"))
ext = float(d["extent"])
R = float(d["radius_mm"])
extent = [-ext, ext, -ext, ext]

fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

for a, key, title in ((ax[0], "slice_true", "true uniform sphere"),
                      (ax[1], "slice_rec", "MLEM reconstruction (normalized)")):
    im = d[key].T
    a.imshow(im, origin="lower", extent=extent, cmap="magma", vmin=0, vmax=1.3)
    a.set_title(title)
    a.set_xlabel("x (mm)")
ax[0].set_ylabel("y (mm)")

axp = ax[2]
axp.plot(d["radii"], d["radial_prof"], "-o", ms=3, color="#1f77b4")
axp.axhline(1.0, color="k", lw=0.8, ls="--")
axp.axvline(R, color="gray", lw=0.8, ls=":")
axp.set_xlabel("radius (mm)")
axp.set_ylabel("normalized activity")
axp.set_title("radial profile (flat = good normalization)")
axp.set_ylim(0, 1.4)
axp.grid(alpha=0.3)

fig.tight_layout()
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
out = os.path.join(HERE, "figures", "air_bgo_100kBq.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
