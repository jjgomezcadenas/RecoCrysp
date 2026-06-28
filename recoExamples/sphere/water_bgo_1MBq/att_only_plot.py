#!/usr/bin/env python3
"""Water, attenuation only. Left/middle: central slice of the reconstructed water
sphere without and with attenuation correction. Right: radial profiles of the
two (each normalized to its own interior mean). Without correction the image
cups -- the centre suppressed relative to the rim by ~exp(-mu*2R); the correction
removes it, leaving the same LOR-measure tilt the air runs already showed.

Quantified by the diagnostics printed every run: the centre/edge ratio of each
profile, against the predicted exp(-mu*2R).

  python3 att_only_plot.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "out", "water_bgo_1MBq_att_only.npz"))
R = float(d["radius_mm"])
mu_cm = float(d["mu_per_cm"])
ext = float(d["extent"])
extent = [-ext, ext, -ext, ext]


def _report():
    r = d["radii"]
    pn, pa = d["prof_noac"], d["prof_ac"]
    cen = r < 20.0                       # central region
    edge = (r > R - 24.0) & (r < R - 4.0)   # rim, inside the sphere
    pred = np.exp(-(mu_cm / 10.0) * 2 * R)
    print("water_bgo_1MBq attenuation-only diagnostics")
    print(f"  mu = {mu_cm} /cm   predicted centre/edge = exp(-mu*2R) = {pred:.3f}")
    print(f"  no-AC  centre/edge = {pn[cen].mean() / pn[edge].mean():.3f}")
    print(f"  AC     centre/edge = {pa[cen].mean() / pa[edge].mean():.3f}")
    inside = r < R - 4.0
    print(f"  AC interior min/max (normalized) = "
          f"{pa[inside].min():.3f} / {pa[inside].max():.3f}")


_report()

fig = plt.figure(figsize=(15, 4.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.3], wspace=0.28)

vmax = np.percentile(d["slice_ac"], 99.5)
for j, (key, title) in enumerate((("slice_noac", "no attenuation correction"),
                                  ("slice_ac", "attenuation-corrected"))):
    ax = fig.add_subplot(gs[0, j])
    ax.imshow(d[key].T, origin="lower", extent=extent, cmap="magma", vmin=0, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("x (mm)")
    if j == 0:
        ax.set_ylabel("y (mm)")

axp = fig.add_subplot(gs[0, 2])
axp.plot(d["radii"], d["prof_noac"], "-s", ms=3, color="#d62728", label="no AC")
axp.plot(d["radii"], d["prof_ac"], "-^", ms=3, color="#1f77b4", label="AC")
axp.axhline(1.0, color="gray", lw=0.8, ls=":")
axp.axvline(R, color="gray", lw=0.8, ls=":")
axp.set_xlabel("radius (mm)"); axp.set_ylabel("normalized activity")
axp.set_title("radial profiles")
axp.legend(frameon=False); axp.grid(alpha=0.3)

os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
out = os.path.join(HERE, "figures", "water_bgo_1MBq_att_only.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
