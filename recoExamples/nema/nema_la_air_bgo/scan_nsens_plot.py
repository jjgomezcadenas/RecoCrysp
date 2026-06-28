#!/usr/bin/env python3
"""Source of the background noise (from scan_nsens.jl): background CoV and the
sensitivity-image CoV vs the number of sensitivity LORs, everything else fixed.
Both fall as 1/sqrt(nsens) and the background tracks the sens CoV at a fixed ~5x
amplification (the ill-conditioning). So the background mottle is the Monte-Carlo
sampling noise of the sensitivity image Aᵀ(1) -- under-sampled at 20M LORs on this
fine grid -- amplified by MLEM. It is fixable by sampling sens better (cheap).

  python3 scan_nsens_plot.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "out", "nema_la_air_bgo_nsens_scan.npz"))
ns = d["nsens"].astype(float)
cov = d["cov"]
scov = d["sens_cov"]

fig, ax = plt.subplots(figsize=(8.5, 6))
ax.loglog(ns, cov, "-o", ms=8, color="#1f77b4", label="background CoV")
ax.loglog(ns, scov, "-s", ms=6, color="#888888", label="sensitivity CoV")
ref = cov[0] * np.sqrt(ns[0] / ns)
ax.loglog(ns, ref, "--", color="k", label=r"$1/\sqrt{n_\mathrm{sens}}$ (anchored at min)")
ax.set_xlabel("number of sensitivity LORs"); ax.set_ylabel("CoV")
ax.set_title(f"background noise is the sensitivity sampling (niter={int(d['niter'])} + {float(d['fwhm_mm']):.0f}mm)\n"
             "bg CoV ~ 5x sens CoV, both ~1/sqrt(nsens) -> fixable with more LORs")
ax.grid(alpha=0.3, which="both"); ax.legend(frameon=False)
for x, y in zip(ns, cov):
    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 9), fontsize=8, ha="center")

os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
out = os.path.join(HERE, "figures", "nema_la_air_bgo_nsens_scan.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
