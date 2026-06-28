#!/usr/bin/env python3
"""Noise vs counts (from scan_counts.jl): residual background CoV vs the number of
true coincidences N, at FIXED regularization (MLEM niter + fixed post-filter). The
dashed line is the 1/sqrt(N) Poisson expectation anchored at the largest N. If the
data tracked it, the floor would be statistics-limited; a flat measured curve says
the floor is set by the reconstruction (ill-conditioning), not by counts.

  python3 scan_counts_plot.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "out", "nema_la_air_bgo_counts_scan.npz"))
N = d["N"].astype(float)
cov = d["cov"]

fig, ax = plt.subplots(figsize=(8, 6))
ax.loglog(N, cov, "-o", ms=7, color="#1f77b4", label="measured residual CoV")
# 1/sqrt(N) reference anchored at the largest N
poisson = cov[-1] * np.sqrt(N[-1] / N)
ax.loglog(N, poisson, "--", color="k", label=r"Poisson $1/\sqrt{N}$ (anchored at max N)")

ax.set_xlabel("number of true coincidences N")
ax.set_ylabel("residual background CoV")
ax.set_title(f"noise vs counts (MLEM niter={int(d['niter'])} + {float(d['fwhm_mm']):.0f} mm filter)\n"
             "flat measured curve -> floor is the reconstruction, not statistics")
ax.grid(alpha=0.3, which="both")
ax.legend(frameon=False)
for x, y in zip(N, cov):
    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")

os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
out = os.path.join(HERE, "figures", "nema_la_air_bgo_counts_scan.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
