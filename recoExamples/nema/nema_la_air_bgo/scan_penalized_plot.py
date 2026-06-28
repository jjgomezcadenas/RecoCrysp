#!/usr/bin/env python3
"""Conditioning test (from scan_penalized.jl): background noise (CoV) and the
10 mm-sphere CRC vs iteration, for a range of quadratic-smoothness-prior beta
(beta=0 == MLEM). Left: MLEM's CoV grows linearly without bound; as beta grows the
regularized CoV plateaus -- the noise was ill-conditioning, bounded inside the
algorithm. Right: the contrast cost -- higher beta trades small-sphere CRC for the
noise control (a quadratic prior behaves like a linear smoother, similar to the
post-filter; an edge-preserving prior would do better).

  python3 scan_penalized_plot.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "out", "nema_la_air_bgo_penalized_scan.npz"))
it = d["iters"]
betas = d["betas"]
cov = d["cov"]            # (nbeta, niter)
crc = d["crc"]            # (nbeta, niter, nsphere)
diam = d["diam_mm"]
i10 = int(np.argmin(diam))   # smallest sphere

fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.5))
cmap = plt.cm.plasma(np.linspace(0, 0.85, len(betas)))
for b in range(len(betas)):
    lab = "MLEM (β=0)" if betas[b] == 0 else f"β={betas[b]:.0f}"
    axL.plot(it, cov[b], "-o", ms=3, color=cmap[b], label=lab)
    axR.plot(it, crc[b, :, i10], "-o", ms=3, color=cmap[b], label=lab)

axL.set_xlabel("iteration"); axL.set_ylabel("background CoV (noise)")
axL.set_title("noise vs iteration: MLEM grows, prior bounds it")
axL.grid(alpha=0.3); axL.legend(frameon=False)

axR.axhline(100, color="gray", lw=0.8, ls=":")
axR.set_xlabel("iteration"); axR.set_ylabel(f"CRC, {diam[i10]:.0f} mm sphere (%)")
axR.set_title("contrast cost of regularization")
axR.grid(alpha=0.3); axR.legend(frameon=False)

os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
out = os.path.join(HERE, "figures", "nema_la_air_bgo_penalized_scan.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
