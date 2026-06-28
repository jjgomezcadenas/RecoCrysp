#!/usr/bin/env python3
"""The MLEM bias-variance tradeoff for NEMA, from scan_niter.jl. Left: the raw
components -- per-sphere contrast recovery (CRC, %) and background noise (CoV) vs
iteration; contrast converges (small spheres slowest), noise grows without bound.
Right: the combined figure of merit CRC/CoV (contrast-to-noise) per sphere, which
PEAKS at the useful stopping iteration -- past it, noise grows faster than any
remaining contrast gain.

  python3 scan_niter_plot.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "out", "nema_la_air_bgo_niter_scan.npz"))
it = d["iters"]
cov = d["cov"]
crc = d["crc"]            # (niter, n_spheres)
diam = d["diam_mm"]
cnr = crc / cov[:, None]  # CRC/CoV per sphere

fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.5))
cmap = plt.cm.viridis(np.linspace(0, 0.9, len(diam)))

# left: components
for s in range(len(diam)):
    axL.plot(it, crc[:, s], "-o", ms=3, color=cmap[s], label=f"{diam[s]:.0f} mm")
axL.axhline(100, color="gray", lw=0.8, ls=":")
axL.set_xlabel("MLEM iteration"); axL.set_ylabel("contrast recovery CRC (%)")
axL.grid(alpha=0.3); axL.legend(title="sphere", frameon=False, ncol=2, fontsize=8, loc="lower right")
axT = axL.twinx()
axT.plot(it, cov, "-s", ms=3, color="#d62728")
axT.set_ylabel("background CoV (noise)", color="#d62728"); axT.tick_params(axis="y", labelcolor="#d62728")
axL.set_title("components: CRC and noise vs iteration")

# right: CRC/CoV figure of merit
for s in range(len(diam)):
    axR.plot(it, cnr[:, s], "-o", ms=3, color=cmap[s], label=f"{diam[s]:.0f} mm")
    ipk = int(np.argmax(cnr[:, s]))
    axR.plot(it[ipk], cnr[ipk, s], "*", ms=11, color=cmap[s])
axR.set_xlabel("MLEM iteration"); axR.set_ylabel("CRC / CoV  (contrast-to-noise)")
axR.grid(alpha=0.3); axR.legend(title="sphere (★=peak)", frameon=False, ncol=2, fontsize=8)
axR.set_title("figure of merit: CRC/CoV vs iteration")

os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
out = os.path.join(HERE, "figures", "nema_la_air_bgo_niter_scan.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
