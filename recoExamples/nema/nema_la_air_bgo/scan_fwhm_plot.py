#!/usr/bin/env python3
"""Post-filter smoothness/contrast tradeoff, from scan_fwhm.jl (one fixed recon,
swept filter width). Left: the raw components -- per-sphere CRC and background CoV
vs FWHM; heavier filter smooths the background but lowers small-sphere CRC. Right:
the combined figure of merit CRC/CoV (contrast-to-noise) per sphere, which PEAKS
at the best filter width -- below it noise dominates, above it the filter eats the
contrast. The look is irrelevant; this peak is the selection criterion.

  python3 scan_fwhm_plot.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "out", "nema_la_air_bgo_fwhm_scan.npz"))
fwhm = d["fwhm_mm"]
cov = d["cov"]
crc = d["crc"]
diam = d["diam_mm"]
cnr = crc / cov[:, None]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.5))
cmap = plt.cm.viridis(np.linspace(0, 0.9, len(diam)))

for s in range(len(diam)):
    axL.plot(fwhm, crc[:, s], "-o", ms=3, color=cmap[s], label=f"{diam[s]:.0f} mm")
axL.axhline(100, color="gray", lw=0.8, ls=":")
axL.set_xlabel("post-filter FWHM (mm)"); axL.set_ylabel("contrast recovery CRC (%)")
axL.grid(alpha=0.3); axL.legend(title="sphere", frameon=False, ncol=2, fontsize=8, loc="center right")
axT = axL.twinx()
axT.plot(fwhm, cov, "-s", ms=4, color="#d62728")
axT.set_ylabel("background CoV (noise)", color="#d62728"); axT.tick_params(axis="y", labelcolor="#d62728")
axL.set_title("components: CRC and noise vs filter width")

for s in range(len(diam)):
    axR.plot(fwhm, cnr[:, s], "-o", ms=3, color=cmap[s], label=f"{diam[s]:.0f} mm")
    ipk = int(np.argmax(cnr[:, s]))
    axR.plot(fwhm[ipk], cnr[ipk, s], "*", ms=11, color=cmap[s])
axR.set_xlabel("post-filter FWHM (mm)"); axR.set_ylabel("CRC / CoV  (contrast-to-noise)")
axR.grid(alpha=0.3); axR.legend(title="sphere (★=peak)", frameon=False, ncol=2, fontsize=8)
axR.set_title("figure of merit: CRC/CoV vs filter width")

os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
out = os.path.join(HERE, "figures", "nema_la_air_bgo_fwhm_scan.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
