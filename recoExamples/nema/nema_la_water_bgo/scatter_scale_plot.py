#!/usr/bin/env python3
"""Summarize the scatter-contamination scale ladder on the water NEMA phantom.
After the listmode normalization fix (contamination on the forward-model intensity
scale), a heuristic global multiplier on the scatter term was scanned: 1.0 / 1.2 /
1.5 / 2.0. Left: corrected CRC vs sphere diameter for each scale, against gold (the
trues-only ceiling) and clinical. Right: the operating point -- mean residual to gold
and 10 mm background variability vs scale. The residual closes ~uniformly and reaches
gold near scale 1.5 (with BV still below clinical); scale 2.0 over-subtracts.

  python3 scatter_scale_plot.py   (reads the bsrem_rdp_b05_isc[_scXX] npz in out/)
"""
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

# scale -> output tag (1.0 is the un-multiplied intensity-scaled run)
LADDER = [(1.0, "bsrem_rdp_b05_isc"), (1.2, "bsrem_rdp_b05_isc_sc12"),
          (1.5, "bsrem_rdp_b05_isc_sc15"), (2.0, "bsrem_rdp_b05_isc_sc20")]
CLIN_DIAM = np.array([10., 13., 17., 22., 28., 37.])
CLIN_CR = np.array([47.4, 59.3, 67.0, 77.0, 82.5, 85.1])
CLIN_BV = np.array([16.4, 12.1, 9.1, 6.6, 5.1, 3.8])

data = [(s, np.load(os.path.join(OUT, f"nema_la_water_bgo_{t}.npz"))) for s, t in LADDER]
diam = data[0][1]["diam_mm"]; order = np.argsort(diam); dia = diam[order]
ismall = int(np.argmin(diam))
colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(data)))

fig, (axc, axo) = plt.subplots(1, 2, figsize=(13, 5.2))

# left: corrected CRC vs diameter for each scale, + gold + clinical
axc.plot(dia, np.asarray(data[0][1]["crc_gold"])[order], "--", color="gray", lw=1.6,
         label="gold = trues, AC (ceiling)")
axc.plot(CLIN_DIAM, CLIN_CR, "k-D", ms=4, lw=1.2, label="clinical [ref]")
for (s, d), col in zip(data, colors):
    axc.plot(dia, np.asarray(d["crc_corr"])[order], "-o", ms=4, color=col, label=f"scale {s:.1f}")
axc.set_xlabel("sphere diameter (mm)"); axc.set_ylabel("corrected CRC (%)")
axc.set_title("scatter scale ladder: corrected CRC vs sphere size")
axc.grid(alpha=0.3); axc.legend(frameon=False, fontsize=8)

# right: operating point -- mean residual to gold and 10 mm BV vs scale
scales = np.array([s for s, _ in data])
resid = np.array([float(np.mean(np.asarray(d["crc_gold"]) - np.asarray(d["crc_corr"]))) for _, d in data])
bv10 = np.array([float(np.asarray(d["bv_corr"])[ismall]) for _, d in data])
axo.axhline(0, color="gray", lw=0.8, ls=":")
axo.plot(scales, resid, "-o", color="#1f77b4", label="mean (gold - corr) residual")
axo.set_xlabel("scatter scale factor"); axo.set_ylabel("mean residual to gold (CRC %)", color="#1f77b4")
axo.tick_params(axis="y", labelcolor="#1f77b4")
axo.set_title("operating point vs scale (residual -> 0 near 1.5)")
axo.grid(alpha=0.3)
axb = axo.twinx()
axb.plot(scales, bv10, "-s", color="#d62728", label="10 mm BV")
axb.axhline(CLIN_BV[0], color="#d62728", lw=0.8, ls="--", label="clinical 10 mm BV")
axb.set_ylabel("10 mm background variability (%)", color="#d62728")
axb.tick_params(axis="y", labelcolor="#d62728")
lines = axo.get_lines()[1:2] + axb.get_lines()
axo.legend(lines, [l.get_label() for l in lines], frameon=False, fontsize=8, loc="upper left")

figdir = os.path.join(HERE, "figures"); os.makedirs(figdir, exist_ok=True)
out = os.path.join(figdir, "nema_la_water_bgo_scatter_scale.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
