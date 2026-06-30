#!/usr/bin/env python3
"""Overlay every reconstructed method variant on the water NEMA phantom -- the
decision plot. Reads all out/nema_la_water_bgo_<tag>.npz written by
compare_methods.jl. Left: corrected-CRC vs sphere diameter, one curve per method
(gold shown once as the dashed target). Right: the frontier -- smallest-sphere CRC
vs background CoV (corrected), one point per method (up-and-left is better).

  python3 compare_plot.py [--name LABEL] [tag ...]
    --name LABEL  -> figures/nema_la_water_bgo_compare_<LABEL>.png (documents which
                    comparison this is; without it, ..._compare.png).
    tag ...       -> which variants to overlay (default: all variants in out/).
"""
import os
import re
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

args = sys.argv[1:]
name = ""
if "--name" in args:
    i = args.index("--name"); name = args[i + 1]; del args[i:i + 2]

if args:
    files = [os.path.join(OUT, f"nema_la_water_bgo_{t}.npz") for t in args]
else:
    files = glob.glob(os.path.join(OUT, "nema_la_water_bgo_*.npz"))
files = [f for f in files if "_cache_" not in os.path.basename(f)]
# natural sort (so quad_b30 precedes quad_b100, etc.)
def _nat(f): return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", f)]
files = sorted(files, key=_nat)
if not files:
    sys.exit("no variant npz found in out/ -- run compare_methods.jl first")

tags = [os.path.basename(f)[len("nema_la_water_bgo_"):-len(".npz")] for f in files]
variants = [np.load(f) for f in files]
diam = variants[0]["diam_mm"]; order = np.argsort(diam)
colors = plt.cm.tab10(np.linspace(0, 1, max(len(variants), 3)))

# clinical reference: GE Discovery MI 3-ring, NEMA NU2-2012 IQ, TOF-OSEM (no PSF)
# (Vandendriessche 2019). CR and BV per sphere diameter (28/37mm were cold in 2012).
CLIN_DIAM = np.array([10., 13., 17., 22., 28., 37.])
CLIN_CR   = np.array([47.4, 59.3, 67.0, 77.0, 82.5, 85.1])
CLIN_BV   = np.array([16.4, 12.1, 9.1, 6.6, 5.1, 3.8])

fig, (axc, axf) = plt.subplots(1, 2, figsize=(14, 6))

# left: corrected CRC vs diameter, per method; gold target + clinical reference
axc.plot(diam[order], np.asarray(variants[0]["crc_gold"])[order], "--", color="gray",
         lw=1.5, label="gold = trues, AC (best case / upper bound)")
axc.plot(CLIN_DIAM, CLIN_CR, "k-D", ms=5, lw=1.5, label="clinical TOF-OSEM [ref]")
for tag, d, col in zip(tags, variants, colors):
    axc.plot(diam[order], np.asarray(d["crc_corr"])[order], "-o", ms=5, color=col, label=tag)
axc.axhline(100, color="gray", lw=0.6, ls=":")
axc.set_xlabel("sphere diameter (mm)"); axc.set_ylabel("corrected CRC (%)")
axc.set_title("contrast recovery vs sphere size (corrected)")
axc.grid(alpha=0.3); axc.legend(frameon=False, fontsize=8)

# right: NEMA frontier -- small-sphere CRC vs NEMA background variability (both %).
# clinical point (10mm) marked; clinical sits up-and-right (high CR, high BV).
ismall = int(np.argmin(diam))
axf.scatter(CLIN_BV[0], CLIN_CR[0], s=160, marker="*", color="k", zorder=4,
            label="clinical 10mm [ref]")
for tag, d, col in zip(tags, variants, colors):
    x = float(np.asarray(d["bv_corr"])[ismall]); y = float(np.asarray(d["crc_corr"])[ismall])
    axf.scatter(x, y, s=70, color=col, zorder=3)
    axf.annotate(tag, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
axf.set_xlabel(f"NEMA background variability of {int(diam[ismall])} mm sphere (%)")
axf.set_ylabel(f"CRC of {int(diam[ismall])} mm sphere (%)")
axf.set_title("NEMA frontier: small-sphere contrast vs variability\n(clinical = up-and-right)")
axf.grid(alpha=0.3); axf.legend(frameon=False, fontsize=8)

figdir = os.path.join(HERE, "figures"); os.makedirs(figdir, exist_ok=True)
stem = "nema_la_water_bgo_compare" + (f"_{name}" if name else "")
out = os.path.join(figdir, f"{stem}.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out, "  variants:", ", ".join(tags))
