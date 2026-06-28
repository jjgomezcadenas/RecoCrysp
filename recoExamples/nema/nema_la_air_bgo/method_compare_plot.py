#!/usr/bin/env python3
"""Compare regularization methods on the same axis: small-sphere contrast vs
background noise. For each method we trace the (CoV, CRC) frontier OVER iterations
at a fixed beta -- as iterations proceed both grow, sweeping a curve; the method
whose curve sits highest (more CRC at a given CoV) is better. Metric: mean CRC of
the three smallest spheres (10/13/17 mm). Reads the penalized (quadratic) and
huber (OSL) scans. Shows whether the edge-preserving Huber prior beats the
quadratic prior / MLEM in the stable beta regime.

  python3 method_compare_plot.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
dq = np.load(os.path.join(HERE, "out", "nema_la_air_bgo_penalized_scan.npz"))
dh = np.load(os.path.join(HERE, "out", "nema_la_air_bgo_huber_scan.npz"))
df = np.load(os.path.join(HERE, "out", "nema_la_air_bgo_fwhm_scan.npz"))  # MLEM + post-filter
diam = dq["diam_mm"]
small = np.argsort(diam)[:3]                 # 10, 13, 17 mm


def trace(d, beta):
    bi = int(np.argmin(np.abs(d["betas"] - beta)))
    cov = d["cov"][bi]
    crc = d["crc"][bi][:, small].mean(axis=1)  # mean over the 3 smallest spheres
    return cov, crc


def postfilter_trace(d):
    # MLEM + Gaussian post-filter, swept over FWHM (same CoV metric)
    return d["cov"], d["crc"][:, small].mean(axis=1)


fig, ax = plt.subplots(figsize=(8.5, 6))
curves = [
    (dq, 0.0,    "MLEM (β=0)",           "k",       "-o"),
    (dq, 300.0,  "quadratic β=300",      "#1f77b4", "-s"),
    (dq, 1000.0, "quadratic β=1000",     "#5fa2dd", "-s"),
    (dh, 300.0,  "Huber β=300",          "#d62728", "-^"),
    (dh, 1000.0, "Huber β=1000",         "#ff9896", "-^"),
]
for d, beta, lab, col, style in curves:
    cov, crc = trace(d, beta)
    ax.plot(cov, crc, style, ms=3, color=col, label=lab)
# MLEM + post-filter, swept over FWHM (the honest comparison the iterative
# traces alone omit) -- post-filtering moves MLEM off its unfiltered trace
covf, crcf = postfilter_trace(df)
ax.plot(covf, crcf, "-D", ms=4, color="#2ca02c", label="MLEM + post-filter (FWHM swept)")

ax.set_xlabel("background CoV (noise)")
ax.set_ylabel("mean CRC, 10/13/17 mm spheres (%)")
ax.set_title("contrast vs noise frontier: post-filter ≈ Huber > quadratic ≈ MLEM\n"
             "(edge-preservation gives no clear gain for these low-contrast small spheres)")
ax.grid(alpha=0.3)
ax.legend(frameon=False)
ax.set_xlim(left=0)

out = os.path.join(HERE, "figures", "nema_la_air_bgo_method_compare.png")
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
