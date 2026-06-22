#!/usr/bin/env python3
"""Plot the OSEM example (run.jl -> osem_results.npz).

Top row: three central-slice images -- the truth, the MLEM reconstruction, and
the OSEM reconstruction (m_show subsets) at the SAME number of image updates;
they are visually the same. Bottom row: convergence of the error against the
truth, plotted twice -- against the image-update count (MLEM and OSEM overlay:
one OSEM update ~ one MLEM iteration, and the error is U-shaped: semi-
convergence) and against wall-clock solve time (OSEM-M reaches the same image
about M times sooner).

  python3 osem_plot.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "osem_results.npz"))
ext = float(d["extent"])
extent = [-ext, ext, -ext, ext]
subsets = [int(m) for m in d["subsets"]]
m_show = int(d["m_show"])

fig = plt.figure(figsize=(15, 8.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0], hspace=0.32, wspace=0.25)

# --- top row: images ------------------------------------------------------------
imgs = [("slice_truth", "truth (rods)"),
        ("slice_mlem", f"MLEM, {int(d['mlem_upd'][-1])} updates"),
        ("slice_osem", f"OSEM ({m_show} subsets), same updates")]
for j, (key, title) in enumerate(imgs):
    ax = fig.add_subplot(gs[0, j])
    im = d[key].T
    vmax = np.percentile(im, 99.7) if im.max() > 0 else 1.0
    ax.imshow(im, origin="lower", extent=extent, cmap="magma", vmin=0, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("x (mm)")
    if j == 0:
        ax.set_ylabel("y (mm)")

# --- bottom row: convergence ----------------------------------------------------
colors = {m: c for m, c in zip(subsets, ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"])}

ax_u = fig.add_subplot(gs[1, 0:2])      # error vs image updates (wide)
ax_t = fig.add_subplot(gs[1, 2])        # error vs wall-clock

ax_u.plot(d["mlem_upd"], d["mlem_rer"], "k-o", ms=3, lw=1.5, label="MLEM")
ax_t.plot(d["mlem_tim"], d["mlem_rer"], "k-o", ms=3, lw=1.5, label="MLEM")
for m in subsets:
    ax_u.plot(d[f"osem{m}_upd"], d[f"osem{m}_rer"], "-s", ms=3, color=colors[m],
              label=f"OSEM {m}")
    ax_t.plot(d[f"osem{m}_tim"], d[f"osem{m}_rer"], "-s", ms=3, color=colors[m],
              label=f"OSEM {m}")

ax_u.set_xlabel("image updates")
ax_u.set_ylabel("error vs truth")
ax_u.set_title("vs image updates: one OSEM update ~ one MLEM iteration")
ax_u.legend(frameon=False, ncol=2, fontsize=9)
ax_u.grid(alpha=0.3)

ax_t.set_xlabel("solve time (s)")
ax_t.set_title("vs wall-clock: OSEM is faster")
ax_t.grid(alpha=0.3)

fig.suptitle("Act 1 - sharp data: MLEM vs OSEM (only the solver changes)",
             y=0.98, fontsize=13)
out = os.path.join(HERE, "osem_result.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)

# === Act 2: smeared data — semi-convergence and early stopping ==================
# Three images (the recoverable G*x, the early-stop minimum, the overfit final
# iterate) and the error-vs-updates curve with the minima marked. The error
# against G*x dips and climbs: stop at the bottom.
fig2 = plt.figure(figsize=(15, 4.6))
gs2 = fig2.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.5], wspace=0.28)

i_min = int(np.argmin(d["mlem2_rer"]))
imgs2 = [("slice_blur", r"recoverable  $G\,x$"),
         ("slice_mlem2_min", f"MLEM early stop ({int(d['mlem2_upd'][i_min])} updates)"),
         ("slice_mlem2_end", f"MLEM run on ({int(d['mlem2_upd'][-1])} updates)")]
for j, (key, title) in enumerate(imgs2):
    ax = fig2.add_subplot(gs2[0, j])
    im = d[key].T
    vmax = np.percentile(d["slice_blur"].T, 99.7)
    ax.imshow(im, origin="lower", extent=extent, cmap="magma", vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (mm)")
    if j == 0:
        ax.set_ylabel("y (mm)")

axc = fig2.add_subplot(gs2[0, 3])
axc.plot(d["mlem2_upd"], d["mlem2_rer"], "k-o", ms=3, lw=1.5, label="MLEM")
axc.plot(d["osem2_upd"], d["osem2_rer"], "-s", ms=3, color="#2ca02c",
         label=f"OSEM {m_show}")
# mark each solver's minimum (the early-stopping point)
jm = int(np.argmin(d["mlem2_rer"]))
jo = int(np.argmin(d["osem2_rer"]))
axc.plot(d["mlem2_upd"][jm], d["mlem2_rer"][jm], "ko", ms=9, mfc="none", mew=1.8)
axc.plot(d["osem2_upd"][jo], d["osem2_rer"][jo], "o", ms=9, mfc="none", mew=1.8,
         color="#2ca02c")
axc.annotate("stop here", (d["mlem2_upd"][jm], d["mlem2_rer"][jm]),
             textcoords="offset points", xytext=(12, -2), fontsize=9)
axc.set_xlabel("image updates")
axc.set_ylabel(r"error vs $G\,x$")
axc.set_title("error dips, then climbs (overfitting)")
axc.legend(frameon=False)
axc.grid(alpha=0.3)

fig2.suptitle(r"Act 2 - smeared data: semi-convergence and early stopping",
              y=1.0, fontsize=13)
out2 = os.path.join(HERE, "osem_earlystop_result.png")
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print("wrote", out2)
