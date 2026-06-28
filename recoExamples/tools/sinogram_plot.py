#!/usr/bin/env python3
"""Inspect a background-model sinogram (from sinogram_dump.jl). Generic over the
background class (scatter or randoms) and phantom. Collapsing the obliquity axis,
shows the prompt sinogram P, the background sinogram B, the raw background
fraction B/P, and the smoothed fraction B~/P~ that the model actually applies --
all in (radial offset s_r, axial midpoint z_m). The dotted line marks the edge
radius (e.g. the phantom body): beyond it few/no true LORs land, so the
background fraction should rise toward 1. The bottom panel is the fraction vs s_r.

The printed diagnostics give the background fraction inside vs outside the edge --
the test of whether the model is strong enough where the background dominates.

  python3 sinogram_plot.py <scenario>/out/<tag>_sinogram.npz [class]
  (writes the figure to the sibling <scenario>/figures/)
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) not in (2, 3):
    sys.exit("usage: sinogram_plot.py <tag>_sinogram.npz [class]")
NPZ = os.path.abspath(sys.argv[1])
cls = sys.argv[2] if len(sys.argv) == 3 else "background"
d = np.load(NPZ)
R = float(d["radius_mm"])
sr0, sr1 = map(float, d["span_sr"])
zm0, zm1 = map(float, d["span_zm"])
S, P, Ssm, Psm = d["S"], d["P"], d["Ssm"], d["Psm"]
n_sr = S.shape[0]
sr_cen = sr0 + (np.arange(n_sr) + 0.5) * (sr1 - sr0) / n_sr


def _frac(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b > 0)


P2, S2 = P.sum(2), S.sum(2)
fr_raw = _frac(S2, P2)
fr_sm = _frac(Ssm.sum(2), Psm.sum(2))
Sr, Pr = Ssm.sum((1, 2)), Psm.sum((1, 2))
fr_sr = _frac(Sr, Pr)
inside, outside = sr_cen < R, sr_cen > R


def _report():
    sm = d["smooth"]
    fi = Sr[inside].sum() / max(Pr[inside].sum(), 1)
    fo = Sr[outside].sum() / max(Pr[outside].sum(), 1)
    print(f"{cls}-model sinogram diagnostics")
    print(f"  s_r span = [{sr0:.1f}, {sr1:.1f}] mm,  edge R = {R:.0f} mm,  "
          f"smooth (s_r,z_m,dz) = ({sm[0]:.1f},{sm[1]:.1f},{sm[2]:.1f}) bins")
    print(f"  {cls} fraction  s_r < R = {fi:.3f}")
    print(f"  {cls} fraction  s_r > R = {fo:.3f}   (should approach 1 if the halo is pure {cls})")
    print(f"  prompts with s_r > R = {Pr[outside].sum() / Pr.sum():.3%} of all")


_report()

ext = [zm0, zm1, sr0, sr1]
fig = plt.figure(figsize=(13, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0], hspace=0.3, wspace=0.28)
panels = [(P2, "prompts P", "viridis"), (S2, f"{cls} B", "magma")]
for ax_pos, (arr, title, cmap) in zip([(0, 0), (0, 1)], panels):
    ax = fig.add_subplot(gs[ax_pos])
    im = ax.imshow(arr, origin="lower", extent=ext, aspect="auto", cmap=cmap)
    ax.axhline(R, color="w", lw=0.8, ls=":")
    ax.set_title(title); ax.set_xlabel("z_m (mm)"); ax.set_ylabel("s_r (mm)")
    fig.colorbar(im, ax=ax, fraction=0.046)

axf = fig.add_subplot(gs[1, 0])
im = axf.imshow(fr_sm, origin="lower", extent=ext, aspect="auto", cmap="inferno", vmin=0, vmax=1)
axf.axhline(R, color="w", lw=0.8, ls=":")
axf.set_title(f"smoothed fraction B~/P~ (model)"); axf.set_xlabel("z_m (mm)"); axf.set_ylabel("s_r (mm)")
fig.colorbar(im, ax=axf, fraction=0.046)

axr = fig.add_subplot(gs[1, 1])
axr.plot(sr_cen, _frac(S2.sum(1), P2.sum(1)), "-o", ms=3, color="0.6", label="raw")
axr.plot(sr_cen, fr_sr, "-^", ms=3, color="#1f77b4", label="smoothed (model)")
axr.axvline(R, color="gray", lw=0.8, ls=":")
axr.axhline(1.0, color="gray", lw=0.6, ls=":")
axr.set_xlabel("s_r (mm)"); axr.set_ylabel(f"{cls} fraction")
axr.set_title(f"{cls} fraction vs radial offset")
axr.legend(frameon=False); axr.grid(alpha=0.3)

# npz lives in <scenario>/out/; write the figure to the sibling <scenario>/figures/
figdir = os.path.join(os.path.dirname(os.path.dirname(NPZ)), "figures")
os.makedirs(figdir, exist_ok=True)
out = os.path.join(figdir, os.path.basename(NPZ).replace(".npz", ".png"))
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
