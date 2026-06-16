#!/usr/bin/env python3
"""Generate figures/joseph_principle.png for the RecoCrysp tutorial (projection).

A 2D schematic of Joseph's method: a LOR crosses a voxel grid; the method steps
over the voxel planes perpendicular to the ray's principal axis, interpolates in
each plane, and scales the plane sum by the correction factor Delta/cos(theta).
Geometry is computed numerically so the crossings and weights are exact.

Run from this directory:  python3 joseph_principle.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arc

Nx, Ny = 7, 5
m = 0.35                       # ray slope; |m| < 1 so x is the principal axis
y0 = 1.15
cos_t = 1.0 / np.hypot(1.0, m)
theta_deg = np.degrees(np.arctan(m))

xs = np.arange(Nx)             # voxel planes (columns) at the voxel-centre x's
yc = y0 + m * xs               # ray crossing height at each plane

fig, ax = plt.subplots(figsize=(7.8, 4.8))

# voxel grid (cell boundaries) and centres
for gx in np.arange(-0.5, Nx, 1.0):
    ax.plot([gx, gx], [-0.5, Ny - 0.5], color="0.82", lw=0.8, zorder=1)
for gy in np.arange(-0.5, Ny, 1.0):
    ax.plot([-0.5, Nx - 0.5], [gy, gy], color="0.82", lw=0.8, zorder=1)
Xc, Yc = np.meshgrid(xs, np.arange(Ny))
ax.plot(Xc.ravel(), Yc.ravel(), "+", color="0.65", ms=5, zorder=2)

# voxel planes (perpendicular to the principal axis)
for x in xs:
    ax.plot([x, x], [-0.5, Ny - 0.5], color="C0", lw=0.6, ls="--", alpha=0.30, zorder=1)

# interpolation neighbours at one representative plane (column kc)
kc = 3
yk = yc[kc]
jlo = int(np.floor(yk)); jhi = jlo + 1
whi = yk - jlo; wlo = 1.0 - whi
for j in (jlo, jhi):
    ax.add_patch(Rectangle((kc - 0.5, j - 0.5), 1, 1, facecolor="#ffd9b3",
                           edgecolor="none", zorder=0))
ax.plot([kc, kc], [jlo, jhi], color="#cc7000", lw=1.3, ls=":", zorder=3)
ax.text(kc + 0.62, jlo, f"$w_-={wlo:.2f}$", va="center", ha="left", fontsize=9)
ax.text(kc + 0.62, jhi, f"$w_+={whi:.2f}$", va="center", ha="left", fontsize=9)

# the LOR
xr = np.array([-0.5, Nx - 0.5])
yr = y0 + m * xr
ax.annotate("", xy=(xr[1], yr[1]), xytext=(xr[0], yr[0]),
            arrowprops=dict(arrowstyle="-|>", color="C0", lw=2.2), zorder=4)
ax.text(xr[1] - 0.1, yr[1] + 0.22, "LOR", color="C0", fontsize=11, ha="right")

# ray-plane crossings
ax.plot(xs, yc, "o", color="C0", ms=5.5, zorder=5)

# angle theta at the x=1 crossing
xa, ya = xs[1], yc[1]
ax.plot([xa, xa + 1.5], [ya, ya], color="0.4", lw=1.0, ls="--", zorder=3)
ax.add_patch(Arc((xa, ya), 1.6, 1.6, angle=0, theta1=0, theta2=theta_deg,
                 color="0.3", lw=1.2, zorder=3))
ax.text(xa + 0.95, ya + 0.16, r"$\theta$", fontsize=11)

# correction-factor triangle between planes x=5 and x=6
x1, x2 = xs[5], xs[6]
yA, yB = yc[5], yc[6]
ax.plot([x1, x2], [yA, yA], color="#1a7d1a", lw=1.4, zorder=4)      # Delta
ax.plot([x2, x2], [yA, yB], color="#1a7d1a", lw=1.4, zorder=4)      # rise
sq = 0.12
ax.plot([x2 - sq, x2 - sq, x2], [yA, yA + sq, yA + sq], color="#1a7d1a", lw=0.9, zorder=4)
ax.text((x1 + x2) / 2, yA - 0.20, r"$\Delta$", color="#1a7d1a",
        ha="center", va="top", fontsize=10)
ax.text((x1 + x2) / 2 - 0.08, (yA + yB) / 2 + 0.18,
        r"$\Delta/\cos\theta$", color="#1a7d1a", ha="right", va="bottom", fontsize=10)

# principal-axis arrow
ax.annotate("", xy=(2.1, -1.05), xytext=(-0.5, -1.05),
            arrowprops=dict(arrowstyle="-|>", color="0.3", lw=1.4))
ax.text(0.8, -1.32, "principal axis", color="0.3", ha="center", fontsize=10)
ax.text(3.0, Ny - 0.32, "voxel planes", color="C0", fontsize=9, alpha=0.8, ha="center")

ax.set_xlim(-0.9, Nx)
ax.set_ylim(-1.6, Ny - 0.1)
ax.set_aspect("equal")
ax.axis("off")
fig.tight_layout()
fig.savefig("joseph_principle.png", dpi=150, bbox_inches="tight")
print("wrote joseph_principle.png")
