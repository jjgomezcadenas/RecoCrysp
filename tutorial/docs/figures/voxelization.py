#!/usr/bin/env python3
"""Generate figures/voxelization.png for the RecoCrysp tutorial (grid section).

A continuous transverse activity distribution (left) and its representation on a
coarse voxel grid (right), to illustrate voxelization of the image. Pure
matplotlib; run from this directory:  python3 voxelization.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

FOV = 100.0          # half-width of the displayed domain, mm
N = 20               # voxels per side on the coarse grid
DX = 2 * FOV / N     # voxel size, mm


def phantom(x, y):
    """Two uniform disks: a large warm one and a smaller hot one."""
    f = np.where(x**2 + y**2 <= 70.0**2, 1.0, 0.0)
    f += np.where((x - 28.0) ** 2 + (y - 22.0) ** 2 <= 20.0**2, 1.0, 0.0)
    return f


# continuous field on a fine grid
fine = 800
g = np.linspace(-FOV, FOV, fine)
Xc, Yc = np.meshgrid(g, g)
cont = phantom(Xc, Yc)

# voxelized field: average the continuous field over each voxel (supersampling)
ss = fine // N
vox = cont.reshape(N, ss, N, ss).mean(axis=(1, 3))

fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.6, 4.4))
extent = [-FOV, FOV, -FOV, FOV]
kw = dict(extent=extent, origin="lower", cmap="magma", vmin=0, vmax=2)

axL.imshow(cont, **kw)
axL.set_title(r"continuous activity $f(\mathbf{u})$")

axR.imshow(vox, interpolation="nearest", **kw)
axR.set_title(r"voxelized image $\mathbf{x}$")
# draw voxel boundaries
for k in range(N + 1):
    axR.axvline(-FOV + k * DX, color="white", lw=0.4, alpha=0.5)
    axR.axhline(-FOV + k * DX, color="white", lw=0.4, alpha=0.5)

for ax in (axL, axR):
    ax.set_xlabel("x (mm)")
    ax.set_aspect("equal")
    ax.set_xticks([-100, -50, 0, 50, 100])
    ax.set_yticks([-100, -50, 0, 50, 100])
axL.set_ylabel("y (mm)")

# annotate one voxel size on the right panel
axR.annotate("", xy=(-FOV + DX, -FOV + 0.4 * DX), xytext=(-FOV, -FOV + 0.4 * DX),
             arrowprops=dict(arrowstyle="<->", color="white", lw=1.0))
axR.text(-FOV + 0.5 * DX, -FOV + 1.1 * DX, r"$\Delta$", color="white",
         ha="center", va="bottom", fontsize=11)

fig.tight_layout()
fig.savefig("voxelization.png", dpi=150, bbox_inches="tight")
print("wrote voxelization.png")
