#!/usr/bin/env python3
"""Background noise of a reconstructed central slice: mean, std, and coefficient
of variation (CoV = std/mean) over a uniform annular ROI. CoV is the per-voxel
noise -- the speckle that grows with MLEM/OSEM iterations and shrinks with larger
voxels / a post-filter. Generic over any scenario npz that stores a 2D slice and
the transverse half-extent.

  python3 recon_noise.py <npz> [slice_key=slice_gold] [r_in_mm=10] [r_out_mm=25]

The ROI is the annulus r_in < r < r_out (mm) in the slice plane -- pick a
uniform, structure-free region (e.g. background away from hot features).
"""
import sys
import numpy as np

if not 2 <= len(sys.argv) <= 5:
    sys.exit(__doc__)
npz = sys.argv[1]
slice_key = sys.argv[2] if len(sys.argv) > 2 else "slice_gold"
r_in = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
r_out = float(sys.argv[4]) if len(sys.argv) > 4 else 25.0

d = np.load(npz)
if slice_key not in d:
    sys.exit(f"no '{slice_key}' in {npz}; available: {[k for k in d.files if k.startswith('slice')]}")
img = d[slice_key]
ext = float(d["extent_xy"]) if "extent_xy" in d else float(d["extent"])
n = img.shape[0]
ax = np.linspace(-ext, ext, n)
X, Y = np.meshgrid(ax, ax, indexing="ij")
r = np.hypot(X, Y)
roi = (r > r_in) & (r < r_out)
m = img[roi]

print(f"recon noise: {npz}  [{slice_key}]")
print(f"  ROI annulus {r_in:.0f} < r < {r_out:.0f} mm  ({roi.sum()} voxels)")
print(f"  mean {m.mean():.4f}   std {m.std():.4f}   CoV {m.std() / m.mean():.2f}")
