"""Generate reference sinogram LOR coordinates from the Python parallelproj
geometry code, for cross-validation against the Julia port.

Uses only the pure-Python geometry modules (pet_scanners, pet_lors) with the
numpy array-API backend; the compiled parallelproj_core is NOT needed. The
local parallelproj source is added to sys.path so nothing has to be installed.

Output: reference_lors.npz with, for each (symmetry_axis, sinogram_order) case,
the flattened LOR start/end coordinates (C-order over the spatial sinogram
axes) plus the scalar geometry parameters, so the Julia side can rebuild the
identical scanner.
"""

import sys
from pathlib import Path

# point at the local parallelproj clone (sibling of RecoCrysp)
PP_SRC = Path(__file__).resolve().parents[3] / "parallelproj" / "src"
sys.path.insert(0, str(PP_SRC))

import numpy as np
import array_api_compat.numpy as xp

import parallelproj.pet_scanners as pps
import parallelproj.pet_lors as ppl

dev = "cpu"

# small scanner so the arrays stay tiny and the comparison is fast
radius = 0.5 * (744.1 + 2 * 8.51)
num_sides = 8
num_lor_endpoints_per_side = 4
lor_spacing = 4.03125
num_rings = 5
radial_trim = 2

ring_positions = 5.31556 * xp.arange(num_rings, device=dev, dtype=xp.float32) + (
    xp.astype(xp.arange(num_rings, device=dev) // 9, xp.float32)
) * 2.8
ring_positions -= 0.5 * xp.max(ring_positions)

sinogram_orders = ["PVR", "PRV", "VPR", "VRP", "RPV", "RVP"]
symmetry_axes = [0, 1, 2]

out = {}
out["radius"] = np.float64(radius)
out["num_sides"] = np.int64(num_sides)
out["num_lor_endpoints_per_side"] = np.int64(num_lor_endpoints_per_side)
out["lor_spacing"] = np.float64(lor_spacing)
out["num_rings"] = np.int64(num_rings)
out["radial_trim"] = np.int64(radial_trim)
out["ring_positions"] = np.asarray(ring_positions, dtype=np.float64)

for sym in symmetry_axes:
    scanner = pps.RegularPolygonPETScannerGeometry(
        xp,
        dev,
        radius=radius,
        num_sides=num_sides,
        num_lor_endpoints_per_side=num_lor_endpoints_per_side,
        lor_spacing=lor_spacing,
        ring_positions=ring_positions,
        symmetry_axis=sym,
    )
    for order in sinogram_orders:
        desc = ppl.RegularPolygonPETLORDescriptor(
            scanner,
            radial_trim=radial_trim,
            sinogram_order=ppl.SinogramSpatialAxisOrder[order],
        )
        # all views
        xstart, xend = desc.get_lor_coordinates()
        # C-order flatten over the spatial sinogram axes -> (nlors, 3)
        xs = np.asarray(xstart, dtype=np.float64).reshape(-1, 3)
        xe = np.asarray(xend, dtype=np.float64).reshape(-1, 3)
        key = f"sym{sym}_{order}"
        out[f"{key}__xstart"] = xs
        out[f"{key}__xend"] = xe
        out[f"{key}__shape"] = np.asarray(desc.spatial_sinogram_shape, dtype=np.int64)
        print(
            f"sym={sym} {order}: spatial shape {desc.spatial_sinogram_shape}, "
            f"{xs.shape[0]} LORs"
        )

out_path = Path(__file__).resolve().parent / "reference_lors.npz"
np.savez(out_path, **out)
print(f"\nwrote {out_path}")
