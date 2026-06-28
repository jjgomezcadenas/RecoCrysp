#!/usr/bin/env python3
"""Find the CRC/CoV (contrast-to-noise) optimum of a parameter scan. Reads a scan
npz holding a per-sphere CRC matrix (`crc`, shape (nsteps, nspheres)), a
background-noise vector (`cov`), sphere diameters (`diam_mm`), and a swept-axis
vector (`iters` for an iteration scan or `fwhm_mm` for a post-filter scan). Reports
the swept-axis value that maximizes CRC/CoV for each sphere -- the operating point
that maximizes detectability, independent of the visual look.

  python3 scan_optimum.py <scan.npz>
"""
import os
import sys
import numpy as np

if len(sys.argv) != 2:
    sys.exit(__doc__)
npz = os.path.abspath(sys.argv[1])
d = np.load(npz)

xkey = next((k for k in ("iters", "fwhm_mm") if k in d), None)
if xkey is None:
    sys.exit(f"no 'iters' or 'fwhm_mm' axis in {npz}")
x = d[xkey]
crc = d["crc"]
cov = d["cov"]
diam = d["diam_mm"]
cnr = crc / cov[:, None]

print(f"CRC/CoV optimum of {os.path.basename(npz)}  (axis: {xkey})")
print(f"  {'diam':>5}  {xkey+'*':>8}  {'CRC%':>6}  {'CoV':>6}  {'CRC/CoV':>8}")
peaks = []
for s in range(len(diam)):
    i = int(np.argmax(cnr[:, s]))
    peaks.append(x[i])
    print(f"  {diam[s]:5.0f}  {x[i]:8.0f}  {crc[i, s]:6.1f}  {cov[i]:6.3f}  {cnr[i, s]:8.0f}")
# a single operating point: the value best for the SMALLEST (hardest) sphere
ismall = int(np.argmin(diam))
ibest = int(np.argmax(cnr[:, ismall]))
print(f"  -> best for the {diam[ismall]:.0f} mm (hardest) sphere: {xkey} = {x[ibest]:.0f}")
print(f"     median peak across spheres: {xkey} = {np.median(peaks):.0f}")
