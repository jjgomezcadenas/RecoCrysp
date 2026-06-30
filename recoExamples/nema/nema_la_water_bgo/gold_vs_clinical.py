#!/usr/bin/env python3
"""Compare the GOLD reconstruction (true coincidences, attenuation-corrected) and
the CORRECTED reconstruction (prompts with AC+scatter+randoms) against the clinical
NEMA reference, per sphere diameter, for one or more variants.

The point of the comparison: "gold" is the best case our pipeline can do -- the true
coincidences, perfectly attenuation-corrected, reconstructed with the variant's own
method. If gold already matches the clinical CR curve, the residual gap to clinical
is NOT reconstruction-side; it is whatever separates `corr` from `gold` (the
scatter/randoms correction + attenuation modelling). This script prints, per
variant: diameter, GOLD CRC, corrected CRC, clinical CR, and the two gaps
(clinical-minus-gold, gold-minus-corr) so the dominant gap is explicit.

  python3 gold_vs_clinical.py [tag ...]   (default: all variants in out/)
"""
import os
import re
import sys
import glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

# clinical reference: GE Discovery MI 3-ring, NEMA NU2-2012 IQ, TOF-OSEM (no PSF)
# (Vandendriessche 2019). CR per sphere diameter (mm).
CLIN_DIAM = np.array([10., 13., 17., 22., 28., 37.])
CLIN_CR = np.array([47.4, 59.3, 67.0, 77.0, 82.5, 85.1])


def variant_files(args):
    if args:
        files = [os.path.join(OUT, f"nema_la_water_bgo_{t}.npz") for t in args]
    else:
        files = glob.glob(os.path.join(OUT, "nema_la_water_bgo_*.npz"))
    files = [f for f in files if "_cache_" not in os.path.basename(f)]

    def _nat(f):
        return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", f)]
    return sorted(files, key=_nat)


def clinical_on(diam):
    """Clinical CR interpolated onto the variant's sphere diameters."""
    return np.interp(diam, CLIN_DIAM, CLIN_CR)


def report(tag, d):
    diam = d["diam_mm"]; order = np.argsort(diam)
    dia = diam[order]
    gold = np.asarray(d["crc_gold"])[order]
    uncorr = np.asarray(d["crc_uncorr"])[order]
    corr = np.asarray(d["crc_corr"])[order]
    clin = clinical_on(dia)
    print(f"--- {tag} ---")
    print("  diam (mm)        :", "  ".join(f"{x:5.0f}" for x in dia))
    print("  GOLD CRC (%)     :", "  ".join(f"{x:5.0f}" for x in gold))
    print("  uncorr CRC (%)   :", "  ".join(f"{x:5.0f}" for x in uncorr))
    print("  corr CRC (%)     :", "  ".join(f"{x:5.0f}" for x in corr))
    print("  clinical CR (%)  :", "  ".join(f"{x:5.0f}" for x in clin))
    # decomposition of the gold -> corr loss
    print("  clinical - gold  :", "  ".join(f"{x:+5.0f}" for x in clin - gold),
          f"  (mean {np.mean(clin - gold):+.1f})  [reconstruction]")
    print("  gold - uncorr    :", "  ".join(f"{x:+5.0f}" for x in gold - uncorr),
          f"  (mean {np.mean(gold - uncorr):+.1f})  [raw contamination damage]")
    print("  corr - uncorr    :", "  ".join(f"{x:+5.0f}" for x in corr - uncorr),
          f"  (mean {np.mean(corr - uncorr):+.1f})  [recovered by our model]")
    print("  gold - corr      :", "  ".join(f"{x:+5.0f}" for x in gold - corr),
          f"  (mean {np.mean(gold - corr):+.1f})  [residual model fails to recover]")


def main():
    files = variant_files(sys.argv[1:])
    if not files:
        sys.exit("no variant npz found in out/ -- run compare_methods.jl first")
    for f in files:
        tag = os.path.basename(f)[len("nema_la_water_bgo_"):-len(".npz")]
        report(tag, np.load(f))
    print("\nReads: if (clinical - gold) ~ 0, gold matches clinical and the gap to")
    print("clinical is the (gold - corr) correction loss, not the reconstruction.")


if __name__ == "__main__":
    main()
