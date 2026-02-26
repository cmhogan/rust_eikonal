#!/usr/bin/env python3
"""
Generate a scikit-fmm first-order reference traveltime for the Marmousi model.

This script produces test_fixtures/marmousi_skfmm_ref.npy, which is used by
the Rust integration test `marmousi_vs_skfmm` to validate the FIM solver.

Requirements: numpy, scikit-fmm
Usage:        python3 test_fixtures/generate_marmousi_reference.py
"""

import sys
from pathlib import Path

import numpy as np

try:
    import skfmm
except ImportError:
    sys.exit("scikit-fmm is not installed. Install with: pip install scikit-fmm")

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "test_fixtures"
FIXTURES.mkdir(exist_ok=True)

# --- Load Marmousi velocity model -------------------------------------------
npy_path = ROOT / "marmousi_vel.npy"

if not npy_path.exists():
    sys.exit(f"marmousi_vel.npy not found at {npy_path}")

vel = np.load(npy_path)
assert vel.shape == (601, 1881), f"Unexpected shape: {vel.shape}"
assert vel.flags["C_CONTIGUOUS"], "Expected C-contiguous array"

dx = 5.0  # metres
nz, nx = vel.shape

# --- Compute skfmm reference ------------------------------------------------
# Source at grid index [0, 940] => coordinate (z=0, x=4700 m).
# Build a signed-distance level set seeded at the source.
src_i, src_j = 0, 940
Z, X = np.mgrid[0:nz, 0:nx]
phi = np.sqrt(((Z - src_i) ** 2 + (X - src_j) ** 2).astype(float)) * dx
phi[src_i, src_j] = -dx  # negative = inside the source region

tt_ref = skfmm.travel_time(phi, vel, dx=dx, order=1)
tt_ref = np.asarray(tt_ref, dtype=np.float64)

out_path = FIXTURES / "marmousi_skfmm_ref.npy"
np.save(out_path, tt_ref)

print(f"Saved {out_path}")
print(f"  shape : {tt_ref.shape}")
print(f"  tt[0,0]  = {tt_ref[0,0]:.6f} s")
print(f"  tt[0,940]= {tt_ref[0,940]:.6f} s")
print(f"  range : [{tt_ref.min():.6f}, {tt_ref.max():.6f}] s")
