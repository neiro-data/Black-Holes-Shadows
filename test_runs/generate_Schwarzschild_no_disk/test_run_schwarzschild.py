#!/usr/bin/env python
"""End-to-end Schwarzschild (Weyl-coordinate, MD=0) black-hole shadow run.

Orchestrates the three pipeline stages documented in README.md's "Pipeline"
section, using a small/fast resolution suited for a first sanity run rather
than a publication-quality image:

1. `generate_matriz.generate_lambda_matrix` -- tabulate the lambda-potential
   lookup table (coarse 200x200 grid; exact for MD=0 since lambda then has a
   closed form, so a coarse grid is still accurate).
2. `test_Z_SHADOW.trace_shadow` -- ray-trace a small (40x40 before halving)
   quarter-image shadow grid serially. USE_DISK=False disables the
   disk-crossing branch, so this produces a pure BH-shadow trace.
3. `symmetry.render_shadow` -- classify captured/(beyond-disk if enabled)/
   neither, mirror the quadrant into the full image, and save a PNG.

No geodesic or classification code is duplicated here: this file only calls
the three refactored pipeline functions with a shared (M, MD, b) config.

Run with: uv run python test_run_schwarzschild.py
"""

import os
import sys
import time

import numpy as np

# This script lives at test_runs/generate_Schwarzschild_no_disk/, but the
# pipeline modules (generate_matriz, test_Z_SHADOW, symmetry) and the
# general_methods package live at the repo root (two levels up). Put the repo
# root on sys.path so the imports resolve regardless of the current working
# directory the script is launched from.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from generate_matriz import generate_lambda_matrix
from test_Z_SHADOW import trace_shadow
from symmetry import render_shadow

M = 1.0
MD = 0.0
b = 6.0
N_POINTS = 20
USE_DISK = False  # pure Schwarzschild shadow, no disk-crossing classification

# Anchor outputs to this script's folder so they land in generate_Schwarzschild_no_disk/
# regardless of the working directory the script is launched from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MATRIX_DIR = os.path.join(SCRIPT_DIR, "test_run_schwarzschild_matrices")
MATRIX_PATH = os.path.join(MATRIX_DIR, "Mat_nu_disk0.0")
OUT_DIR = os.path.join(SCRIPT_DIR, "image_generation")
OUT_PATH = os.path.join(OUT_DIR, "schwarzschild_shadow.png")

if __name__ == "__main__":
   os.makedirs(MATRIX_DIR, exist_ok=True)
   os.makedirs(OUT_DIR, exist_ok=True)

   start = time.time()
   print("Stage 1/3: tabulating lambda-potential matrix...")
   generate_lambda_matrix(M=M, MD=MD, b=b, n=200, out_path=MATRIX_PATH)

   print(f"Stage 1 is completed: {(time.time() - start):.2f} seconds")

   start = time.time()
   print("Stage 2/3: ray-tracing shadow quarter-grid...")
   Mat, Mz, alfa, beta = trace_shadow(M=M, MD=MD, b=b, n=N_POINTS, matrix_path=MATRIX_PATH, use_disk=USE_DISK)
   np.savetxt(os.path.join(MATRIX_DIR, "Mat"), Mat)
   np.savetxt(os.path.join(MATRIX_DIR, "Mz"), Mz)

   print(f"Stage 2 is completed: {(time.time() - start):.2f} seconds")

   print("Stage 3/3: classifying, mirroring, and rendering shadow image...")
   captured = render_shadow(Mat, Mz, M=M, MD=MD, b=b, out_path=OUT_PATH, use_disk=USE_DISK)

   print(f"Done. Captured (shadow) pixels: {captured}. Figure saved to {OUT_PATH}.")
