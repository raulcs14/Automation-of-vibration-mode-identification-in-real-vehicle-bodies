"""
Diagnostic: where is the real torsion axis, and does centering it matter?

The production rigid_rotation_fit assumes each slice rotates about the GLOBAL X
axis (Y=0, Z=0):

    Uz = theta * Y          Uy = -theta * Z         (1 unknown: theta)

But a vehicle's torsion (shear) axis usually sits below the geometric origin,
so a real torsion mode may rotate about (Y0, Z0) != (0, 0).  A more general fit
per slice solves for the centre too:

    Uz = theta * (Y - Y0)   Uy = -theta * (Z - Z0)  (3 unknowns)

which is linear in (a, b, c) with a = theta, b = -theta*Y0, c = theta*Z0:

    Uz = a*Y + b            Uy = -a*Z + c
    => Y0 = -b/a,  Z0 = c/a

This script compares, per torsion mode:
  R2_axisX  : rigid_uz with the axis pinned at the origin  (production)
  R2_free   : rigid_uz with (Y0, Z0) fitted free per slice
  Y0, Z0    : amplitude-weighted mean fitted centre across slices (in mm)

If R2_free >> R2_axisX and Z0 is consistently far from 0, the axis is offset and
centering the production fit is worth it.  If they are close, the global X axis
is fine and the extra parameters would only add permissiveness/noise.

Run:  py tests/SEAT/torsion_identification/explore_rotation_center.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np

from seat_model.reader import read_hdf5_modal
from common.torsion_analysis import scan_torsion_scores_v2, rigid_rotation_fit
from common.visualization.torsion_plots import _classify_row

MODAL_H5 = Path("data/seat_model/TB/ansa/modal/output/000_Header_TB_modal_run.h5")
N_SLICES = 100
THR      = 0.5
MIN_R2   = 100.0
MIN_NODES = 8


def free_center_fit(node_xyz, uy, uz, n_slices,
                    min_radius_sq=MIN_R2, min_nodes=MIN_NODES):
    """
    Per-slice rigid rotation with a FREE centre (Y0, Z0), amplitude-weighted.

    Returns (rigid_uz_free, Y0_mean, Z0_mean) where the centre is the
    energy-weighted mean of the per-slice fitted centres (mm).  Only the Uz
    fit is reported as R2 so it is directly comparable to production rigid_uz.
    """
    X, Y, Z = node_xyz[:, 0], node_xyz[:, 1], node_xyz[:, 2]
    valid   = (Y ** 2 + Z ** 2) > min_radius_sq
    bins    = np.linspace(X.min(), X.max(), n_slices + 1)
    bi      = np.digitize(X, bins)

    r2_uz, y0s, z0s, weights = [], [], [], []
    for b in range(1, n_slices + 1):
        m = (bi == b) & valid
        if m.sum() < min_nodes:
            continue
        Ys, Zs, uys, uzs = Y[m], Z[m], uy[m], uz[m]

        # Fit Uz = a*Y + b  and  Uy = -a*Z + c jointly for (a, b, c).
        # Stack both equations: a couples them, b only in Uz, c only in Uy.
        # Normal equations (least squares) for the 3 unknowns.
        n = m.sum()
        SYY = float(np.sum(Ys * Ys)); SY = float(np.sum(Ys))
        SZZ = float(np.sum(Zs * Zs)); SZ = float(np.sum(Zs))
        SUzY = float(np.sum(uzs * Ys)); SUz = float(np.sum(uzs))
        SUyZ = float(np.sum(uys * Zs)); SUy = float(np.sum(uys))

        # d/da: (SYY+SZZ)a + SY*b - SZ*c = SUzY - SUyZ
        # d/db: SY*a + n*b            = SUz
        # d/dc: -SZ*a        + n*c    = SUy
        A = np.array([[SYY + SZZ, SY, -SZ],
                      [SY,        n,   0.0],
                      [-SZ,       0.0, n]])
        rhs = np.array([SUzY - SUyZ, SUz, SUy])
        try:
            a, bb, c = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            continue
        if abs(a) < 1e-30:
            continue

        # Uz-only R2 with the fitted centre
        uz_pred = a * Ys + bb
        ss_res = float(np.sum((uzs - uz_pred) ** 2))
        ss_tot = float(np.sum(uzs ** 2))
        if ss_tot < 1e-30:
            continue
        r2_uz.append(max(0.0, 1.0 - ss_res / ss_tot))
        y0s.append(-bb / a)         # Y0 = -b/a
        z0s.append(c / a)           # Z0 =  c/a
        weights.append(float(np.sum(uzs ** 2 + uys ** 2)))

    if not weights:
        return 0.0, float("nan"), float("nan")
    w = np.array(weights); w = w / w.sum()
    return (float(np.dot(w, r2_uz)),
            float(np.dot(w, y0s)),
            float(np.dot(w, z0s)))


# ---------------------------------------------------------------------------
print("Loading TB modes...")
data = read_hdf5_modal(MODAL_H5)
node_xyz, modes, freq = data["node_xyz"], data["modes"], data["freq"]
nNodes = len(node_xyz)
Uy_idx = np.arange(1, 6 * nNodes, 6)
Uz_idx = np.arange(2, 6 * nNodes, 6)

# geometry reference: where is the structure in Z?
Zc = node_xyz[:, 2]
print(f"\nModel Z range: [{Zc.min():.0f}, {Zc.max():.0f}] mm,  "
      f"median Z = {np.median(Zc):.0f} mm  (origin Z=0)\n")

results = scan_torsion_scores_v2(node_xyz, modes, freq, n_slices=N_SLICES)
tors = [r for r in results if _classify_row(r, THR) == "TORSION"]
tors.sort(key=lambda r: -float(r["combined"]))

# Common-axis fit: a SINGLE axis for the whole model at the structure's median
# Z (and median Y), so each slice still has only theta as unknown (no per-slice
# permissiveness) but the axis sits where the real torsion axis is.  Implemented
# by shifting the coordinates and reusing the production fit.
Y_axis = float(np.median(node_xyz[:, 1]))
Z_axis = float(np.median(node_xyz[:, 2]))
node_shift = node_xyz.copy()
node_shift[:, 1] -= Y_axis
node_shift[:, 2] -= Z_axis
print(f"Common axis placed at (Y={Y_axis:.0f}, Z={Z_axis:.0f}) mm\n")

print("R2_axisX = origin axis (production);  R2_common = single median axis;")
print("R2_free  = free per-slice centre (most permissive)")
print(f"  {'mode':>4} {'freq':>7} {'R2_axisX':>8} {'R2_common':>9} {'R2_free':>8} "
      f"{'Y0(mm)':>7} {'Z0(mm)':>7}")
print("-" * 62)
for row in tors[:8]:
    mi = int(row["mode_idx"]) - 1
    uy, uz = modes[Uy_idx, mi], modes[Uz_idx, mi]
    r2_axisx, _  = rigid_rotation_fit(node_xyz,   uy, uz, N_SLICES, min_radius_sq=MIN_R2)
    r2_common, _ = rigid_rotation_fit(node_shift, uy, uz, N_SLICES, min_radius_sq=MIN_R2)
    r2_free, y0, z0 = free_center_fit(node_xyz, uy, uz, N_SLICES)
    print(f"  {int(row['mode_idx']):4d} {row['freq_hz']:7.2f} "
          f"{r2_axisx:8.3f} {r2_common:9.3f} {r2_free:8.3f} "
          f"{y0:7.0f} {z0:7.0f}")

print("\nWant: R2_common lifts the TRUE torsion mode (22) but NOT the others,")
print("i.e. it improves physics without the permissiveness of the free fit.")
