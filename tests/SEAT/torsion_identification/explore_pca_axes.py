"""
Diagnostic: derive the torsion axis from the geometry via PCA instead of
assuming it is the global X axis.

Why
---
rigid_rotation_fit currently hard-codes two geometric assumptions:
  1. the spin (longitudinal) axis is global X,
  2. it passes through the origin (Y=0, Z=0).
A model oriented along Y or Z, or with its torsion axis off the origin, breaks
both.  The shape of a car body is a long ellipsoid, so its principal axes ARE
the natural body frame: the longest principal direction is the morro-cola
(spin) axis, the other two are width and height.

Method (robust, no mesh weighting)
-----------------------------------
  centre = median(node_xyz, axis=0)            # robust to mesh-density bias
  C      = cov(node_xyz - centre)              # 3x3 covariance
  λ, V   = eigh(C)                             # principal directions
  e_long = V[:, argmax(λ)]                     # longitudinal -> spin axis
  remaining two: the more vertical (largest |global Z| component) -> e_vert,
                 the last -> e_lat
Coordinates AND modal displacements are projected onto (e_long, e_lat, e_vert),
then rigid_rotation_fit is applied about the new longitudinal axis (now "X'")
through the centre.  Result is compared to the production origin-X fit.

Run:  py tests/SEAT/torsion_identification/explore_pca_axes.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np

from seat_model.reader import read_hdf5_modal
from common.torsion_analysis import (
    scan_torsion_scores_v2, rigid_rotation_fit, pca_body_frame,
)
from common.visualization.torsion_plots import _classify_row

MODAL_H5 = Path("data/seat_model/TB/ansa/modal/output/000_Header_TB_modal_run.h5")
N_SLICES = 100
THR      = 0.5
MIN_R2   = 100.0


# pca_body_frame is imported from common.torsion_analysis so this diagnostic
# always reflects the production frame (PCA heading levelled to the ground).

# ---------------------------------------------------------------------------
print("Loading TB modes...")
data = read_hdf5_modal(MODAL_H5)
node_xyz, modes, freq = data["node_xyz"], data["modes"], data["freq"]
nNodes = len(node_xyz)
Ux_idx = np.arange(0, 6 * nNodes, 6)
Uy_idx = np.arange(1, 6 * nNodes, 6)
Uz_idx = np.arange(2, 6 * nNodes, 6)

centre, R, lam = pca_body_frame(node_xyz)
print("\nPCA body frame (robust median centre, longitudinal heading levelled):")
print(f"  centre      = ({centre[0]:.0f}, {centre[1]:.0f}, {centre[2]:.0f}) mm")
print(f"  e_long (spin) = ({R[0,0]:+.3f}, {R[0,1]:+.3f}, {R[0,2]:+.3f})")
print(f"  e_lat         = ({R[1,0]:+.3f}, {R[1,1]:+.3f}, {R[1,2]:+.3f})")
print(f"  e_vert        = ({R[2,0]:+.3f}, {R[2,1]:+.3f}, {R[2,2]:+.3f})")
heading = np.degrees(np.arctan2(R[0, 1], R[0, 0]))   # yaw of the spin axis in XY
print(f"  spin axis is horizontal (Z-comp={R[0,2]:+.4f}); heading vs +X = {heading:+.2f} deg")

# project coordinates into the body frame
xyz_b = (node_xyz - centre) @ R.T       # (nNodes,3): cols = long, lat, vert

results = scan_torsion_scores_v2(node_xyz, modes, freq, n_slices=N_SLICES)
tors = [r for r in results if _classify_row(r, THR) == "TORSION"]
tors.sort(key=lambda r: -float(r["combined"]))

print("\nrigid_uz: production (origin-X) vs PCA body-frame axis through centre")
print(f"  {'mode':>4} {'freq':>7} {'R2_prod':>8} {'R2_pca':>7} {'dR2':>6}")
print("-" * 42)
for row in tors[:8]:
    mi = int(row["mode_idx"]) - 1
    ux, uy, uz = modes[Ux_idx, mi], modes[Uy_idx, mi], modes[Uz_idx, mi]

    # production fit: global axes, origin
    r2_prod, _ = rigid_rotation_fit(node_xyz, uy, uz, N_SLICES, min_radius_sq=MIN_R2)

    # PCA fit: rotate displacement vectors into the body frame, fit about long axis
    disp = np.column_stack([ux, uy, uz]) @ R.T   # (nNodes,3): u_long,u_lat,u_vert
    # in the body frame, "X"=long, "Y"=lat, "Z"=vert -> reuse rigid_rotation_fit
    r2_pca, _ = rigid_rotation_fit(xyz_b, disp[:, 1], disp[:, 2], N_SLICES,
                                   min_radius_sq=MIN_R2)
    print(f"  {int(row['mode_idx']):4d} {row['freq_hz']:7.2f} "
          f"{r2_prod:8.3f} {r2_pca:7.3f} {r2_pca - r2_prod:+6.3f}")

print("\nIf the spin axis ~ global X and centre Z is the offset we found before,")
print("R2_pca should match the 'common axis' result (lifts mode 22, not the rest).")

# ---------------------------------------------------------------------------
# Visualise the derived rotation axis over the geometry
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

X, Y, Z = node_xyz[:, 0], node_xyz[:, 1], node_xyz[:, 2]

# subsample the node cloud for a light scatter
rng = np.random.default_rng(0)
idx = rng.choice(len(node_xyz), size=min(4000, len(node_xyz)), replace=False)

# axis half-lengths scaled from the principal extents (2.5 sigma each)
half = 2.5 * np.sqrt(lam)
e_long, e_lat, e_vert = R[0], R[1], R[2]

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X[idx], Y[idx], Z[idx], s=2, alpha=0.25, color="lightsteelblue",
           linewidths=0)

# draw each body axis as a line through the robust centre
for vec, h, color, label in [
    (e_long, half[0], "firebrick",  "rotation axis (longitudinal)"),
    (e_lat,  half[1], "seagreen",   "lateral"),
    (e_vert, half[2], "steelblue",  "vertical"),
]:
    p0 = centre - vec * h
    p1 = centre + vec * h
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
            color=color, lw=3 if label.startswith("rotation") else 1.8,
            label=label)

# mark the robust centre the axis passes through
ax.scatter(*centre, color="black", s=60, marker="o", label="robust centre")

# equal aspect so the axes are not visually distorted
xc, yc, zc = (X.max()+X.min())/2, (Y.max()+Y.min())/2, (Z.max()+Z.min())/2
rng_half = max(X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()) / 2
ax.set_xlim(xc-rng_half, xc+rng_half)
ax.set_ylim(yc-rng_half, yc+rng_half)
ax.set_zlim(zc-rng_half, zc+rng_half)
ax.set_box_aspect((1, 1, 1))            # cube box so equal limits look equal
ax.view_init(elev=18, azim=-72)         # slight 3/4 view to read the tilt

ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]"); ax.set_zlabel("Z [mm]")
ax.set_title(
    "PCA-derived rotation axis over the TB geometry\n"
    f"centre = ({centre[0]:.0f}, {centre[1]:.0f}, {centre[2]:.0f}) mm,   "
    f"spin tilt vs global X = {np.degrees(np.arccos(abs(R[0,0]))):.1f}°"
)
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
plt.show()
