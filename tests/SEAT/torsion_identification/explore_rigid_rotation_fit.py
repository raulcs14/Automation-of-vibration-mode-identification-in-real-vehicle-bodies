"""
Physical refinement of the torsion antisymmetry score.

Motivation
----------
The current `score_lr = -corr(Uz_left, Uz_right)` only tests whether left and
right are in OPPOSITE PHASE along X.  Correlation is blind to two things a real
rigid rotation about X must satisfy:

  1. Amplitude antisymmetry: Uz_left ~ -Uz_right with the SAME magnitude.
     corr = +1 even if one side barely moves (Uz_left = -0.01*Uz_right).
  2. Lever-arm scaling: a rigid rotation gives Uz = theta_x * Y, so the
     displacement must grow with |Y|.  corr ignores amplitude entirely.

A more physical, self-differentiating metric is the goodness-of-fit of the mode
to an IDEAL rigid rotation field.  For each X-slice, a rigid rotation predicts

    Uz_node = theta_slice * Y_node          (and  Uy_node = -theta_slice * Z_node)

We fit theta_slice per slice (least squares through the origin) and measure the
fraction of the slice's Uz (and Uy) variance explained by that rotation.  This
is naturally in [0, 1], equals 1 ONLY for a pure rotation, and degrades
physically (not by an artificial exponent) when the mode couples bending,
lateral motion, or a one-sided amplitude.

Variants compared
-----------------
  score_lr      : current correlation fingerprint (reference)
  rigid_uz      : R^2 of Uz vs theta*Y, averaged over slices (lever-arm aware)
  rigid_uzuy    : R^2 of the full (Uz,Uy) field vs the rigid-rotation prediction
  rigid_x_lin   : rigid_uzuy folded with how linear theta(X) is (true 1st torsion)

Run:  py tests/SEAT/torsion_identification/explore_rigid_rotation_fit.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np

from seat_model.reader import read_hdf5_modal
from common.torsion_analysis import scan_torsion_scores_v2
from common.visualization.torsion_plots import _classify_row

MODAL_H5 = Path("data/seat_model/TB/ansa/modal/output/000_Header_TB_modal_run.h5")
N_SLICES = 100
THR      = 0.5
MIN_R2   = 100.0      # min Y^2+Z^2 for a node to have a usable lever arm


def rigid_fit_with_theta_lin(node_xyz, uy, uz, n_slices, min_radius_sq=MIN_R2):
    """
    Prototype variant of common.torsion_analysis.rigid_rotation_fit that ALSO
    returns theta_lin (how linear theta(X) is), used here to explore the
    rigid_x_lin = rigid_uzuy * theta_lin "pure first torsion" indicator.

    Kept local (not promoted to production) because the production fit already
    folds linearity in separately via the soft gate on `linearity`; this script
    only exists to visualise the rigid_x_lin combination.  Named distinctly so
    it does not shadow the production rigid_rotation_fit.

    For each slice fit theta minimising || (Uz - theta*Y, Uy + theta*Z) ||^2
    (closed form), then R^2 = 1 - ss_res/ss_tot over that slice.  Returns the
    amplitude-weighted mean R^2 across slices (slices that actually move count
    more), plus the per-slice theta for the linearity-of-theta(X) check.
    """
    X, Y, Z = node_xyz[:, 0], node_xyz[:, 1], node_xyz[:, 2]
    lever = Y**2 + Z**2
    valid = lever > min_radius_sq
    bins  = np.linspace(X.min(), X.max(), n_slices + 1)
    bi    = np.digitize(X, bins)

    xs, thetas, r2_full, r2_uz, weights = [], [], [], [], []
    for b in range(1, n_slices + 1):
        m = (bi == b) & valid
        if m.sum() < 8:
            continue
        Xs, Ys, Zs, uys, uzs = X[m], Y[m], Z[m], uy[m], uz[m]

        # theta minimising ||Uz - theta*Y||^2 + ||Uy + theta*Z||^2
        num = float(np.sum(uzs * Ys - uys * Zs))
        den = float(np.sum(Ys**2 + Zs**2))
        if den < 1e-30:
            continue
        theta = num / den

        # full (Uy,Uz) rigid-rotation R^2
        res = (uzs - theta * Ys)**2 + (uys + theta * Zs)**2
        tot = uzs**2 + uys**2
        ss_res, ss_tot = float(res.sum()), float(tot.sum())
        r2f = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 1e-30 else 0.0

        # Uz-only (lever-arm) R^2: directly comparable to score_lr
        res_z = (uzs - theta * Ys)**2
        tot_z = uzs**2
        r2z = max(0.0, 1.0 - float(res_z.sum()) / float(tot_z.sum())) \
            if float(tot_z.sum()) > 1e-30 else 0.0

        xs.append(float(Xs.mean()))
        thetas.append(theta)
        r2_full.append(r2f)
        r2_uz.append(r2z)
        weights.append(ss_tot)          # weight slices by how much they move

    if not weights:
        return dict(rigid_uz=0.0, rigid_uzuy=0.0, theta_lin=0.0)

    w = np.array(weights); w = w / w.sum()
    rigid_uz   = float(np.dot(w, r2_uz))
    rigid_uzuy = float(np.dot(w, r2_full))

    # linearity of theta(X): R^2 of a straight-line fit (1st torsion mode = linear)
    xs = np.array(xs); th = np.array(thetas)
    if len(xs) >= 4:
        c = np.polyfit(xs, th, 1)
        ss_r = float(np.sum((th - np.polyval(c, xs))**2))
        ss_t = float(np.sum((th - th.mean())**2))
        theta_lin = max(0.0, 1.0 - ss_r / ss_t) if ss_t > 1e-30 else 0.0
    else:
        theta_lin = 0.0

    return dict(rigid_uz=rigid_uz, rigid_uzuy=rigid_uzuy, theta_lin=theta_lin)


# ---------------------------------------------------------------------------
print("Loading TB modes...")
data = read_hdf5_modal(MODAL_H5)
node_xyz, modes, freq = data["node_xyz"], data["modes"], data["freq"]
nNodes = len(node_xyz)
Uy_idx = np.arange(1, 6 * nNodes, 6)
Uz_idx = np.arange(2, 6 * nNodes, 6)

results = scan_torsion_scores_v2(node_xyz, modes, freq, n_slices=N_SLICES)

rows = []
for row in results:
    mi = int(row["mode_idx"]) - 1
    fit = rigid_fit_with_theta_lin(node_xyz, modes[Uy_idx, mi], modes[Uz_idx, mi], N_SLICES)
    rows.append((row, fit))

# rank torsion modes by the current combined for the comparison table
tors = [(r, f) for r, f in rows if _classify_row(r, THR) == "TORSION"]
tors.sort(key=lambda rf: -float(rf[0]["combined"]))

print(f"\nTORSION modes — current score_lr vs lever-arm-aware rigid-rotation fit")
print(f"(rigid_uz = R^2 of Uz vs theta*Y;  rigid_uzuy = R^2 of full rotation;")
print(f" rigid_x_lin = rigid_uzuy * theta_lin = pure FIRST torsion)\n")
print(f"  {'mode':>4} {'freq':>7} {'sc_lr':>6} {'rig_uz':>6} {'rig_uzuy':>8} "
      f"{'th_lin':>6} {'rig_x_lin':>9}")
print("-" * 60)
for row, fit in tors[:10]:
    rxl = fit["rigid_uzuy"] * fit["theta_lin"]
    print(f"  {int(row['mode_idx']):4d} {row['freq_hz']:7.2f} "
          f"{row['score_lr']:+6.3f} {fit['rigid_uz']:6.3f} "
          f"{fit['rigid_uzuy']:8.3f} {fit['theta_lin']:6.3f} {rxl:9.4f}")

print("\nIf rig_uz separates mode 22 from 82/83 (which score_lr does NOT),")
print("the lever-arm fit is the more physical, self-differentiating metric.")
