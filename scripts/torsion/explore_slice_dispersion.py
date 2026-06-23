"""
[EXPLORATION] Diagnostic: intra-slice theta_x dispersion for the top torsional modes.

For each slice, plots the mean theta_x AND the +/- std band.
If std >> mean in many slices, the per-slice average is not representative
(rotations cancel within the bin).

This operates in the SAME geometry as the production pipeline: coordinates and
modal displacements are projected into the PCA body frame (pca_body_frame) before
theta_x is computed, so the rotation axis is the levelled longitudinal axis
through the robust centroid (not the raw global origin, which sits ~283 mm below
the real axis in Z for the TB model).  Classification also matches the core
(_classify_row): TORSION is decided by score_lr alone.

Run:  py scripts/torsion/explore_slice_dispersion.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
import _bootstrap  # noqa: F401  -- puts repo root (and scripts/) on sys.path
import numpy as np
import matplotlib.pyplot as plt

from seat_model.reader import read_hdf5_modal
from common.torsion_analysis import scan_torsion_scores_v2, pca_body_frame
from common.visualization.torsion_plots import _classify_row

# ---------------------------------------------------------------------------
MODAL_H5 = Path("data/seat_model/TB/ansa/modal/output/000_Header_TB_modal_run.h5")
N_SLICES  = 30
N_TOP     = 6
THR       = 0.5
# ---------------------------------------------------------------------------


def theta_x_profile_with_std(node_xyz, uy, uz, n_slices, min_radius_sq=100.0):
    """
    Same binning as theta_x_profile but also returns per-slice std and node count.

    Returns
    -------
    x_centers : (n_valid,)
    means     : (n_valid,)  mean theta_x per slice
    stds      : (n_valid,)  std  theta_x per slice
    counts    : (n_valid,)  number of nodes per slice
    """
    X, Y, Z = node_xyz[:, 0], node_xyz[:, 1], node_xyz[:, 2]
    denom = Y**2 + Z**2
    valid = denom > min_radius_sq
    theta_x = np.where(valid, (uz * Y - uy * Z) / np.where(valid, denom, 1.0), np.nan)

    bins = np.linspace(X.min(), X.max(), n_slices + 1)
    bi   = np.digitize(X, bins)

    x_centers, means, stds, counts = [], [], [], []
    for b in range(1, n_slices + 1):
        mask = (bi == b) & valid
        vals = theta_x[mask]
        if mask.sum() > 3:
            x_centers.append(float(X[mask].mean()))
            means.append(float(np.nanmean(vals)))
            stds.append(float(np.nanstd(vals)))
            counts.append(int(mask.sum()))

    return (np.array(x_centers), np.array(means),
            np.array(stds),      np.array(counts))


print("Loading TB modes...")
data       = read_hdf5_modal(MODAL_H5)
node_xyz_g = data["node_xyz"]      # raw global coordinates
modes      = data["modes"]
freq       = data["freq"]
nNodes     = len(node_xyz_g)

# Project into the PCA body frame, exactly as scan_torsion_scores_v2 does, so the
# theta_x profiles below use the real (levelled, centred) torsion axis rather than
# the raw global origin.  coords carry the body-frame node positions; per mode we
# rotate the (ux,uy,uz) displacement field with the same R (no translation).
centre, R, _ = pca_body_frame(node_xyz_g)
node_xyz = (node_xyz_g - centre) @ R.T          # body-frame coords (long, lat, vert)

print("Computing torsion scores...")
results = scan_torsion_scores_v2(node_xyz_g, modes, freq,
                                 n_slices=N_SLICES, skip_rigid=True)

# Classification matches the core (_classify_row): TORSION by score_lr alone.
torsion_rows = [r for r in results
                if _classify_row(r, THR) in {"TORSION", "ROLLING"}][:N_TOP]

Ux_idx = np.arange(0, 6 * nNodes, 6)
Uy_idx = np.arange(1, 6 * nNodes, 6)
Uz_idx = np.arange(2, 6 * nNodes, 6)


def _body_frame_uyuz(mi: int):
    """Return (uy, uz) of mode mi rotated into the body frame."""
    u = np.column_stack([modes[Ux_idx, mi], modes[Uy_idx, mi], modes[Uz_idx, mi]]) @ R.T
    return u[:, 1], u[:, 2]


COLORS = {"TORSION": "firebrick", "ROLLING": "darkorange"}

ncols = min(3, len(torsion_rows))
nrows = int(np.ceil(len(torsion_rows) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
axes = axes.flatten()

for i, row in enumerate(torsion_rows):
    mi    = int(row["mode_idx"]) - 1
    fhz   = float(row["freq_hz"])
    mtype = _classify_row(row, THR)
    color = COLORS.get(mtype, "steelblue")

    uy, uz = _body_frame_uyuz(mi)
    x_c, means, stds, counts = theta_x_profile_with_std(node_xyz, uy, uz, N_SLICES)

    # normalise so all modes are on the same scale
    scale = np.abs(means).max() or 1.0
    means_n = means / scale
    stds_n  = stds  / scale

    snr = np.abs(means) / (stds + 1e-30)  # per-slice SNR

    ax = axes[i]
    ax.plot(x_c, means_n, "o-", color=color, lw=1.5, ms=4, label="mean θₓ (norm)")
    ax.fill_between(x_c, means_n - stds_n, means_n + stds_n,
                    color=color, alpha=0.2, label="±1 std")
    ax.axhline(0, color="k", lw=0.7, ls=":")

    # flag slices where std > mean (potential cancellation)
    bad = stds > np.abs(means)
    if bad.any():
        ax.scatter(x_c[bad], means_n[bad], marker="x", color="red",
                   s=60, zorder=5, label=f"std>|mean| ({bad.sum()} slices)")

    ax2 = ax.twinx()
    ax2.bar(x_c, snr, width=(x_c[1] - x_c[0]) * 0.6 if len(x_c) > 1 else 50,
            color="gray", alpha=0.2, label="SNR = |mean|/std")
    ax2.set_ylabel("SNR per slice", fontsize=7, color="gray")
    ax2.tick_params(labelsize=6, colors="gray")
    ax2.set_ylim(0, max(snr.max() * 1.2, 1))

    ax.set_title(
        f"Mode {int(row['mode_idx'])}  {fhz:.2f} Hz  [{mtype}]\n"
        f"bad slices (std>|mean|): {bad.sum()}/{len(means)}",
        fontsize=8,
    )
    ax.set_xlabel("X (mm)", fontsize=8)
    ax.set_ylabel("θₓ normalised", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6)

for j in range(len(torsion_rows), len(axes)):
    axes[j].set_visible(False)

fig.suptitle(
    f"Intra-slice θₓ dispersion — top {len(torsion_rows)} TORSION/ROLLING modes\n"
    f"Red ✕ = slices where std > |mean| (potential cancellation within bin)",
    fontsize=10,
)
fig.tight_layout()

# ---------------------------------------------------------------------------
# Figure 2: per-slice theta_x split by quadrant (L/R x Top/Bot)
# ---------------------------------------------------------------------------

def theta_x_by_quadrant(node_xyz, uy, uz, n_slices, min_radius_sq=100.0):
    """
    For each X-slice compute mean theta_x separately for 4 quadrants, split at
    the rotation axis (Y = 0 lateral, Z = 0 vertical, in the body frame):
        L-Top : Y > 0, Z > 0
        L-Bot : Y > 0, Z <= 0
        R-Top : Y < 0, Z > 0
        R-Bot : Y < 0, Z <= 0

    Returns
    -------
    x_centers : (n_valid,)
    q         : dict with keys 'LT','LB','RT','RB' → (n_valid,) mean theta_x
    """
    X, Y, Z = node_xyz[:, 0], node_xyz[:, 1], node_xyz[:, 2]
    # Split top/bottom at the rotation axis itself.  node_xyz is already in the
    # body frame (centred on the robust centroid), so the torsion axis is Z = 0 —
    # the same axis theta_x is measured about below.  Using Z.mean() instead would
    # offset the divider from the axis (mean != median centroid).
    Z_mid = 0.0

    denom  = Y**2 + Z**2
    valid  = denom > min_radius_sq
    theta_x = np.where(valid, (uz * Y - uy * Z) / np.where(valid, denom, 1.0), np.nan)

    bins = np.linspace(X.min(), X.max(), n_slices + 1)
    bi   = np.digitize(X, bins)

    quadrants = {
        "L-Top": (Y > 0) & (Z >  Z_mid),
        "L-Bot": (Y > 0) & (Z <= Z_mid),
        "R-Top": (Y < 0) & (Z >  Z_mid),
        "R-Bot": (Y < 0) & (Z <= Z_mid),
    }

    x_centers = []
    q = {k: [] for k in quadrants}

    for b in range(1, n_slices + 1):
        in_bin = (bi == b) & valid
        if in_bin.sum() < 4:
            continue
        x_centers.append(float(X[in_bin].mean()))
        for k, qmask in quadrants.items():
            mask = in_bin & qmask
            q[k].append(float(np.nanmean(theta_x[mask])) if mask.sum() > 1 else np.nan)

    return np.array(x_centers), {k: np.array(v) for k, v in q.items()}


Q_COLORS = {"L-Top": "#e41a1c", "L-Bot": "#ff7f00",
            "R-Top": "#377eb8", "R-Bot": "#4daf4a"}

fig2, axes2 = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
axes2 = axes2.flatten()

for i, row in enumerate(torsion_rows):
    mi    = int(row["mode_idx"]) - 1
    fhz   = float(row["freq_hz"])
    mtype = _classify_row(row, THR)

    uy, uz = _body_frame_uyuz(mi)
    x_c, q = theta_x_by_quadrant(node_xyz, uy, uz, N_SLICES)

    # normalise by global max across all quadrants
    all_vals = np.concatenate([v[~np.isnan(v)] for v in q.values()])
    scale = np.abs(all_vals).max() or 1.0

    ax = axes2[i]
    for label, vals in q.items():
        ax.plot(x_c, vals / scale, "o-", color=Q_COLORS[label],
                lw=1.2, ms=3, alpha=0.85, label=label)

    ax.axhline(0, color="k", lw=0.7, ls=":")
    ax.set_title(
        f"Mode {int(row['mode_idx'])}  {fhz:.2f} Hz  [{mtype}]\n"
        f"θₓ by quadrant (normalised)",
        fontsize=8,
    )
    ax.set_xlabel("X (mm)", fontsize=8)
    ax.set_ylabel("θₓ normalised", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, ncol=2)

for j in range(len(torsion_rows), len(axes2)):
    axes2[j].set_visible(False)

fig2.suptitle(
    f"Per-quadrant θₓ — top {len(torsion_rows)} TORSION/ROLLING modes\n"
    "L/R = Y>0 / Y<0    Top/Bot = Z above/below the rotation axis (Z=0)\n"
    "curves overlapping = rigid rotation;  curves diverging = section not rotating as one block",
    fontsize=9,
)
fig2.tight_layout()
plt.show()
