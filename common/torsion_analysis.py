"""
Torsion identification metrics based on modal displacements.

Core idea (Andres): slice the model in planes perpendicular to X and check
whether each slice rotates as a rigid body around X.

Key insight on the correct metric
----------------------------------
Nastran eigenvectors are mass-normalised (phi^T M phi = 1), so absolute
amplitudes carry no physical meaning for comparing rotational vs translational
content.  Instead we use a purely geometric, scale-free criterion:

  For a **pure torsion** mode, nodes on the LEFT side of the car (Y > 0) and
  nodes on the RIGHT side (Y < 0) move in OPPOSITE directions in Z and Y
  (antisymmetric w.r.t. the XZ plane).
  -> corr(Uz_left_binned, Uz_right_binned) ~ -1

  For a **pure bending** mode (e.g. vertical bending) both sides move together:
  -> corr(Uz_left_binned, Uz_right_binned) ~ +1

  For a **lateral bending** mode both sides move together in Y:
  -> corr(Uy_left_binned, Uy_right_binned) ~ +1   (and Uz corr ~ anything)

Public API
----------
  torsion_score_v2    : composite score for a single mode (linearity x centering x antisym x uniformity)
  scan_torsion_scores_v2 : apply torsion_score_v2 to every mode, return structured array sorted by score
  theta_x_profile     : rotation angle per X-slice for visualisation
  spatial_uniformity  : Shannon-entropy uniformity metric
"""

import numpy as np


def _lr_bin_means(
    X: np.ndarray,
    Y: np.ndarray,
    disp: np.ndarray,
    n_slices: int,
    y_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split nodes into left (Y > y_threshold) and right (Y < -y_threshold),
    bin by X, return (x_centers, left_means, right_means).
    Only bins where both sides have > 2 nodes are included.
    """
    left  = Y >  y_threshold
    right = Y < -y_threshold
    bins  = np.linspace(X.min(), X.max(), n_slices + 1)
    bi    = np.digitize(X, bins)

    x_c, l_means, r_means = [], [], []
    for b in range(1, n_slices + 1):
        ml = (bi == b) & left
        mr = (bi == b) & right
        if ml.sum() > 2 and mr.sum() > 2:
            x_c.append(float(X[ml].mean()))
            l_means.append(float(disp[ml].mean()))
            r_means.append(float(disp[mr].mean()))

    return np.array(x_c), np.array(l_means), np.array(r_means)


def theta_x_profile(
    node_xyz: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    n_slices: int = 10,
    min_radius_sq: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute mean theta_x per X-slice for visualisation.

    theta_x_i = (Uz_i * Y_i - Uy_i * Z_i) / (Y_i^2 + Z_i^2)

    Returns
    -------
    x_centers   : (n_valid_slices,)
    theta_means : (n_valid_slices,)
    """
    Y = node_xyz[:, 1]
    Z = node_xyz[:, 2]
    X = node_xyz[:, 0]
    denom = Y**2 + Z**2
    valid = denom > min_radius_sq
    theta_x = np.where(valid, (uz * Y - uy * Z) / np.where(valid, denom, 1.0), np.nan)

    bins = np.linspace(X.min(), X.max(), n_slices + 1)
    bi   = np.digitize(X, bins)
    x_centers, theta_means = [], []
    for b in range(1, n_slices + 1):
        mask = (bi == b) & valid
        if mask.sum() > 3:
            x_centers.append(float(X[mask].mean()))
            theta_means.append(float(np.nanmean(theta_x[mask])))

    return np.array(x_centers), np.array(theta_means)


def spatial_uniformity(
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
) -> float:
    """
    Measure how globally distributed a mode shape is across nodes.

    Uses normalised Shannon entropy of the translational kinetic-energy proxy
    per node:  e_i = Ux_i^2 + Uy_i^2 + Uz_i^2

    Returns
    -------
    uniformity : float in (0, 1]
        ~1  → energy spread uniformly across all nodes  (global mode)
        ~0  → energy concentrated in a few nodes         (local mode)

    Formula
    -------
        p_i        = e_i / sum(e_j)          # probability mass per node
        H_raw      = -sum(p_i * log(p_i))    # Shannon entropy
        H_max      = log(N)                   # entropy of uniform distribution
        uniformity = H_raw / H_max
    """
    e = ux ** 2 + uy ** 2 + uz ** 2
    total = float(e.sum())
    if total < 1e-30:
        return 0.0
    p = e / total
    # avoid log(0): nodes with p==0 contribute 0 to entropy
    mask = p > 0.0
    H = float(-np.sum(p[mask] * np.log(p[mask])))
    H_max = float(np.log(len(p)))
    return float(H / H_max) if H_max > 0 else 0.0


def torsion_score_v2(
    node_xyz: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    n_slices: int = 20,
    y_threshold: float = 50.0,
    min_radius_sq: float = 100.0,
) -> dict:
    """
    Refined torsion score that measures two independent properties of the
    theta_x(X) profile and combines them multiplicatively.

    Sub-scores
    ----------
    linearity  : R^2 of the linear fit theta_x ~ a*X + b, BUT only meaningful
                 when the profile has significant amplitude.  Computed as:

                     snr      = range(theta_x) / std_within_slices
                     R2_raw   = 1 - ss_res / ss_tot   (standard R^2)
                     linearity = R2_raw * tanh(snr / 3)

                 The tanh factor suppresses R^2 when the profile is nearly flat
                 (small snr), preventing noisy flat profiles from scoring high.

    centering  : how close the rotation centre x0 = -b/a is to the geometric
                 centre of the model.  Defined as:

                     centering = max(0,  1 - 2*|x0 - X_mid| / X_span)

                 so it is 1 when x0 = X_mid, 0 when x0 is at the model edge,
                 and linear in between.  No free sigma parameter needed.

    combined   : linearity * centering * antisym * uniformity   in [0, 1]

    The antisymmetry scores (score_Uz, score_Uy) from torsion_score() are also
    returned for reference.

    Parameters
    ----------
    node_xyz      : (nNodes, 3)
    ux            : (nNodes,)   X modal displacement
    uy            : (nNodes,)   Y modal displacement
    uz            : (nNodes,)   Z modal displacement
    n_slices      : X bins for the theta_x profile
    y_threshold   : nodes with |Y| < y_threshold excluded from antisymmetry calc
    min_radius_sq : nodes with Y^2+Z^2 < this excluded from theta_x calc

    Returns
    -------
    dict with keys:
        linearity, centering, antisym, uniformity, combined,
        x0, R2,
        score_Uz, score_Uy,
        x_centers, theta_means   (profile arrays for plotting)
    """
    # --- theta_x profile ---
    x_c, th = theta_x_profile(node_xyz, uy, uz, n_slices, min_radius_sq)

    X    = node_xyz[:, 0]
    X_min, X_max = float(X.min()), float(X.max())
    X_span = X_max - X_min
    X_mid  = (X_min + X_max) / 2.0

    linearity = 0.0
    centering = 0.0
    x0        = float("nan")
    R2        = 0.0

    if len(x_c) >= 4:
        coeffs  = np.polyfit(x_c, th, 1)
        th_fit  = np.polyval(coeffs, x_c)
        ss_res  = float(np.sum((th - th_fit) ** 2))
        ss_tot  = float(np.sum((th - th.mean()) ** 2))
        R2_raw  = float(max(0.0, 1.0 - ss_res / ss_tot)) if ss_tot > 1e-30 else 0.0

        # SNR of the profile: range / mean within-slice std
        # To get within-slice std we need the full theta_x values, not just means.
        # Approximate: std of the slice means is the signal; sqrt(ss_res/n) is noise.
        n    = len(th)
        noise = float(np.sqrt(ss_res / n)) if n > 0 else 1.0
        th_range = float(th.max() - th.min())
        snr  = th_range / (noise + 1e-30)
        # tanh suppressor: ~0 when snr<1 (flat/noisy), ~1 when snr>6 (clean profile)
        snr_weight = float(np.tanh(snr / 3.0))

        R2        = R2_raw * snr_weight
        linearity = R2

        a, b = float(coeffs[0]), float(coeffs[1])
        if abs(a) > 1e-20:
            x0 = -b / a
            # sqrt softens the penalty for slightly off-centre rotation centres
            # while still driving to 0 when x0 is outside the model
            centering = float(np.sqrt(max(0.0, 1.0 - 2.0 * abs(x0 - X_mid) / X_span)))

    # --- antisymmetry scores ---
    Y = node_xyz[:, 1]
    _, ul_z, ur_z = _lr_bin_means(X, Y, uz, n_slices, y_threshold)
    _, ul_y, ur_y = _lr_bin_means(X, Y, uy, n_slices, y_threshold)

    def _ac(l, r):
        return float(-np.corrcoef(l, r)[0, 1]) if len(l) >= 3 else 0.0

    score_Uz = _ac(ul_z, ur_z)
    score_Uy = _ac(ul_y, ur_y)

    # antisymmetry: clip to [0,1] — negative means bending (penalise to 0)
    antisym = float(max(0.0, score_Uz, score_Uy))

    # spatial uniformity
    uniformity = spatial_uniformity(ux, uy, uz)

    # combined: all four factors must be high simultaneously
    combined = linearity * centering * antisym * uniformity

    return dict(
        linearity   = linearity,
        centering   = centering,
        antisym     = antisym,
        combined    = combined,
        uniformity  = uniformity,
        x0          = x0,
        R2          = R2,
        score_Uz    = score_Uz,
        score_Uy    = score_Uy,
        x_centers   = x_c,
        theta_means = th,
    )


def scan_torsion_scores_v2(
    node_xyz: np.ndarray,
    modes: np.ndarray,
    freq: np.ndarray,
    n_slices: int = 20,
    y_threshold: float = 50.0,
    min_radius_sq: float = 100.0,
    skip_rigid: bool = True,
) -> np.ndarray:
    """
    Compute torsion_score_v2 for every mode.

    Returns structured array sorted by ``combined`` descending, with fields:
        mode_idx, freq_hz, combined, linearity, centering, x0, score_Uz, score_Uy
    """
    nNodes = len(node_xyz)
    nModes = modes.shape[1]
    Ux_idx = np.arange(0, 6 * nNodes, 6)
    Uy_idx = np.arange(1, 6 * nNodes, 6)
    Uz_idx = np.arange(2, 6 * nNodes, 6)

    records = []
    for mi in range(nModes):
        if skip_rigid and freq[mi] < 0.5:
            continue
        res = torsion_score_v2(
            node_xyz,
            modes[Ux_idx, mi], modes[Uy_idx, mi], modes[Uz_idx, mi],
            n_slices, y_threshold, min_radius_sq,
        )
        records.append((
            mi + 1, float(freq[mi]),
            res["combined"], res["linearity"], res["centering"],
            res["antisym"], res["uniformity"], res["x0"],
            res["score_Uz"], res["score_Uy"],
        ))

    dtype = np.dtype([
        ("mode_idx",   int),
        ("freq_hz",    float),
        ("combined",   float),
        ("linearity",  float),
        ("centering",  float),
        ("antisym",    float),
        ("uniformity", float),
        ("x0",         float),
        ("score_Uz",   float),
        ("score_Uy",   float),
    ])
    arr = np.array(records, dtype=dtype)
    return arr[np.argsort(-arr["combined"])]


