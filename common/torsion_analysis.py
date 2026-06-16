"""
Torsion identification metrics based on modal displacements.

Core idea (Andres): slice the model in planes perpendicular to X and check
whether each slice rotates as a rigid body around X.

Key insight on the correct metric
----------------------------------
Nastran eigenvectors are mass-normalised (phi^T M phi = 1), so absolute
amplitudes carry no physical meaning for comparing rotational vs translational
content.  Instead we use purely geometric, scale-free criteria.

A rigid rotation by theta_x about the X axis displaces every node by
delta = theta_x * (X_axis x r), i.e.

    U_z = +theta_x * Y      (a left node, Y>0, rises; a right node falls)
    U_y = -theta_x * Z      (a top node, Z>0, swings one way; a bottom node the other)

So a torsion mode leaves TWO independent antisymmetric fingerprints, each
strongest in a different region of the cross-section:

  LEFT/RIGHT fingerprint (lateral zones, |Y| large) — measured in U_z:
    left (Y>0) and right (Y<0) move OPPOSITE in Z
    -> corr(Uz_left, Uz_right) ~ -1   ->   score_lr = -corr ~ +1

  TOP/BOTTOM fingerprint (upper/lower zones, |Z - Z_axis| large) — measured in U_y:
    top (Z>Z_axis) and bottom (Z<Z_axis) move OPPOSITE in Y
    -> corr(Uy_top, Uy_bot) ~ -1      ->   score_tb = -corr ~ +1

Both scores are ~+1 for pure torsion.  Bending and rigid roll separate cleanly:

  Vertical bending  : both sides move TOGETHER in Z  -> score_lr ~ -1
  Lateral bending   : both sides move TOGETHER in Y  -> score_ly ~ -1
  Rigid lateral roll: whole section translates in Y, top and bottom IN PHASE
                      -> score_tb ~ -1   (distinguishes roll from torsion)

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
    LEFT/RIGHT split: left (Y > y_threshold) vs right (Y < -y_threshold).

    Bin both sides by X and return (x_centers, left_means, right_means).
    Only bins where both sides have > 2 nodes are included.  Used for the
    U_z lateral fingerprint of torsion (and the U_y lateral-bending check).
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


def _tb_bin_means(
    X: np.ndarray,
    Z: np.ndarray,
    disp: np.ndarray,
    n_slices: int,
    z_axis: float,
    z_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    TOP/BOTTOM split about the rotation axis: top (Z > z_axis + z_threshold)
    vs bottom (Z < z_axis - z_threshold).

    Bin both halves by X and return (x_centers, top_means, bot_means).
    Only bins where both halves have > 2 nodes are included.  Used for the
    U_y vertical fingerprint of torsion (U_y = -theta_x * Z, so the sign
    flips across the rotation axis).
    """
    top = Z >  z_axis + z_threshold
    bot = Z <  z_axis - z_threshold
    bins = np.linspace(X.min(), X.max(), n_slices + 1)
    bi   = np.digitize(X, bins)

    x_c, t_means, b_means = [], [], []
    for b in range(1, n_slices + 1):
        mt = (bi == b) & top
        mb = (bi == b) & bot
        if mt.sum() > 2 and mb.sum() > 2:
            x_c.append(float(X[mt].mean()))
            t_means.append(float(disp[mt].mean()))
            b_means.append(float(disp[mb].mean()))

    return np.array(x_c), np.array(t_means), np.array(b_means)


def _x_variation(X: np.ndarray, disp: np.ndarray, n_slices: int) -> float:
    """
    Relative variation along X of the per-slice mean of `disp`.

    Distinguishes a rigid translation (the per-slice mean is ~constant along X,
    variation ~ 0) from a bending shape (the per-slice mean follows a curve,
    variation large).  Scale-free: std over X divided by the mean magnitude.

        var = std_X(slice_mean) / (mean_X|slice_mean| + eps)

    Returns 0 when there are too few slices or the field is ~zero.
    """
    bins = np.linspace(X.min(), X.max(), n_slices + 1)
    bi   = np.digitize(X, bins)
    slice_means = []
    for b in range(1, n_slices + 1):
        m = bi == b
        if m.sum() > 2:
            slice_means.append(float(disp[m].mean()))
    if len(slice_means) < 3:
        return 0.0
    sm = np.array(slice_means)
    scale = float(np.abs(sm).mean())
    if scale < 1e-30:
        return 0.0
    return float(np.std(sm) / scale)


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


def peak_concentration(
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    hot_fraction: float = 0.01,
) -> float:
    """
    Fraction of the modal kinetic energy held by the hottest `hot_fraction`
    of nodes (default the top 1%).

    Complements spatial_uniformity: Shannon entropy is insensitive to a single
    dominant peak sitting on a low background, whereas this metric goes to ~1
    exactly for those localised modes (almost all energy in a few nodes).

        e_i  = Ux_i^2 + Uy_i^2 + Uz_i^2
        peak = sum(top hot_fraction of e_i) / sum(e_i)

    Returns
    -------
    float in (0, 1]
        ~1  → energy concentrated in a tiny region (local mode, discard)
        low → energy spread over many nodes        (global mode)
    """
    e = ux ** 2 + uy ** 2 + uz ** 2
    total = float(e.sum())
    if total < 1e-30:
        return 1.0
    n_hot = max(1, int(round(len(e) * hot_fraction)))
    hottest = np.partition(e, -n_hot)[-n_hot:]
    return float(hottest.sum() / total)


def torsion_score_v2(
    node_xyz: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    n_slices: int = 20,
    y_threshold: float = 50.0,
    z_threshold: float = 50.0,
    min_radius_sq: float = 100.0,
    peak_thr: float = 0.6,
) -> dict:
    """
    Refined torsion score that measures two independent properties of the
    theta_x(X) profile and combines them multiplicatively.

    Sub-scores
    ----------
    linearity  : R^2 of the linear fit theta_x ~ a*X + b, BUT only meaningful
                 when the profile has significant amplitude.  Computed as:

                     noise    = sqrt(ss_res / n)        # RMSE of the linear fit
                     snr      = range(theta_x) / noise  # signal-to-residual ratio
                     R2_raw   = 1 - ss_res / ss_tot     # standard R^2
                     linearity = R2_raw * tanh(snr / 3)

                 The tanh factor suppresses R^2 when the profile is nearly flat
                 (small snr), preventing noisy flat profiles from scoring high.
                 Note: `noise` is the residual RMSE of the linear fit, not the
                 within-slice std (see explore_slice_dispersion.py for the latter).

    centering  : how close the rotation centre x0 = -b/a is to the geometric
                 centre of the model.  Defined as:

                     centering = sqrt(max(0,  1 - 2*|x0 - X_mid| / X_span))

                 so it is 1 when x0 = X_mid and 0 when x0 is at the model edge.
                 The sqrt makes the falloff concave (penalises small offsets
                 less than a linear ramp).  No free sigma parameter needed.

    antisym    : torsion antisymmetry driving the combined ranking.  Based on
                 the robust lateral fingerprint score_lr (U_z left/right), with
                 a small bonus from score_tb when the vertical fingerprint
                 agrees:  antisym = clip(score_lr_+ * (1 + 0.25*score_tb_+), 0, 1).
                 score_tb can only raise antisym, never lower it, because on
                 trimmed bodies U_y is noisy (local/lumped-mass motion).

    combined   : linearity * centering * antisym * uniformity

    Antisymmetry sub-scores (all in [-1, 1], +1 = torsion fingerprint present)
        score_lr : -corr(Uz_left,  Uz_right)   lateral  fingerprint (U_z)
        score_tb : -corr(Uy_top,   Uy_bottom)  vertical fingerprint (U_y)
        score_ly : -corr(Uy_left,  Uy_right)   +1 = lateral-bending/roll antisym
        score_xvar : >= 0, relative variation of per-slice mean U_y along X.
                     ~0 = rigid roll (uniform U_y); large = lateral bending.
    are returned for reference and classification.

    Parameters
    ----------
    node_xyz      : (nNodes, 3)
    ux            : (nNodes,)   X modal displacement
    uy            : (nNodes,)   Y modal displacement
    uz            : (nNodes,)   Z modal displacement
    n_slices      : X bins for the theta_x profile
    y_threshold   : nodes with |Y| < y_threshold excluded from L/R antisym calc
    z_threshold   : nodes with |Z - Z_axis| < z_threshold excluded from T/B calc
    min_radius_sq : nodes with Y^2+Z^2 < this excluded from theta_x calc
    peak_thr      : energy fraction in the hottest 1% of nodes above which the
                    mode is vetoed as local (combined forced to 0)

    Returns
    -------
    dict with keys:
        linearity, centering, antisym, uniformity, peak, combined,
        x0, R2,
        score_lr, score_tb, score_ly, score_xvar,
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

        n        = len(th)
        noise    = float(np.sqrt(ss_res / n)) if n > 0 else 1.0
        th_range = float(th.max() - th.min())
        snr      = th_range / (noise + 1e-30)
        snr_weight = float(np.tanh(snr / 3.0))

        R2        = R2_raw * snr_weight
        linearity = R2
        slope     = float(coeffs[0])

        if abs(slope) > 1e-20:
            x0 = -float(coeffs[1]) / slope
            centering = float(np.sqrt(max(0.0, 1.0 - 2.0 * abs(x0 - X_mid) / X_span)))

    # --- antisymmetry scores ---
    Y = node_xyz[:, 1]
    Z = node_xyz[:, 2]
    # Rotation axis height: median Z of nodes outside the central core, robust
    # to mesh-density bias (the central core has Y^2+Z^2 ~ 0 and no lever arm).
    z_axis = float(np.median(Z[(Y**2 + Z**2) > min_radius_sq])) \
        if np.any((Y**2 + Z**2) > min_radius_sq) else float(np.median(Z))

    def _anti_corr(a, b):
        # -corr: +1 when the two binned profiles are perfectly out of phase.
        # Returns 0 if a side is flat (zero variance -> corr undefined) or too
        # few bins, so a missing fingerprint never poisons the product/min.
        if len(a) < 3 or np.std(a) < 1e-30 or np.std(b) < 1e-30:
            return 0.0
        return float(-np.corrcoef(a, b)[0, 1])

    # Lateral fingerprint: U_z antisymmetric left/right  (U_z = +theta_x * Y)
    _, uz_l, uz_r = _lr_bin_means(X, Y, uz, n_slices, y_threshold)
    score_lr = _anti_corr(uz_l, uz_r)

    # Vertical fingerprint: U_y antisymmetric top/bottom  (U_y = -theta_x * Z)
    _, uy_t, uy_b = _tb_bin_means(X, Z, uy, n_slices, z_axis, z_threshold)
    score_tb = _anti_corr(uy_t, uy_b)

    # Lateral-bending / rigid-roll discriminant: U_y antisymmetric left/right
    _, uy_l, uy_r = _lr_bin_means(X, Y, uy, n_slices, y_threshold)
    score_ly = _anti_corr(uy_l, uy_r)

    # Roll-vs-lateral-bending discriminant: how much the per-slice mean U_y
    # varies along X.  Rigid roll translates the whole body in Y uniformly
    # (flat profile -> ~0); lateral bending curves along X (-> large).
    score_xvar = _x_variation(X, uy, n_slices)

    # antisym drives the combined ranking.  On real trimmed bodies U_z (lateral
    # left/right) is the clean torsion fingerprint; U_y carries a lot of local
    # motion (suspension, lumped masses), so score_tb is noisy and unreliable
    # as a hard requirement.  Base antisym on score_lr and let score_tb only
    # *boost* (never reduce) it when the vertical fingerprint agrees.
    lr_pos = max(0.0, score_lr)
    antisym = float(min(1.0, lr_pos * (1.0 + 0.25 * max(0.0, score_tb))))

    uniformity = spatial_uniformity(ux, uy, uz)

    # Peak-concentration veto: Shannon entropy alone lets through modes whose
    # energy is almost entirely in a tiny region but with a low spread-out
    # background.  peak_concentration goes to ~1 for those; if a mode parks
    # more than peak_thr of its energy in the hottest 1% of nodes it is local,
    # not a global torsion mode, so its combined score is zeroed outright.
    peak = peak_concentration(ux, uy, uz)
    local_veto = 0.0 if peak > peak_thr else 1.0

    # product of four sub-scores, each in [0, 1] (not a geometric mean:
    # adding factors does shrink the scale, but ranking is unaffected)
    combined = linearity * centering * antisym * uniformity * local_veto

    return dict(
        linearity    = linearity,
        centering    = centering,
        antisym      = antisym,
        uniformity   = uniformity,
        peak         = peak,
        combined     = combined,
        x0           = x0,
        R2           = R2,
        score_lr     = score_lr,
        score_tb     = score_tb,
        score_ly     = score_ly,
        score_xvar   = score_xvar,
        x_centers    = x_c,
        theta_means  = th,
    )


def scan_torsion_scores_v2(
    node_xyz: np.ndarray,
    modes: np.ndarray,
    freq: np.ndarray,
    n_slices: int = 20,
    y_threshold: float = 50.0,
    z_threshold: float = 50.0,
    min_radius_sq: float = 100.0,
    peak_thr: float = 0.6,
    skip_rigid: bool = True,
) -> np.ndarray:
    """
    Compute torsion_score_v2 for every mode.

    peak_thr : modes parking more than this fraction of their energy in the
               hottest 1% of nodes are vetoed (combined forced to 0).

    Returns structured array sorted by ``combined`` descending, with fields:
        mode_idx, freq_hz, combined, linearity, centering, antisym, uniformity,
        peak, x0, score_lr, score_tb, score_ly, score_xvar
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
            n_slices=n_slices, y_threshold=y_threshold,
            z_threshold=z_threshold, min_radius_sq=min_radius_sq,
            peak_thr=peak_thr,
        )
        records.append((
            mi + 1, float(freq[mi]),
            res["combined"], res["linearity"], res["centering"],
            res["antisym"], res["uniformity"], res["peak"], res["x0"],
            res["score_lr"], res["score_tb"], res["score_ly"], res["score_xvar"],
        ))

    dtype = np.dtype([
        ("mode_idx",   int),
        ("freq_hz",    float),
        ("combined",   float),
        ("linearity",  float),
        ("centering",  float),
        ("antisym",    float),
        ("uniformity", float),
        ("peak",       float),
        ("x0",         float),
        ("score_lr",   float),
        ("score_tb",   float),
        ("score_ly",   float),
        ("score_xvar", float),
    ])
    arr = np.array(records, dtype=dtype)
    return arr[np.argsort(-arr["combined"])]


