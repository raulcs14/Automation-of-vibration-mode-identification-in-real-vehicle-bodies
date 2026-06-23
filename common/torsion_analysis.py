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
  pca_body_frame      : geometry-derived body frame (longitudinal torsion axis,
                        levelled to the ground, through the robust centroid)
  rigid_rotation_fit  : lever-arm-aware goodness-of-fit of a mode to a rigid
                        rotation about the longitudinal axis (drives the ranking)
  torsion_score_v2    : composite score for a single mode
                        (antisym x gate(linearity) x gate(centering) x local_veto)
  scan_torsion_scores_v2 : apply torsion_score_v2 to every mode, return structured array sorted by score
  theta_x_profile     : rotation angle per X-slice for visualisation
  spatial_uniformity  : Shannon-entropy uniformity metric (reported, not ranked)
"""

import numpy as np


def pca_body_frame(node_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Derive the vehicle's body frame from the node cloud by principal-axis
    analysis, so the torsion axis is computed from the geometry instead of being
    assumed to be the global X axis through the origin.

    A car body is a long ellipsoid: its longest principal direction is the
    morro-cola (longitudinal) axis about which torsion occurs.  This makes the
    criterion orientation-free — a model meshed along Y or Z is handled without
    any hard-coded axis.

    Physical refinement: the torsion axis is the longitudinal axis of a vehicle
    sitting on the ground, i.e. it must be PARALLEL TO THE GROUND.  The raw PCA
    principal direction is not quite horizontal because PCA maximises node
    scatter and is therefore biased by body SHAPE (a tall roof at the rear vs a
    low bonnet at the front tilts it).  On the TB model the raw principal axis
    tilts 3.3 deg from horizontal while the actual floor slopes only ~1.3 deg —
    most of that tilt is a shape artefact, not real mounting tilt.  So we take
    only the HEADING of the principal axis (its projection onto the horizontal
    plane) and keep "up" as global Z.  This removes the shape bias and matches
    the physics of torsion (twist about a level longitudinal axis) while still
    correcting the important offset: the axis passes through the structure's
    robust mid-height, not the origin.

    Robust by design (no mesh weighting): the centre is the MEDIAN node position
    (insensitive to locally dense mesh regions that would drag a mean centroid).
    No tunable parameters are introduced — the frame is purely geometric.

    Method
    ------
        centre  = median(node_xyz, axis=0)
        lam, V  = eigh(cov(node_xyz - centre))   # principal directions
        e_long0 = eigenvector of largest lam     # raw longitudinal (may tilt)
        e_long  = e_long0 projected onto the horizontal plane, renormalised
        e_vert  = global Z (up)
        e_lat   = e_vert x e_long                # right-handed, horizontal

    Axes are sign-oriented for readability (e_long points +X-ish, e_vert +Z-ish)
    so the body frame stays close to the global frame for a conventionally
    aligned model and the score sign conventions (Uz = +theta*Y_body) are kept.

    Parameters
    ----------
    node_xyz : (nNodes, 3) node coordinates

    Returns
    -------
    centre : (3,)   robust centroid the spin axis passes through
    R      : (3, 3) rotation whose ROWS are (e_long, e_lat, e_vert).  For a node
             p, the body-frame coordinate is R @ (p - centre); for a displacement
             vector u (no translation), the body-frame displacement is R @ u.
    lam    : (3,)   principal variances ordered (longitudinal, lateral, vertical)
    """
    centre = np.median(node_xyz, axis=0)
    P = node_xyz - centre
    C = np.cov(P, rowvar=False)
    lam, V = np.linalg.eigh(C)                 # ascending lam; columns = eigvecs

    order = np.argsort(lam)[::-1]              # [longest, mid, shortest]
    lam = lam[order]
    V = V[:, order]
    e_long = V[:, 0]                           # raw longitudinal (largest extent)

    # consistent sign: longitudinal points +X-ish before we level it
    if e_long[0] < 0:
        e_long = -e_long

    # Level the axis: keep only its horizontal heading (zero the vertical comp),
    # so torsion is measured about a ground-parallel longitudinal axis.  Fall
    # back to the raw axis in the degenerate case of a (near-)vertical body.
    e_long_h = e_long.copy()
    e_long_h[2] = 0.0
    norm = np.linalg.norm(e_long_h)
    e_long = e_long_h / norm if norm > 1e-12 else e_long

    # "up" is global Z; lateral completes a right-handed, horizontal frame
    e_vert = np.array([0.0, 0.0, 1.0])
    e_lat = np.cross(e_vert, e_long)
    e_lat /= np.linalg.norm(e_lat)
    e_vert = np.cross(e_long, e_lat)           # re-orthonormalise (already unit)

    R = np.vstack([e_long, e_lat, e_vert])
    return centre, R, lam


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


def rigid_rotation_fit(
    node_xyz: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    n_slices: int,
    min_radius_sq: float = 100.0,
    min_nodes: int | None = None,
) -> tuple[float, float]:
    """
    Goodness-of-fit of the modal field to an IDEAL rigid rotation about X.

    Motivation
    ----------
    The correlation-based fingerprint score_lr = -corr(Uz_left, Uz_right) only
    tests whether the two sides are in OPPOSITE PHASE along X.  Correlation is
    blind to two properties a real rigid rotation must satisfy:

      1. Amplitude antisymmetry — Uz_left ~ -Uz_right with the SAME magnitude.
         corr = +1 even if one side barely moves (Uz_left = -0.01 * Uz_right).
      2. Lever-arm scaling — a rotation gives Uz = theta_x * Y, so displacement
         must grow with |Y|.  corr ignores amplitude entirely.

    This routine instead asks "how close is the field to an actual rotation?".
    A rigid rotation by theta about X predicts, per node:

        Uz = +theta * Y        Uy = -theta * Z

    For each X-slice we fit the single theta that best explains both components
    (closed-form least squares) and measure the fraction of the slice's motion
    explained by that rotation (R^2).  The result is naturally in [0, 1], equals
    1 ONLY for a pure rotation, and degrades physically — not via an artificial
    exponent — when the mode couples bending, lateral motion, or a one-sided
    amplitude.  This makes it far more discriminating than score_lr (e.g. it
    cleanly separates a true torsion mode from high-frequency local modes whose
    sides happen to be out of phase but do not scale with the lever arm).

    Per-slice fit
    -------------
    theta minimises  || Uz - theta*Y ||^2 + || Uy + theta*Z ||^2, giving

        theta = sum(Uz*Y - Uy*Z) / sum(Y^2 + Z^2)

    Two R^2 values are accumulated per slice and averaged across slices,
    AMPLITUDE-WEIGHTED so slices that actually move dominate (a near-still slice
    carries no torsion information and must not dilute the score):

        rigid_uz   : R^2 of Uz vs theta*Y only.  Directly comparable to score_lr
                     but lever-arm aware; this is the lateral torsion fingerprint
                     used to rank modes.
        rigid_uzuy : R^2 of the full (Uz, Uy) field vs the rigid prediction.
                     Returned for reference / diagnostics (stricter, but Uy is
                     noisier on trimmed bodies — see score_tb discussion).

    Parameters
    ----------
    node_xyz      : (nNodes, 3) node coordinates
    uy, uz        : (nNodes,)   modal displacements in Y and Z
    n_slices      : number of X-bins
    min_radius_sq : nodes with Y^2 + Z^2 below this have no usable lever arm and
                    are excluded (same convention as theta_x_profile)
    min_nodes     : a slice needs at least this many valid nodes to be fitted.
                    None (default) picks it from the mesh density: 8 for dense
                    FE bodies (thousands of nodes) down to 3 for a coarse model
                    like the 35-node simple chassis, where a fixed 8 would empty
                    every slice and force the fit to 0.  Pass an int to override.

    Returns
    -------
    (rigid_uz, rigid_uzuy) : both floats in [0, 1]; (0, 0) if no slice qualifies
    """
    X, Y, Z = node_xyz[:, 0], node_xyz[:, 1], node_xyz[:, 2]
    valid   = (Y ** 2 + Z ** 2) > min_radius_sq

    # Adaptive slice-occupancy floor: scale to the mesh so a coarse model is not
    # rejected.  Average valid nodes per slice is n_valid / n_slices; require half
    # of that, clamped to [3, 8].  Dense FE bodies hit the 8 cap (unchanged
    # behaviour); the 35-node simple chassis drops to 3 so its slices qualify.
    if min_nodes is None:
        per_slice = int(valid.sum()) / max(n_slices, 1)
        min_nodes = int(np.clip(round(per_slice / 2.0), 3, 8))

    bins    = np.linspace(X.min(), X.max(), n_slices + 1)
    bi      = np.digitize(X, bins)

    r2_uz, r2_full, weights = [], [], []
    for b in range(1, n_slices + 1):
        m = (bi == b) & valid
        if m.sum() < min_nodes:
            continue
        Ys, Zs, uys, uzs = Y[m], Z[m], uy[m], uz[m]

        # best-fit rotation angle for this slice (closed form)
        den = float(np.sum(Ys ** 2 + Zs ** 2))
        if den < 1e-30:
            continue
        theta = float(np.sum(uzs * Ys - uys * Zs)) / den

        # full (Uy, Uz) rigid-rotation R^2
        ss_res = float(np.sum((uzs - theta * Ys) ** 2 + (uys + theta * Zs) ** 2))
        ss_tot = float(np.sum(uzs ** 2 + uys ** 2))
        if ss_tot < 1e-30:
            continue
        r2_full.append(max(0.0, 1.0 - ss_res / ss_tot))

        # Uz-only (lever-arm) R^2 — the lateral fingerprint used for ranking
        ss_res_z = float(np.sum((uzs - theta * Ys) ** 2))
        ss_tot_z = float(np.sum(uzs ** 2))
        r2_uz.append(max(0.0, 1.0 - ss_res_z / ss_tot_z) if ss_tot_z > 1e-30 else 0.0)

        # weight this slice by how much it moves (ss_tot is a kinetic-energy proxy)
        weights.append(ss_tot)

    if not weights:
        return 0.0, 0.0

    w = np.array(weights)
    w = w / w.sum()
    rigid_uz   = float(np.dot(w, np.array(r2_uz)))
    rigid_uzuy = float(np.dot(w, np.array(r2_full)))
    return rigid_uz, rigid_uzuy


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


def _soft_gate(value: float, x0: float, k: float = 12.0) -> float:
    """
    Smooth 0->1 quality gate: sigmoid centred at x0 with slope k.

    Used to turn a quality sub-score (linearity, centering) into a near-binary
    "pass" factor that does NOT crush the dynamic range of the ranking metric.
    A hard threshold (value >= x0) drops borderline torsion modes; the sigmoid
    keeps them with a graded penalty, so recall stays high while the product no
    longer compresses good modes toward zero the way a raw linear factor does.

        gate(x0) = 0.5,   gate(x0 + 2/k) ~ 0.92,   gate(x0 - 2/k) ~ 0.08
    """
    return float(1.0 / (1.0 + np.exp(-k * (value - x0))))


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
    lin_gate: float = 0.30,
    cen_gate: float = 0.40,
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

    antisym    : torsion strength driving the combined ranking.  It is the
                 GEOMETRIC MEAN of the two rigid-rotation fits,
                 sqrt(rigid_uz * rigid_uzuy) (see rigid_rotation_fit), NOT the
                 correlation score_lr.  A true rigid rotation must satisfy both
                 Uz = theta*Y (lateral, rigid_uz) AND Uy = -theta*Z (vertical,
                 the extra term in rigid_uzuy); the geometric mean is the physical
                 "AND" of the two R^2, so a mode antisymmetric in one plane only
                 is penalised while a genuine torsion mode (high on both) is not.
                 No artificial exponent.  Range [0, 1].

    combined   : antisym * gate(linearity) * gate(centering) * local_veto

                 antisym (the physical fingerprint) drives the ranking and is
                 left ungated so it uses the full [0, 1] range; linearity and
                 centering enter as smooth sigmoid quality gates (see _soft_gate)
                 instead of raw linear factors, which keeps borderline torsion
                 modes (high recall) while still letting the best mode stand out.
                 uniformity is no longer a factor (near-constant across global
                 modes; localisation handled by the local_veto from peak).

    Classification sub-scores (all in [-1, 1], +1 = torsion fingerprint present)
        score_lr : -corr(Uz_left,  Uz_right)   lateral  fingerprint (U_z)
        score_tb : -corr(Uy_top,   Uy_bottom)  vertical fingerprint (U_y)
        score_ly : -corr(Uy_left,  Uy_right)   +1 = lateral-bending/roll antisym
        score_xvar : >= 0, relative variation of per-slice mean U_y along X.
                     ~0 = rigid roll (uniform U_y); large = lateral bending.
    These drive the TORSION/BENDING/ROLLING classification (classify_scores),
    not the ranking; the ranking uses antisym = sqrt(rigid_uz * rigid_uzuy) above.

    Diagnostic sub-score
        rigid_uzuy : R^2 of the full (Uz, Uy) field vs the rigid-rotation
                     prediction.  Stricter than rigid_uz but Uy is noisier on
                     trimmed bodies, so it is reported only, not used to rank.

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
    lin_gate      : linearity value at which the soft gate is 0.5 (below this the
                    theta_x profile is too noisy/flat to trust)
    cen_gate      : centering value at which the soft gate is 0.5 (below this the
                    rotation centre sits too far from the geometric centre)

    Returns
    -------
    dict with keys:
        linearity, centering, antisym, uniformity, peak, combined,
        x0, rigid_uzuy,
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

        linearity = R2_raw * snr_weight
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

    # Rigid-rotation goodness-of-fit (lever-arm aware).  Unlike the correlation
    # scores above, this measures how closely the field matches an ACTUAL
    # rotation, so it sees amplitude antisymmetry and lever-arm scaling, not just
    # phase.  Two R^2 are returned: rigid_uz uses the lateral Uz=theta*Y law only;
    # rigid_uzuy uses the FULL field (Uz=theta*Y AND Uy=-theta*Z).
    rigid_uz, rigid_uzuy = rigid_rotation_fit(
        node_xyz, uy, uz, n_slices, min_radius_sq=min_radius_sq)

    # antisym drives the combined ranking.  A genuine torsion mode must satisfy
    # BOTH rigid-rotation laws simultaneously: Uz = theta*Y (lateral, rigid_uz)
    # AND Uy = -theta*Z (vertical, the extra constraint inside rigid_uzuy).  We
    # therefore use the GEOMETRIC MEAN of the two R^2 as the discriminant:
    #
    #     antisym = sqrt(rigid_uz * rigid_uzuy)
    #
    # This is a physical "AND" of two independent goodness-of-fit measures, not an
    # artificial exponent: a mode with a clean lateral fingerprint but a poor
    # vertical one (antisymmetric in one plane only -> not a true rigid rotation)
    # is penalised, while a real torsion mode, high on both, is barely touched.
    # It sharpens the separation of the top torsion mode from look-alikes on both
    # the simple chassis and the trimmed body without distorting absolute scores
    # (geometric mean, unlike the raw product, does not over-penalise the noisier
    # Uy fit on trimmed bodies).  The correlation scores (score_lr, score_tb,
    # score_ly, score_xvar) are retained above for CLASSIFICATION only.
    ruz  = float(np.clip(rigid_uz,   0.0, 1.0))
    ruzy = float(np.clip(rigid_uzuy, 0.0, 1.0))
    antisym = float(np.sqrt(ruz * ruzy))

    uniformity = spatial_uniformity(ux, uy, uz)

    # Peak-concentration veto: Shannon entropy alone lets through modes whose
    # energy is almost entirely in a tiny region but with a low spread-out
    # background.  peak_concentration goes to ~1 for those; if a mode parks
    # more than peak_thr of its energy in the hottest 1% of nodes it is local,
    # not a global torsion mode, so its combined score is zeroed outright.
    peak = peak_concentration(ux, uy, uz)
    local_veto = 0.0 if peak > peak_thr else 1.0

    # Ranking metric (soft-gate form):
    #   combined = antisym * gate(linearity) * gate(centering) * local_veto
    #
    # antisym (sqrt(rigid_uz * rigid_uzuy)) is the discriminant
    # and is left ungated so it spans the full [0, 1] range — this is what makes
    # the top torsion mode stand out from the rest.  linearity and centering are
    # quality conditions, applied as smooth sigmoid gates rather than raw linear
    # factors: a raw product of four [0,1] terms compresses good modes toward
    # zero (0.8^4 ~ 0.41) and barely separates them, whereas soft gates pass a
    # qualified mode at ~1 and only bite when quality is genuinely poor.
    # uniformity is dropped from the product: it is near-constant across global
    # modes (torsion AND bending), so it only rescaled the ranking without
    # discriminating; localisation is already handled by local_veto (peak).
    # It is still returned for reference / plotting.
    combined = (
        antisym
        * _soft_gate(linearity, lin_gate)
        * _soft_gate(centering, cen_gate)
        * local_veto
    )

    return dict(
        linearity    = linearity,
        centering    = centering,
        antisym      = antisym,
        uniformity   = uniformity,
        peak         = peak,
        combined     = combined,
        x0           = x0,
        rigid_uzuy   = rigid_uzuy,
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
    lin_gate: float = 0.30,
    cen_gate: float = 0.40,
    skip_rigid: bool = True,
    use_body_frame: bool = True,
) -> np.ndarray:
    """
    Compute torsion_score_v2 for every mode.

    peak_thr : modes parking more than this fraction of their energy in the
               hottest 1% of nodes are vetoed (combined forced to 0).
    use_body_frame : if True (default), the node cloud's principal-axis body
               frame is derived once (pca_body_frame) and BOTH coordinates and
               modal displacements are projected into it before scoring, so the
               torsion axis is the geometric longitudinal axis through the robust
               centroid rather than the assumed global X axis through the origin.
               Set False to score in the raw global frame (e.g. for comparison).

    All downstream geometry (theta_x profile, rigid-rotation fit, antisymmetry
    fingerprints, classification) then operates in the body frame, where axis 0
    is longitudinal ("X"), axis 1 lateral ("Y") and axis 2 vertical ("Z"); the
    existing sign conventions (Uz = +theta*Y) are preserved because the frame is
    oriented to stay close to the global axes.

    Returns structured array sorted by ``combined`` descending, with fields:
        mode_idx, freq_hz, combined, linearity, centering, antisym, uniformity,
        peak, x0, rigid_uzuy, score_lr, score_tb, score_ly, score_xvar

    antisym is sqrt(rigid_uz * rigid_uzuy) (geometric mean of the lateral and
    full-field rigid fits) that drives the ranking;
    rigid_uzuy is the stricter full-field fit kept for diagnostics.
    """
    nNodes = len(node_xyz)
    nModes = modes.shape[1]
    Ux_idx = np.arange(0, 6 * nNodes, 6)
    Uy_idx = np.arange(1, 6 * nNodes, 6)
    Uz_idx = np.arange(2, 6 * nNodes, 6)

    # Derive the body frame once and project coordinates into it.  Displacements
    # are rotated per mode below (R @ u, no translation since u is a vector).
    if use_body_frame:
        centre, R, _ = pca_body_frame(node_xyz)
        coords = (node_xyz - centre) @ R.T          # cols: long, lat, vert
    else:
        R = None
        coords = node_xyz

    records = []
    for mi in range(nModes):
        if skip_rigid and freq[mi] < 0.5:
            continue
        ux, uy, uz = modes[Ux_idx, mi], modes[Uy_idx, mi], modes[Uz_idx, mi]
        if R is not None:
            # rotate the displacement field into the body frame (vector, no shift)
            u_body = np.column_stack([ux, uy, uz]) @ R.T
            ux, uy, uz = u_body[:, 0], u_body[:, 1], u_body[:, 2]
        res = torsion_score_v2(
            coords,
            ux, uy, uz,
            n_slices=n_slices, y_threshold=y_threshold,
            z_threshold=z_threshold, min_radius_sq=min_radius_sq,
            peak_thr=peak_thr, lin_gate=lin_gate, cen_gate=cen_gate,
        )
        records.append((
            mi + 1, float(freq[mi]),
            res["combined"], res["linearity"], res["centering"],
            res["antisym"], res["uniformity"], res["peak"], res["x0"],
            res["rigid_uzuy"],
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
        ("rigid_uzuy", float),
        ("score_lr",   float),
        ("score_tb",   float),
        ("score_ly",   float),
        ("score_xvar", float),
    ])
    arr = np.array(records, dtype=dtype)
    return arr[np.argsort(-arr["combined"])]


