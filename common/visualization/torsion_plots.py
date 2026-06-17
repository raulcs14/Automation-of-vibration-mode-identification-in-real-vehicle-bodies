"""
Torsion identification plots.

Public API
----------
plot_cross_sections   : 3-D scatter of model nodes coloured by X-slice
plot_mode_map         : score_lr vs score_tb classification map (Fig 1)
plot_theta_profiles   : theta_x(X) profiles for top torsional modes  (Fig 2)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Helpers shared across plots
# ---------------------------------------------------------------------------

COLORS = {
    "TORSION":    "firebrick",
    "ROLLING":    "darkorange",
    "BENDING-V":  "steelblue",
    "BENDING-L":  "seagreen",
    "LOCAL/MIXTO": "gray",
}


def classify_scores(score_lr: float, score_tb: float, score_ly: float,
                    score_xvar: float = 0.0,
                    thr: float = 0.5, xvar_thr: float = 0.3) -> str:
    """
    Classify a mode from its antisymmetry fingerprints.

    score_lr   : -corr(Uz_left,  Uz_right)   lateral  torsion fingerprint (U_z)
    score_tb   : -corr(Uy_top,   Uy_bottom)  vertical torsion fingerprint (U_y)
    score_ly   : -corr(Uy_left,  Uy_right)   lateral antisymmetry
    score_xvar : variation of per-slice mean U_y along X (roll ~0, bending large)

    TORSION is decided by score_lr alone: on real trimmed bodies the lateral
    left/right U_z fingerprint is the clean, reliable signature of a rotation
    about X, whereas the U_y-based score_tb is contaminated by local/lumped-mass
    motion and is too noisy to require.  score_tb is reported and used as a
    confidence boost (see torsion_score_v2.antisym), not as a gate.

    A lateral U_y motion (top/bottom in phase -> score_tb < -thr) with weak
    lateral torsion is either a rigid roll (U_y uniform along X -> score_xvar
    small) or a lateral bending mode (U_y curves along X -> score_xvar large).
    score_xvar splits the two; without it they are indistinguishable from U_y.
    """
    if score_lr > thr:                 return "TORSION"      # lateral U_z antisym
    if score_lr < -thr:                return "BENDING-V"    # vertical bending
    if score_tb < -thr or score_ly < -thr:
        # lateral U_y motion: roll if uniform along X, bending if it curves
        return "BENDING-L" if score_xvar > xvar_thr else "ROLLING"
    return "LOCAL/MIXTO"


def _classify_row(row, thr: float = 0.5) -> str:
    """Classify a structured-array record from scan_torsion_scores_v2."""
    return classify_scores(
        float(row["score_lr"]), float(row["score_tb"]),
        float(row["score_ly"]), float(row["score_xvar"]), thr)


# ---------------------------------------------------------------------------
# Figure 0 — 3-D cross-section geometry
# ---------------------------------------------------------------------------

def plot_cross_sections(
    node_xyz: np.ndarray,
    n_slices: int = 20,
    ax=None,
    max_nodes_shown: int = 5000,
):
    """
    3-D scatter of all model nodes coloured by their X-slice assignment.

    Parameters
    ----------
    node_xyz        : (nNodes, 3) — node coordinates in mm
    n_slices        : number of X-bins (must match the value used in analysis)
    ax              : existing 3-D Axes to draw on; created if None
    max_nodes_shown : subsample if the model has more nodes than this

    Returns
    -------
    ax : the 3-D Axes object
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    X, Y, Z = node_xyz[:, 0], node_xyz[:, 1], node_xyz[:, 2]
    bins = np.linspace(X.min(), X.max(), n_slices + 1)
    bi   = np.digitize(X, bins)

    idx = np.arange(len(node_xyz))
    if len(idx) > max_nodes_shown:
        rng = np.random.default_rng(42)
        idx = rng.choice(idx, size=max_nodes_shown, replace=False)

    cmap   = plt.get_cmap("tab20", n_slices)
    colors = cmap(((bi[idx] - 1) / max(n_slices - 1, 1)))

    if ax is None:
        fig = plt.figure(figsize=(14, 8))
        ax  = fig.add_subplot(111, projection="3d")

    ax.scatter(X[idx], Y[idx], Z[idx], c=colors, s=2, alpha=0.5, linewidths=0)

    # filled slice volumes alternating two light colours
    y_lo, y_hi = float(Y.min()), float(Y.max())
    z_lo, z_hi = float(Z.min()), float(Z.max())
    panel_colors = ["#d0e8f7", "#f7e8d0"]
    yy = np.array([y_lo, y_hi])
    zz = np.array([z_lo, z_hi])
    YY, ZZ = np.meshgrid(yy, zz)
    for b_idx, (x0, x1) in enumerate(zip(bins[:-1], bins[1:])):
        fc = panel_colors[b_idx % 2]
        for xx in (x0, x1):
            XX = np.full_like(YY, xx)
            ax.plot_surface(XX, YY, ZZ, color=fc, alpha=0.15,
                            linewidth=0, antialiased=False)

    # legend: representative subset of slices
    legend_step = max(1, n_slices // 10)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=cmap(s / max(n_slices - 1, 1)),
                   markersize=6, label=f"Section {s}")
        for s in range(0, n_slices, legend_step)
    ]
    ax.legend(handles=handles, fontsize=6, loc="upper left",
              title="X-slice", title_fontsize=7)

    # equal aspect ratio so the car shape is preserved
    x_c = (X.max() + X.min()) / 2.0
    y_c = (Y.max() + Y.min()) / 2.0
    z_c = (Z.max() + Z.min()) / 2.0
    half = max(X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()) / 2.0
    ax.set_xlim(x_c - half, x_c + half)
    ax.set_ylim(y_c - half, y_c + half)
    ax.set_zlim(z_c - half, z_c + half)

    x_step = float(np.median(np.diff(bins)))
    ax.set_xlabel("X [mm]", fontsize=8)
    ax.set_ylabel("Y [mm]", fontsize=8)
    ax.set_zlabel("Z [mm]", fontsize=8)
    ax.set_title(
        f"Cross-Section Identification — {n_slices} sections (step={x_step:.1f} mm)\n"
        f"Model: {len(node_xyz)} nodes,  X: [{X.min():.1f}, {X.max():.1f}] mm",
        fontsize=8,
    )
    ax.tick_params(labelsize=6)
    return ax


# ---------------------------------------------------------------------------
# Figure 1 — score_lr vs score_tb classification map
# ---------------------------------------------------------------------------

def plot_mode_map(results, model_label: str, thr: float = 0.5):
    """
    Scatter plot of score_lr (lateral U_z fingerprint, X) vs score_tb (vertical
    U_y fingerprint, Y) for all modes, sized by combined score.

    The shaded regions match what classify_scores() actually does: TORSION is
    decided by score_lr alone, so the whole right band (score_lr > thr) is
    torsion regardless of score_tb.  A high-score_lr mode that also sits low in
    score_tb (e.g. modes 20/32) is torsion coupled with some lateral motion —
    still torsion, not rolling.  score_tb is the colour-coded confidence axis,
    not a gate.

    Parameters
    ----------
    results     : structured array from scan_torsion_scores_v2
    model_label : string shown in the title
    thr         : classification threshold (default 0.5)

    Returns
    -------
    fig, ax
    """
    lim = 1.1
    fig, ax = plt.subplots(figsize=(9, 8))

    # TORSION region = whole right band score_lr > thr (classifier uses score_lr
    # alone).  Darker sub-band where score_tb also agrees (cleanest torsion).
    ax.add_patch(mpatches.Rectangle((thr, -lim), lim - thr, 2 * lim,
                                    alpha=0.05, color="firebrick", zorder=0))
    ax.add_patch(mpatches.Rectangle((thr, thr), lim - thr, lim - thr,
                                    alpha=0.06, color="firebrick", zorder=0))
    # BENDING-V region = left band score_lr < -thr
    ax.add_patch(mpatches.Rectangle((-lim, -lim), lim - thr, 2 * lim,
                                    alpha=0.05, color="steelblue", zorder=0))
    # ROLLING / BENDING-L region = central band, score_tb < -thr (only where
    # score_lr is not torsion/bending-V)
    ax.add_patch(mpatches.Rectangle((-thr, -lim), 2 * thr, lim - thr,
                                    alpha=0.06, color="darkorange", zorder=0))

    for row in results:
        slr = float(row["score_lr"]); stb = float(row["score_tb"])
        comb = float(row["combined"])
        mtype = _classify_row(row, thr)
        ax.scatter(slr, stb, color=COLORS[mtype], s=12 + 300 * comb,
                   zorder=3, alpha=0.75, edgecolors="white", linewidths=0.3)
        ax.annotate(str(int(row["mode_idx"])),
                    xy=(slr, stb), xytext=(3, 2), textcoords="offset points",
                    fontsize=6, color=COLORS[mtype])

    for v in [-thr, 0, thr]:
        ax.axhline(v, color="gray", lw=0.6, ls="--" if v == 0 else ":")
        ax.axvline(v, color="gray", lw=0.6, ls="--" if v == 0 else ":")

    kw = dict(fontsize=8, alpha=0.6, ha="center", fontweight="bold")
    ax.text( 0.78,  0.92, "TORSION (clean)",   color="firebrick",  **kw)
    ax.text( 0.78, -0.92, "TORSION (+lateral)", color="firebrick", **kw)
    ax.text( 0.0,  -0.92, "ROLLING / BEND-L",  color="darkorange", **kw)
    ax.text(-0.78,  0.0,  "BENDING-V",         color="steelblue",  **kw)

    patches = [mpatches.Patch(color=c, label=t) for t, c in COLORS.items()]
    for s, lbl in [(12, "combined~0"), (62, "combined=0.15"), (162, "combined=0.5")]:
        ax.scatter([], [], color="k", s=s, alpha=0.6, label=lbl)
    ax.legend(handles=patches + ax.get_legend_handles_labels()[0][-3:],
              loc="lower left", fontsize=7, ncol=2)
    ax.set_xlabel("score_lr  =  -corr(Uz_left, Uz_right)   (+1 = torsion,  -1 = vertical bending)",
                  fontsize=9)
    ax.set_ylabel("score_tb  =  -corr(Uy_top, Uy_bottom)   (+1 = clean torsion,  -1 = coupled lateral / roll)",
                  fontsize=9)
    ax.set_title(
        f"Mode classification [{model_label}]  —  size = combined score\n"
        f"TORSION = score_lr > {thr} (right band).  score_tb is the confidence/coupling axis, not a gate."
    )
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Figure 2 — theta_x profiles for top torsional modes
# ---------------------------------------------------------------------------

def plot_theta_profiles(
    results,
    node_xyz: np.ndarray,
    modes: np.ndarray,
    n_slices: int,
    model_label: str,
    top_modes: int = 6,
    thr: float = 0.5,
    min_rel: float = 0.1,
):
    """
    Grid of theta_x(X) profiles for the top TORSION modes.

    combined_rel = combined / max(combined over all TORSION modes), so the
    strongest torsion mode is 100 %.  Only modes with combined_rel > min_rel
    are shown, capped at top_modes.

    Parameters
    ----------
    results      : structured array from scan_torsion_scores_v2
    node_xyz     : (nNodes, 3)
    modes        : (6*nNodes, nModes)
    n_slices     : number of X-bins used in the analysis
    model_label  : string shown in the suptitle
    top_modes    : maximum number of modes to show
    thr          : classification threshold
    min_rel      : minimum relative score (fraction of the best torsion mode)
                   for a mode to be plotted (default 0.1 = 10 %)

    Returns
    -------
    fig, axes  or  (None, None) if no torsion modes found
    """
    from common.torsion_analysis import theta_x_profile

    all_torsion = [r for r in results if _classify_row(r, thr) == "TORSION"]
    if not all_torsion:
        return None, None

    # relative term: 1.0 for the strongest torsion mode in the whole model
    comb_max = max(float(r["combined"]) for r in all_torsion)
    comb_max = comb_max if comb_max > 0 else 1.0

    torsion_rows = [r for r in all_torsion
                    if float(r["combined"]) / comb_max > min_rel][:top_modes]

    if not torsion_rows:
        return None, None

    nNodes = len(node_xyz)
    Uy_idx = np.arange(1, 6 * nNodes, 6)
    Uz_idx = np.arange(2, 6 * nNodes, 6)
    X_min  = float(node_xyz[:, 0].min())
    X_max  = float(node_xyz[:, 0].max())

    ncols = min(3, len(torsion_rows))
    nrows = int(np.ceil(len(torsion_rows) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes = axes.flatten()

    for i, row in enumerate(torsion_rows):
        mi    = int(row["mode_idx"]) - 1
        fhz   = float(row["freq_hz"])
        comb  = float(row["combined"])
        rel   = comb / comb_max
        lin   = float(row["linearity"])
        cen   = float(row["centering"])
        ant   = float(row["antisym"])      # rigid_uz, drives the ranking
        x0    = float(row["x0"])
        mtype = _classify_row(row, thr)

        uy = modes[Uy_idx, mi]; uz = modes[Uz_idx, mi]
        x_c, th = theta_x_profile(node_xyz, uy, uz, n_slices=n_slices)

        ax = axes[i]
        ax.plot(x_c, th, "o-", color=COLORS.get(mtype, "steelblue"),
                lw=1.5, ms=4, label="theta_x")
        if len(x_c) >= 4:
            coeffs = np.polyfit(x_c, th, 1)
            x_fit  = np.array([x_c.min(), x_c.max()])
            ax.plot(x_fit, np.polyval(coeffs, x_fit), "--", color="tomato",
                    lw=1.3, label=f"lineal (lin={lin:.2f})")
            if X_min <= x0 <= X_max:
                ax.axvline(x0, color="tomato", lw=0.9, ls=":",
                           label=f"x0={x0:.0f} mm")
        ax.axhline(0, color="k", lw=0.7, ls=":")
        ax.set_xlabel("X (mm)", fontsize=8)
        ax.set_ylabel("theta_x (rad/u)", fontsize=8)
        ax.set_title(
            f"Mode {int(row['mode_idx'])}  {fhz:.2f} Hz  [{mtype}]\n"
            f"comb={comb:.3f}  (rel={rel*100:.0f}%)\n"
            f"lin={lin:.2f}  cen={cen:.2f}  ant={ant:.2f}",
            fontsize=7.5,
        )
        ax.legend(fontsize=6); ax.grid(True, alpha=0.3); ax.tick_params(labelsize=7)

    for j in range(len(torsion_rows), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Top {len(torsion_rows)} TORSION/ROLLING modes [{model_label}]"
        f"  —  combined = antisym x gate(lin) x gate(cen)  ({n_slices} slices)",
        fontsize=10,
    )
    fig.tight_layout()
    return fig, axes
