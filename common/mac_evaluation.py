"""
Comparative evaluation of MAC methods / weightings.

PURPOSE
-------
The standard MAC pipeline identifies which mode best matches a given reference
(e.g. torsion), but it does not tell you whether one weighting method is
*better* than another.  "Better" here means two things simultaneously:
  1. The target mode has a HIGH MAC  (sensitivity  -- the method finds torsion)
  2. All other modes have LOW MACs   (specificity  -- the method rejects non-torsion)

This module quantifies both properties so different methods (Identity, Mass,
Stiffness, Energy, Shear, ...) can be ranked on a common scale.

METRICS EXPLAINED
-----------------
Given the column of MAC values for one method vs the torsion reference,
i.e. col[i] = MAC(mode_i, torsion):

  mac_target       MAC of the single best-matching mode.
                   Ideal value: 1.0.  Tells you whether torsion is *found*.

  target_rank      Position of that mode when all modes are sorted by MAC
                   descending.  Ideal value: 1 (it is the top mode).
                   If rank > 1, another mode has a higher MAC than the real
                   torsion mode -- a false positive.

  FP mean          Mean MAC of all modes EXCEPT the best match.
                   Ideal value: ~0.  High FP mean means "many modes look like
                   torsion", which makes the identification unreliable.

  FP max           Worst-case false positive -- the highest MAC among the
                   non-target modes.  Even if FP mean is low, a single FP max
                   close to mac_target is dangerous.

  FP p95           95th percentile of non-target MACs.  More robust than
                   FP max when there are outlier modes.

  discrimination   mac_target / FP mean.  The signal-to-noise ratio of the
                   identification.  A ratio of 10 means the target mode has
                   10x higher MAC than the average non-target mode.
                   This is the primary ranking criterion.

  specificity      Fraction of non-target modes whose MAC is below threshold
                   (default 0.1).  A value of 1.0 means every non-torsional
                   mode is cleanly rejected.  A value of 0.5 means half the
                   modes have MAC >= 0.1 against torsion.

INTERPRETING THE PLOTS
----------------------
plot_evaluation -- 4-panel bar chart, one bar per method:
  Top-left  : MAC(target)       -- taller bar = method finds torsion better
  Top-right : FP mean MAC       -- shorter bar = less contamination from other modes
  Bot-left  : Discrimination    -- taller bar = cleaner separation signal/noise
  Bot-right : Specificity       -- taller bar = more modes correctly rejected

  Reference lines: dashed black at 0.8 (good MAC threshold) and 0.9 (good
  specificity threshold).  A method is "good" when its bar clears 0.8 in
  top-left and bot-right, and stays below 0.1 in top-right.

plot_mac_distribution -- one curve per method, x-axis = mode number:
  Each point is MAC(mode_i, torsion).  The ideal curve has one sharp spike
  at the torsion mode and is flat near zero everywhere else.  Curves that
  are "lumpy" (many elevated modes) indicate the method cannot discriminate.

Typical use
-----------
from common.mac_evaluation import evaluate_mac_methods, print_evaluation_table, plot_evaluation

results = evaluate_mac_methods(
    mac_matrices   = mac_matrices,   # dict: label -> (nModes, nRefs)
    target_ref_idx = 0,              # column index of the target reference
    threshold      = 0.1,            # MAC below this counts as "correctly rejected"
)
print_evaluation_table(results)
plot_evaluation(results, freq=freq)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class MethodMetrics:
    """All discriminability metrics for one MAC method/variant."""
    label: str

    # --- Target mode identification ---
    mac_target:     float   # MAC of the best-identified target mode
    target_mode_i:  int     # index of that mode (0-based elastic)
    target_rank:    int     # rank of the target mode in the MAC-sorted list (1 = best)

    # --- False-positive control (modes that are NOT the target) ---
    mac_fp_mean:    float   # mean MAC of non-target modes against the target reference
    mac_fp_max:     float   # max  MAC of non-target modes against the target reference
    mac_fp_p95:     float   # 95th percentile of non-target MAC values

    # --- Discrimination ratio ---
    discrimination: float   # mac_target / mac_fp_mean  (higher = better separation)

    # --- Specificity ---
    specificity:    float   # fraction of non-target modes with MAC < threshold

    # --- AutoMAC diagonal quality (optional, set later if wanted) ---
    automac_offdiag_mean: float = 0.0  # mean off-diagonal AutoMAC (0 = perfectly orthogonal)


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def _compute_metrics(
    label: str,
    mac: np.ndarray,          # (nModes, nRefs)
    target_ref_idx: int,
    threshold: float,
) -> MethodMetrics:
    """Compute all metrics for a single MAC matrix."""
    col = mac[:, target_ref_idx]          # (nModes,) — MAC vs target reference

    target_mode_i  = int(np.argmax(col))
    mac_target     = float(col[target_mode_i])

    # rank: position of the target mode when modes are sorted descending by MAC
    target_rank    = int(np.sum(col > mac_target)) + 1  # ties broken conservatively

    # non-target modes
    mask_fp        = np.ones(len(col), dtype=bool)
    mask_fp[target_mode_i] = False
    fp_vals        = col[mask_fp]

    mac_fp_mean    = float(np.mean(fp_vals))   if len(fp_vals) > 0 else 0.0
    mac_fp_max     = float(np.max(fp_vals))    if len(fp_vals) > 0 else 0.0
    mac_fp_p95     = float(np.percentile(fp_vals, 95)) if len(fp_vals) > 0 else 0.0
    specificity    = float(np.mean(fp_vals < threshold)) if len(fp_vals) > 0 else 1.0
    discrimination = mac_target / mac_fp_mean  if mac_fp_mean > 1e-12 else np.inf

    return MethodMetrics(
        label          = label,
        mac_target     = mac_target,
        target_mode_i  = target_mode_i,
        target_rank    = target_rank,
        mac_fp_mean    = mac_fp_mean,
        mac_fp_max     = mac_fp_max,
        mac_fp_p95     = mac_fp_p95,
        discrimination = discrimination,
        specificity    = specificity,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_mac_methods(
    mac_matrices:   dict,         # label -> (nModes, nRefs)
    target_ref_idx: int = 0,
    threshold:      float = 0.1,
    compute_automac: bool = False,
    modes: Optional[np.ndarray] = None,   # (nDOF, nModes) — needed for AutoMAC
    W=None,                                # weighting for AutoMAC — dict label->W or single W
) -> list[MethodMetrics]:
    """
    Compute discriminability metrics for every MAC variant.

    Parameters
    ----------
    mac_matrices   : dict label -> (nModes, nRefs) MAC matrix
    target_ref_idx : column index of the target reference (default 0)
    threshold      : MAC threshold below which a mode is "correctly rejected" (default 0.1)
    compute_automac: if True, also compute off-diagonal AutoMAC mean per method
    modes          : required if compute_automac=True
    W              : weighting matrix (or dict label->W) for AutoMAC computation

    Returns
    -------
    List of MethodMetrics, one per MAC variant, sorted by discrimination ratio descending.
    """
    results = []
    for label, mac in mac_matrices.items():
        m = _compute_metrics(label, mac, target_ref_idx, threshold)

        if compute_automac and modes is not None:
            from common.mac_core import compute_mac
            w = W.get(label) if isinstance(W, dict) else W
            auto = compute_mac(modes, modes, w)
            n = auto.shape[0]
            if n > 1:
                mask = ~np.eye(n, dtype=bool)
                m.automac_offdiag_mean = float(np.mean(auto[mask]))

        results.append(m)

    results.sort(key=lambda x: x.discrimination, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Console table
# ---------------------------------------------------------------------------

_COL_W = 14

def print_evaluation_table(
    results:        list[MethodMetrics],
    target_name:    str = "Target",
    threshold:      float = 0.1,
    mode_offset:    int = 0,
    freq:           Optional[np.ndarray] = None,
) -> None:
    """
    Print a ranked comparison table of MAC methods.

    Columns
    -------
    Method         : variant label
    MAC(target)    : MAC of the best-matching mode for the target reference
    Mode           : mode number (and frequency if freq provided)
    Rank           : rank of target mode in MAC-sorted list
    FP mean        : mean MAC of all other modes vs target (lower = better)
    FP max         : worst-case false positive MAC
    FP p95         : 95th percentile false positive
    Discrim.ratio  : MAC_target / FP_mean  (higher = better)
    Specificity    : fraction of non-target modes with MAC < threshold
    """
    cols = ["Method", "MAC(target)", "Mode(idx)", "Rank",
            "FP mean", "FP max", "FP p95", "Discrim.", f"Specif.(>{threshold})"]
    col_w = [22] + [_COL_W] * (len(cols) - 1)
    sep   = "-"

    header = "".join(f"{c:^{w}}" for c, w in zip(cols, col_w))
    print(f"\n{'='*len(header)}")
    print(f"  MAC Method Evaluation  --  target: {target_name}")
    print(f"{'='*len(header)}")
    print(header)
    print(sep * len(header))

    for m in results:
        mode_n = mode_offset + m.target_mode_i + 1
        if freq is not None and m.target_mode_i < len(freq):
            mode_str = f"{mode_n} ({freq[m.target_mode_i]:.1f}Hz)"
        else:
            mode_str = str(mode_n)

        disc_str = f"{m.discrimination:.1f}" if np.isfinite(m.discrimination) else "inf"
        vals = [
            m.label[:21],
            f"{m.mac_target:.4f}",
            mode_str,
            str(m.target_rank),
            f"{m.mac_fp_mean:.4f}",
            f"{m.mac_fp_max:.4f}",
            f"{m.mac_fp_p95:.4f}",
            disc_str,
            f"{m.specificity:.3f}",
        ]
        print("".join(f"{v:^{w}}" for v, w in zip(vals, col_w)))

    print(sep * len(header))
    print(f"  Sorted by discrimination ratio (descending).  Threshold = {threshold}")


# ---------------------------------------------------------------------------
# Bar chart
# ---------------------------------------------------------------------------

def plot_evaluation(
    results:     list[MethodMetrics],
    freq:        Optional[np.ndarray] = None,
    target_name: str = "Target",
    threshold:   float = 0.1,
    mode_offset: int = 0,
    title:       str = "",
) -> plt.Figure:
    """
    Bar chart comparing methods across 4 panels:
      1. MAC(target)  — higher is better
      2. FP mean MAC  — lower is better
      3. Discrimination ratio  — higher is better
      4. Specificity  — higher is better
    """
    labels  = [m.label for m in results]
    n       = len(labels)
    x       = np.arange(n)
    colors  = cm.tab20(np.linspace(0, 1, n))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(title or f"MAC Method Comparison — target: {target_name}", fontsize=12)

    def _bar(ax, values, ylabel, good_high: bool, ref_line=None, fmt=".3f"):
        bars = ax.bar(x, values, color=colors, alpha=0.85)
        for bar, v in zip(bars, values):
            vstr = f"{v:.1f}" if abs(v) > 10 else f"{v:{fmt}}"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.01,
                    vstr, ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=9)
        arrow = "↑ better" if good_high else "↓ better"
        ax.set_title(f"{ylabel}  ({arrow})", fontsize=9)
        if ref_line is not None:
            ax.axhline(ref_line, color="k", linewidth=0.8, linestyle="--", alpha=0.4)
        ax.grid(axis="y", alpha=0.3)
        ymax = max(values) * 1.15 if max(values) > 0 else 1.0
        ax.set_ylim(0, ymax)

    _bar(axes[0, 0], [m.mac_target for m in results],
         "MAC(target)", good_high=True, ref_line=0.8)

    _bar(axes[0, 1], [m.mac_fp_mean for m in results],
         "FP mean MAC", good_high=False, ref_line=threshold)

    disc_vals = [m.discrimination if np.isfinite(m.discrimination) else
                 max(r.discrimination for r in results if np.isfinite(r.discrimination)) * 1.05
                 for m in results]
    _bar(axes[1, 0], disc_vals,
         "Discrimination ratio", good_high=True, fmt=".1f")

    _bar(axes[1, 1], [m.specificity for m in results],
         f"Specificity (MAC<{threshold})", good_high=True, ref_line=0.9)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Per-mode detail: MAC distribution across methods for one reference
# ---------------------------------------------------------------------------

def plot_mac_distribution(
    mac_matrices:   dict,
    target_ref_idx: int = 0,
    freq:           Optional[np.ndarray] = None,
    target_name:    str = "Target",
    mode_offset:    int = 0,
    title:          str = "",
) -> plt.Figure:
    """
    Line/scatter plot showing MAC vs target reference for every mode,
    one curve per method. Allows visual inspection of which modes are
    "lit up" by each method and how cleanly the target is isolated.
    """
    labels = list(mac_matrices.keys())
    n_vars = len(labels)
    colors = cm.tab20(np.linspace(0, 1, n_vars))

    n_modes = next(iter(mac_matrices.values())).shape[0]
    x       = np.arange(n_modes)
    xlabels = []
    for i in x:
        if freq is not None and i < len(freq):
            xlabels.append(f"{mode_offset+i+1}\n({freq[i]:.1f})")
        else:
            xlabels.append(str(mode_offset + i + 1))

    fig, ax = plt.subplots(figsize=(max(12, n_modes * 0.35), 5))
    for label, color in zip(labels, colors):
        col = mac_matrices[label][:, target_ref_idx]
        ax.plot(x, col, marker="o", markersize=3, linewidth=1,
                label=label, color=color, alpha=0.8)

    ax.axhline(0.8, color="k", linewidth=0.8, linestyle="--", alpha=0.3)
    ax.axhline(0.1, color="r", linewidth=0.6, linestyle=":",  alpha=0.3,
               label=f"threshold 0.1")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=6)
    ax.set_ylabel(f"MAC vs {target_name}", fontsize=9)
    ax.set_title(title or f"MAC distribution — all modes vs {target_name}", fontsize=10)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig
