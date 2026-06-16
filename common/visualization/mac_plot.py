"""
MAC matrix visualization: heatmap and bar chart.
Equivalent to plotMACMatrix.m
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import List, Optional


def plot_mac_matrix(mac: np.ndarray, mode_labels: List[str],
                    ref_labels: List[str],
                    title: str = "MAC Matrix",
                    ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    Plot MAC matrix as a heatmap with numeric values inside each cell.

    Color scale [0, 1]. Text is black if MAC > 0.6, white otherwise.

    Args:
        mac: (nModes, nRefs) MAC matrix
        mode_labels: Y-axis labels (e.g. "Mode 7 — 12.3 Hz")
        ref_labels: X-axis labels (reference motion names)
        title: Figure title
        ax: Existing axes to draw into (optional)

    Returns:
        fig: Matplotlib figure
    """
    n_modes, n_refs = mac.shape

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(8, n_refs * 0.9), max(6, n_modes * 0.5)))
    else:
        fig = ax.get_figure()

    im = ax.imshow(mac, vmin=0, vmax=1, aspect="auto", origin="upper")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(n_refs))
    ax.set_xticklabels(ref_labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(n_modes))
    ax.set_yticklabels(mode_labels, fontsize=8)
    ax.set_xlabel("Reference motion")
    ax.set_ylabel("Dynamic mode")
    ax.set_title(title)

    for i in range(n_modes):
        for j in range(n_refs):
            val = mac[i, j]
            color = "k" if val > 0.6 else "w"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, fontweight="bold", color=color)

    fig.tight_layout()
    return fig


def plot_mac_bar(
    mac_matrices: dict,
    idx: np.ndarray,
    freq: np.ndarray,
    short_names: List[str],
    title: str,
    mode_offset: int = 0,
) -> plt.Figure:
    """
    Bar chart of the best MAC value per mode across multiple MAC variants.

    Each group of bars corresponds to one dynamic mode; each bar within the group
    is one MAC variant (weighting / subdomain). The winning reference name is
    printed inside the bar and the MAC value just above it.

    Parameters
    ----------
    mac_matrices : dict[str, np.ndarray]  variant_label → (nModes, nRefs) MAC matrix
    idx          : indices of the modes to display (into the full freq array)
    freq         : (nModes,) frequencies in Hz
    short_names  : reference motion short labels (same order as MAC columns)
    title        : figure title
    mode_offset  : added to mode index for display (use when skipping rigid modes)

    Returns
    -------
    fig : matplotlib Figure
    """
    labels  = list(mac_matrices.keys())
    n_modes = len(idx)
    n_vars  = len(labels)
    x       = np.arange(n_modes)
    bar_w   = 0.8 / n_vars
    colors  = cm.tab20(np.linspace(0, 1, n_vars))

    fig, ax = plt.subplots(figsize=(max(14, n_modes * 1.0), 6))
    for k, (label, color) in enumerate(zip(labels, colors)):
        mac_block = mac_matrices[label][idx]
        best_j    = mac_block.argmax(axis=1)
        vals      = mac_block.max(axis=1)
        offset    = (k - n_vars / 2 + 0.5) * bar_w
        bars      = ax.bar(x + offset, vals, width=bar_w, label=label,
                           color=color, alpha=0.85)
        for bar, v, j in zip(bars, vals, best_j):
            if v > 0.15:
                sname = short_names[j] if j < len(short_names) else f"Ref{j+1}"
                cx = bar.get_x() + bar.get_width() / 2
                ax.text(cx, v / 2, sname, ha="center", va="center",
                        fontsize=6, rotation=90, color="white", fontweight="bold")
                ax.text(cx, v + 0.01, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=6)

    global_nums = mode_offset + idx + 1
    xlabels = [f"Mode {global_nums[i]}\n({freq[idx[i]]:.1f} Hz)"
               for i in range(n_modes)]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=7)
    ax.set_ylabel("Best MAC value")
    ax.set_title(title)
    ax.set_ylim(0, 1.12)
    ax.axhline(0.8, color="k", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.axhline(0.6, color="k", linewidth=0.6, linestyle=":",  alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig
