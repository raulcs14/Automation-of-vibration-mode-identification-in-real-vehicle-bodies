"""
MAC matrix heatmap visualization.
Equivalent to plotMACMatrix.m
"""

import numpy as np
import matplotlib.pyplot as plt
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
