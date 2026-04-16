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
    raise NotImplementedError
