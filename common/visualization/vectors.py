"""
Force and displacement vector visualization at nodes and subdomains.
Equivalent to plotNodeVectors.m and plotSubdomainVectors.m
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


def plot_node_vectors(ax: plt.Axes, node_coordinates: np.ndarray,
                      vectors: np.ndarray, element_nodes: np.ndarray,
                      title: str = "", color: str = "r") -> None:
    """
    Plot 3D quiver arrows at each node representing a force or displacement vector.

    Args:
        ax: Matplotlib 3D axes
        node_coordinates: (nNodes, 3)
        vectors: (GDof,) full 6-DOF vector; only Ux, Uy, Uz are drawn
        element_nodes: (nElements, 2) for background mesh overlay
        title: Plot title
        color: Arrow color
    """
    raise NotImplementedError


def plot_subdomain_vectors(ax: plt.Axes, node_coordinates: np.ndarray,
                            element_nodes: np.ndarray,
                            subdomains: Dict[str, List[int]],
                            vectors: np.ndarray,
                            title: str = "") -> None:
    """
    Plot one averaged arrow per subdomain at the zone centroid.

    Args:
        ax: Matplotlib 3D axes
        node_coordinates: (nNodes, 3)
        element_nodes: (nElements, 2)
        subdomains: zone name → list of node indices
        vectors: (3·nZones,) or (3·nZones, nModes) — if 2D, mode index 0 is used
        title: Plot title
    """
    raise NotImplementedError
