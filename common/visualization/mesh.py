"""
3D frame mesh plotting.
Equivalent to drawingMesh.m and drawInterpolatedFrame3D.m
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def draw_mesh(ax: plt.Axes, node_coordinates: np.ndarray,
              element_nodes: np.ndarray,
              displacements: Optional[np.ndarray] = None,
              scale: float = 1.0,
              linestyle: str = "k-") -> None:
    """
    Plot undeformed or deformed 3D frame mesh (straight lines between nodes).

    Args:
        ax: Matplotlib 3D axes
        node_coordinates: (nNodes, 3)
        element_nodes: (nElements, 2) 0-based node indices
        displacements: (GDof,) if provided, deformed shape is shown
        scale: Displacement amplification factor
        linestyle: Matplotlib line spec
    """
    raise NotImplementedError


def draw_interpolated_frame(ax: plt.Axes, node_coordinates: np.ndarray,
                             element_nodes: np.ndarray,
                             displacements: np.ndarray,
                             scale: float = 1.0,
                             n_points: int = 20) -> None:
    """
    Plot smooth deformed shape using Hermite cubic interpolation along each element.

    Shape functions: N1=1-3s²+2s³, N2=s-2s²+s³, N3=3s²-2s³, N4=-s²+s³

    Args:
        ax: Matplotlib 3D axes
        node_coordinates: (nNodes, 3)
        element_nodes: (nElements, 2)
        displacements: (GDof,) full 6-DOF displacement vector
        scale: Amplification factor
        n_points: Interpolation points per element
    """
    raise NotImplementedError
