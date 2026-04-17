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
    from simple_model.geometry.chassis import _draw_mesh_lines, _set_equal_axes

    vx = vectors[0::6]
    vy = vectors[1::6]
    vz = vectors[2::6]
    vmag = np.sqrt(vx**2 + vy**2 + vz**2)
    mask = vmag > 0
    scale = 1.0 / vmag[mask].max()

    _draw_mesh_lines(ax, node_coordinates, element_nodes, linestyle="k--")
    ax.quiver(node_coordinates[mask, 0], node_coordinates[mask, 1], node_coordinates[mask, 2],
              vx[mask] * scale, vy[mask] * scale, vz[mask] * scale,
              length=1.0, normalize=False, color=color, linewidth=2, arrow_length_ratio=0.3)
    _set_equal_axes(ax, node_coordinates)
    ax.set_title(title)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.view_init(elev=-45, azim=170)
    ax.grid(True)


def plot_subdomain_vectors(ax: plt.Axes, node_coordinates: np.ndarray,
                            element_nodes: np.ndarray,
                            subdomains: Dict[str, List[int]],
                            vectors: np.ndarray,
                            scale_factor: float = 5.0,
                            mode_index: int = 0,
                            title: str = "") -> None:
    """
    Plot one averaged arrow per subdomain at the zone centroid.

    Args:
        ax: Matplotlib 3D axes
        node_coordinates: (nNodes, 3)
        element_nodes: (nElements, 2)
        subdomains: zone name → list of node indices
        vectors: (3·nZones,) or (3·nZones, nModes); if 2D, mode_index selects the column
        scale_factor: visual scaling multiplier (default 5, matches MATLAB)
        mode_index: 0-based column to extract when vectors is 2D
        title: Plot title
    """
    from simple_model.geometry.chassis import _draw_mesh_lines, _set_equal_axes

    zone_names = list(subdomains.keys())
    n_zones = len(zone_names)

    # Normalise to (n_zones, 3)
    v = np.asarray(vectors)
    if v.ndim == 1:
        v_red = v.reshape(n_zones, 3)
    else:
        v_red = v[:, mode_index].reshape(n_zones, 3)

    vec_mag = np.linalg.norm(v_red, axis=1)
    scale = scale_factor / vec_mag.max()

    _draw_mesh_lines(ax, node_coordinates, element_nodes, linestyle="k--")

    colors = plt.cm.tab20(np.linspace(0, 1, n_zones))

    for k, name in enumerate(zone_names):
        nodes = subdomains[name]
        center = node_coordinates[nodes].mean(axis=0)
        vec = v_red[k] * scale

        ax.quiver(center[0], center[1], center[2],
                  vec[0], vec[1], vec[2],
                  length=1.0, normalize=False,
                  color=colors[k], linewidth=2, arrow_length_ratio=0.3)

        ax.scatter(node_coordinates[nodes, 0],
                   node_coordinates[nodes, 1],
                   node_coordinates[nodes, 2],
                   s=40, color=colors[k], zorder=5)

    _set_equal_axes(ax, node_coordinates)
    ax.set_title(title)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.view_init(elev=-45, azim=170)
    ax.grid(True)
