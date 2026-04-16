"""
Chassis geometry: node coordinates, element connectivity, and subdomain definitions.
Equivalent to buildChassisGeometry.m
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, List, Literal


@dataclass
class ChassisGeometry:
    node_coordinates: np.ndarray        # (nNodes, 3) — x, y, z
    element_nodes: np.ndarray           # (nElements, 2) — node indices (0-based)
    subdomains: Dict[str, List[int]]    # zone name → node indices (0-based)
    inertia_node: int                   # 0-based index of the inertia-relief node


def build_chassis_geometry(
    movement: Literal["torsion", "bending"] = "torsion",
) -> ChassisGeometry:
    """
    Define the 3D car chassis mesh.

    35 nodes in 5 cross-sections along X:
      bumper (x=-1), front (x=0), mid (x=2), rear (x=4), tail (x=5.5)
    plus A-pillar, roof-arc, C-pillar nodes and one inertia-relief node.

    Node indices are 0-based (MATLAB indices minus 1).

    Returns a ChassisGeometry with nodes, elements, and subdomain definitions.
    """
    # ------------------------------------------------------------------
    # Node coordinates  (MATLAB 1-based → Python 0-based in connectivity)
    # ------------------------------------------------------------------
    nodes = np.array([
        # bumper tip  (x = -1.0)
        [-1.0,  0.7,  0.0],   # 0  (MATLAB 1)
        [-1.0, -0.7,  0.0],   # 1
        [-1.0,  0.6,  0.8],   # 2
        [-1.0, -0.6,  0.8],   # 3
        # front section  (x = 0.0)
        [ 0.0,  1.2,  0.0],   # 4  (MATLAB 5)
        [ 0.0, -1.2,  0.0],   # 5
        [ 0.0,  1.1,  1.2],   # 6
        [ 0.0, -1.1,  1.2],   # 7
        [ 0.0,  0.9,  1.6],   # 8
        [ 0.0, -0.9,  1.6],   # 9
        # cabin mid  (x = 2.0)
        [ 2.0,  1.4,  0.0],   # 10 (MATLAB 11)
        [ 2.0, -1.4,  0.0],   # 11
        [ 2.0,  1.3,  1.2],   # 12
        [ 2.0, -1.3,  1.2],   # 13
        [ 2.0,  1.0,  2.4],   # 14
        [ 2.0, -1.0,  2.4],   # 15
        # rear section  (x = 4.0)
        [ 4.0,  1.2,  0.0],   # 16 (MATLAB 17)
        [ 4.0, -1.2,  0.0],   # 17
        [ 4.0,  1.1,  1.2],   # 18
        [ 4.0, -1.1,  1.2],   # 19
        [ 4.0,  0.9,  2.2],   # 20
        [ 4.0, -0.9,  2.2],   # 21
        # tail tip  (x = 5.5)
        [ 5.5,  0.8,  0.0],   # 22 (MATLAB 23)
        [ 5.5, -0.8,  0.0],   # 23
        [ 5.5,  0.7,  1.0],   # 24
        [ 5.5, -0.7,  1.0],   # 25
        [ 5.5,  0.6,  1.4],   # 26
        [ 5.5, -0.6,  1.4],   # 27
        # A-pillar tops  (x ≈ 1.0)
        [ 1.0,  0.95, 2.1],   # 28 (MATLAB 29)
        [ 1.0, -0.95, 2.1],   # 29
        # roof mid curvature  (x ≈ 3.0)
        [ 3.0,  0.95, 2.35],  # 30 (MATLAB 31)
        [ 3.0, -0.95, 2.35],  # 31
        # C-pillar / fastback mid  (x ≈ 4.8)
        [ 4.8,  0.75, 1.85],  # 32 (MATLAB 33)
        [ 4.8, -0.75, 1.85],  # 33
    ], dtype=float)

    # node 34 (MATLAB 35): inertia-relief node, position depends on movement
    if movement == "torsion":
        inertia_coords = [2.0, 0.0, 0.6]
    elif movement == "bending":
        inertia_coords = [1.0, 0.0, 0.6]
    else:
        raise ValueError(f"Unknown movement variant: {movement!r}")

    nodes = np.vstack([nodes, inertia_coords])   # node index 34
    inertia_idx = 34

    # ------------------------------------------------------------------
    # Element connectivity  (0-based)
    # ------------------------------------------------------------------
    def E(*pairs):
        return np.array(pairs, dtype=int) - 1   # convert from 1-based MATLAB

    elems = np.vstack([
        # nose / bumper triangulation
        E([1,5],[2,6],[3,7],[4,8],[1,3],[2,4],[3,4],[1,2]),
        # hood closure
        E([3,9],[4,10]),
        # lower rails
        E([5,11],[11,17],[17,23],[6,12],[12,18],[18,24]),
        # belt rails
        E([7,13],[13,19],[19,25],[8,14],[14,20],[20,26]),
        # roof rails (with curvature nodes)
        E([9,29],[29,15],[15,31],[31,21],[21,33],[33,27]),
        E([10,30],[30,16],[16,32],[32,22],[22,34],[34,28]),
        # cross-members
        E([5,6],[7,8],[9,10],[11,12],[13,14],[15,16],
          [17,18],[19,20],[21,22],[23,24],[25,26],[27,28]),
        # pillars (A, B, C)
        E([5,7],[7,9],[6,8],[8,10],
          [11,13],[13,15],[12,14],[14,16],
          [17,19],[19,21],[18,20],[20,22]),
        # fastback tail
        E([25,27],[26,28]),
        # roof X-braces
        E([29,16],[30,15],[15,22],[16,21]),
        # tail closure
        E([23,25],[24,26],[33,34]),
    ])

    # inertia-relief connections (1-based: node 35 = index 34)
    if movement == "torsion":
        ir = np.array([[10,34],[11,34],[12,34],[13,34]], dtype=int)
    else:  # bending
        ir = np.array([[10,34],[11,34],[12,34],[13,34],
                       [4,34],[5,34],[6,34],[7,34]], dtype=int)

    element_nodes = np.vstack([elems, ir])

    # ------------------------------------------------------------------
    # Subdomains  (0-based node indices)
    # ------------------------------------------------------------------
    subdomains: Dict[str, List[int]] = {
        "front_floor_L": [0, 4],
        "front_floor_R": [1, 5],
        "front_roof_L":  [2, 6, 8, 28],
        "front_roof_R":  [3, 7, 9, 29],
        "mid_floor_L":   [10, 12, 14],
        "mid_floor_R":   [11, 13, 15],
        "rear_floor_L":  [16, 22],
        "rear_floor_R":  [17, 23],
        "rear_mid_L":    [18, 24],
        "rear_mid_R":    [19, 25],
        "rear_roof_L":   [20, 26, 30, 32],
        "rear_roof_R":   [21, 27, 31, 33],
        "inertia":       [inertia_idx],
    }

    _check_all_nodes_assigned(nodes.shape[0], subdomains)

    return ChassisGeometry(
        node_coordinates=nodes,
        element_nodes=element_nodes,
        subdomains=subdomains,
        inertia_node=inertia_idx,
    )


def _check_all_nodes_assigned(n_nodes: int, subdomains: Dict[str, List[int]]) -> None:
    belongs = np.zeros(n_nodes, dtype=int)
    for indices in subdomains.values():
        for idx in indices:
            belongs[idx] += 1
    free = np.where(belongs == 0)[0]
    if free.size == 0:
        print("OK - All nodes belong to at least one subdomain.")
    else:
        print(f"WARNING - Nodes with no subdomain: {free.tolist()}")


# ------------------------------------------------------------------
# Visualisation helpers
# ------------------------------------------------------------------

def plot_chassis(geometry: ChassisGeometry, view: tuple = (135, 20)) -> plt.Figure:
    """Plain mesh plot, no node labels."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    _draw_mesh_lines(ax, geometry.node_coordinates, geometry.element_nodes)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    _set_equal_axes(ax, geometry.node_coordinates)
    ax.grid(True)
    ax.view_init(*view)
    return fig


def plot_chassis_numbered(geometry: ChassisGeometry, view: tuple = (200, 20)) -> plt.Figure:
    """Mesh plot with 1-based node numbers (matching MATLAB output)."""
    nc = geometry.node_coordinates
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    _draw_mesh_lines(ax, nc, geometry.element_nodes, linestyle="k--", marker=".")
    for i, (x, y, z) in enumerate(nc):
        ax.text(x, y, z, str(i + 1),          # 1-based label
                fontsize=9, color="red", fontweight="bold")
    ax.set_title("Node numbering")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    _set_equal_axes(ax, nc)
    ax.grid(True)
    ax.view_init(*view)
    return fig


def _draw_mesh_lines(ax, node_coordinates, element_nodes,
                     linestyle="k-", marker=None) -> None:
    nc = node_coordinates
    fmt = linestyle + (marker or "")
    for e in element_nodes:
        n1, n2 = e
        ax.plot([nc[n1, 0], nc[n2, 0]],
                [nc[n1, 1], nc[n2, 1]],
                [nc[n1, 2], nc[n2, 2]],
                fmt, linewidth=1.2)


def _set_equal_axes(ax, node_coordinates: np.ndarray) -> None:
    """Force equal scale on all three axes, Z=0 at bottom."""
    mins = node_coordinates.min(axis=0)
    maxs = node_coordinates.max(axis=0)
    center = (mins + maxs) / 2
    half = (maxs - mins).max() / 2
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    zlo, zhi = center[2] - half, center[2] + half
    ax.set_zlim(zlo, zhi)
    # Ensure Z increases upward (reset any prior inversion before applying ours)
    if ax.zaxis_inverted():
        ax.invert_zaxis()   # un-invert if already flipped
    # matplotlib 3D default has Z increasing upward — nothing more needed
