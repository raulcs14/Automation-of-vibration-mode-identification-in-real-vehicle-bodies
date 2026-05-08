"""
BIW subdomain definitions for subdomain-averaged MAC analysis.

Subdomains are built geometrically from node coordinates (mm), mirroring the
12-zone layout used for the simple model (chassis.py) but scaled to the full
BIW FEM mesh.

Coordinate conventions (mm):
  X — longitudinal, front (negative) → rear (positive)
  Y — lateral, left (positive) → right (negative), symmetric about 0
  Z — vertical, floor (low) → roof (high)

Zone boundaries (mm) are chosen to align with structural cross-sections:
  Longitudinal  front  / front_cabin / mid_cabin / rear_cabin / rear
  Vertical      floor  / mid         / roof
  Lateral       L (y ≥ 0)            / R (y < 0)

This gives up to 30 zones (5 × 3 × 2), but only those with at least one
A-set node are returned — empty zones are silently dropped.
"""

from typing import Dict, List
import numpy as np

# ---------------------------------------------------------------------------
# Geometry cuts (mm) — tuned for the dummycar BIW mesh
# ---------------------------------------------------------------------------

_X_CUTS = [500.0, 1200.0, 2200.0, 3000.0]   # 5 longitudinal bands
_X_LABELS = ["front", "front_cabin", "mid_cabin", "rear_cabin", "rear"]

_Z_CUTS = [200.0, 600.0]                      # 3 vertical bands
_Z_LABELS = ["floor", "mid", "roof"]


def build_biw_subdomains(
    node_ids: np.ndarray,
    node_xyz: np.ndarray,
) -> Dict[str, List[int]]:
    """
    Partition A-set nodes into geometric subdomains.

    Parameters
    ----------
    node_ids : (nNodes,) int
        Nastran GRID IDs in the current DOF order (A-set, sorted ascending).
    node_xyz : (nNodes, 3) float
        Node coordinates [mm] in the same order.

    Returns
    -------
    subdomains : Dict[str, List[int]]
        zone_name -> list of **positional indices** (0-based) into node_ids /
        node_xyz.  These indices are what average_zones and
        reduce_mk_by_subdomains expect.
        Only zones with at least one node are included.
    """
    x, y, z = node_xyz[:, 0], node_xyz[:, 1], node_xyz[:, 2]

    x_bounds = [-np.inf] + _X_CUTS + [np.inf]
    z_bounds = [-np.inf] + _Z_CUTS + [np.inf]

    subdomains: Dict[str, List[int]] = {}

    for xi, x_lbl in enumerate(_X_LABELS):
        x_lo, x_hi = x_bounds[xi], x_bounds[xi + 1]
        x_mask = (x >= x_lo) & (x < x_hi)

        for zi, z_lbl in enumerate(_Z_LABELS):
            z_lo, z_hi = z_bounds[zi], z_bounds[zi + 1]
            z_mask = (z >= z_lo) & (z < z_hi)

            for side, y_mask in [("L", y >= 0.0), ("R", y < 0.0)]:
                combined = np.where(x_mask & z_mask & y_mask)[0]
                if combined.size > 0:
                    name = f"{x_lbl}_{z_lbl}_{side}"
                    subdomains[name] = combined.tolist()

    return subdomains


def grid_ids_to_node_indices(
    subdomains_grid: Dict[str, List[int]],
    node_ids: np.ndarray,
) -> Dict[str, List[int]]:
    """
    Convert a subdomain dict keyed by Nastran GRID IDs to one keyed by
    positional indices into node_ids.

    Use this when subdomains were defined externally by GRID ID (e.g. from
    a Nastran SET card) rather than by geometry.

    Parameters
    ----------
    subdomains_grid : Dict[str, List[int]]
        zone_name -> list of Nastran GRID IDs
    node_ids : (nNodes,) int
        The ordered node ID array from DofSpace / run_modal_analysis.

    Returns
    -------
    Dict[str, List[int]]
        zone_name -> list of 0-based positional indices.
        GRID IDs not present in node_ids (e.g. excluded from A-set) are
        silently dropped; zones that become empty are omitted.
    """
    id_to_pos = {int(nid): pos for pos, nid in enumerate(node_ids)}
    result: Dict[str, List[int]] = {}
    for name, gids in subdomains_grid.items():
        indices = [id_to_pos[g] for g in gids if g in id_to_pos]
        if indices:
            result[name] = indices
    return result
