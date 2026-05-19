"""
Subdomain definitions for subdomain-averaged MAC analysis (BIW and TB).

Two strategies are available:

1. Shell-based (preferred): subdomains are derived from the PID->GRID mapping
   exported by the META script ``export_biw_subdomains.py``.  Each PSHELL
   property becomes one zone ("pid_<N>").

2. Geometry-based (BIW fallback): nodes are partitioned by coordinate cuts
   into up to 30 zones (5 longitudinal × 3 vertical × 2 lateral).  Used when
   no JSON file is available.

Coordinate conventions (mm):
  X — longitudinal, front (negative) → rear (positive)
  Y — lateral, left (positive) → right (negative), symmetric about 0
  Z — vertical, floor (low) → roof (high)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

_REPO_ROOT   = Path(__file__).resolve().parents[1]
_DATA_ROOT   = _REPO_ROOT / "data" / "seat_model"
_JSON_NAME   = "subdomains.json"

# ---------------------------------------------------------------------------
# Geometry cuts (mm) — BIW fallback only
# ---------------------------------------------------------------------------

_X_CUTS   = [500.0, 1200.0, 2200.0, 3000.0]
_X_LABELS = ["front", "front_cabin", "mid_cabin", "rear_cabin", "rear"]
_Z_CUTS   = [200.0, 600.0]
_Z_LABELS = ["floor", "mid", "roof"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_subdomains(
    variant: str,
    node_ids: np.ndarray,
    node_xyz: np.ndarray,
    json_path: Optional[Path] = None,
) -> Dict[str, List[int]]:
    """
    Build subdomains as positional indices into node_ids for BIW or TB.

    Looks for ``data/ansa_model/<variant>/subdomains.json`` (produced by
    ``export_biw_subdomains.py``).  For BIW falls back to geometric zones
    if the file is missing; TB raises an error instead since there is no
    meaningful geometric fallback for a trimmed body.

    Parameters
    ----------
    variant  : "BIW" or "TB"
    node_ids : (nNodes,) int   — Nastran GRID IDs in A-set order
    node_xyz : (nNodes, 3) float — node coordinates [mm]
    json_path : override path to subdomains.json

    Returns
    -------
    Dict[str, List[int]]
        zone_name -> list of 0-based positional indices into node_ids.
        Empty zones are omitted.
    """
    resolved = Path(json_path) if json_path is not None else _DATA_ROOT / variant / "meta" / _JSON_NAME
    if resolved.exists():
        print(f"  [subdomains] loading PID zones from {resolved}")
        return _build_from_json(resolved, node_ids)

    if variant == "BIW":
        print("  [subdomains] JSON not found, falling back to geometric zones")
        return _build_geometric(node_xyz)

    raise FileNotFoundError(
        f"Subdomain JSON not found: {resolved}\n"
        f"Run export_biw_subdomains.py in META with VARIANT='{variant}'."
    )


def build_biw_subdomains(
    node_ids: np.ndarray,
    node_xyz: np.ndarray,
    json_path: Optional[Path] = None,
) -> Dict[str, List[int]]:
    """Alias for build_subdomains('BIW', ...) — kept for backwards compatibility."""
    return build_subdomains("BIW", node_ids, node_xyz, json_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_from_json(json_path: Path, node_ids: np.ndarray) -> Dict[str, List[int]]:
    with open(json_path, encoding="utf-8") as f:
        pid_grids: Dict[str, List[int]] = json.load(f)
    return grid_ids_to_node_indices(pid_grids, node_ids)


def _build_geometric(node_xyz: np.ndarray) -> Dict[str, List[int]]:
    """Partition nodes by coordinate cuts into up to 30 zones."""
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
                idx = np.where(x_mask & z_mask & y_mask)[0]
                if idx.size > 0:
                    subdomains[f"{x_lbl}_{z_lbl}_{side}"] = idx.tolist()
    return subdomains


def grid_ids_to_node_indices(
    subdomains_grid: Dict[str, List[int]],
    node_ids: np.ndarray,
) -> Dict[str, List[int]]:
    """
    Convert a subdomain dict keyed by Nastran GRID IDs to one keyed by
    positional indices into node_ids.

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
