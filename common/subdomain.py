"""
Subdomain averaging and reduction utilities.
Equivalent to averageZones.m, averageModeSubdomain.m, reduce_MK_by_subdomains.m
"""

import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Tuple


def average_zones(modes_t: np.ndarray, subdomains: Dict[str, List[int]],
                  n_nodes: int) -> np.ndarray:
    """
    Reduce translational mode matrix to one 3-DOF vector per subdomain.

    Args:
        modes_t: Translational mode matrix (3·n_nodes, nModes), layout
                 [Ux_0..Ux_N | Uy_0..Uy_N | Uz_0..Uz_N] — block layout, as
                 produced by translational_dof_indices(), i.e.
                 np.concatenate([arange(0,gdof,6), arange(1,gdof,6), arange(2,gdof,6)]).
        subdomains: zone name → list of node indices (0-based)
        n_nodes: total number of nodes in the model

    Returns:
        modes_red: (3·nZones, nModes) averaged vectors
    """
    zone_names = list(subdomains.keys())
    n_zones = len(zone_names)
    n_modes = modes_t.shape[1] if modes_t.ndim == 2 else 1
    if modes_t.ndim == 1:
        modes_t = modes_t[:, None]

    modes_red = np.zeros((3 * n_zones, n_modes))
    for k, name in enumerate(zone_names):
        nodes = np.asarray(subdomains[name])
        # In modes_t: Ux block = [0..n_nodes), Uy = [n_nodes..2*n_nodes), Uz = [2*n_nodes..)
        ux = modes_t[nodes,            :]   # (ns, nModes)
        uy = modes_t[nodes + n_nodes,  :]
        uz = modes_t[nodes + 2*n_nodes,:]
        modes_red[3*k,   :] = ux.mean(axis=0)
        modes_red[3*k+1, :] = uy.mean(axis=0)
        modes_red[3*k+2, :] = uz.mean(axis=0)

    return modes_red


def reduce_mk_by_subdomains(M_t: np.ndarray, K_t: np.ndarray,
                             subdomains: Dict[str, List[int]],
                             n_nodes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build reduced mass and stiffness matrices via subdomain Galerkin reduction.

    Operates on the translational-only matrices M_t, K_t (3·n_nodes × 3·n_nodes)
    with layout [Ux_0..Ux_N | Uy_0..Uy_N | Uz_0..Uz_N], matching average_zones.

    Returns:
        Mr: (3·nZones, 3·nZones) reduced mass matrix
        Kr: (3·nZones, 3·nZones) reduced stiffness matrix
        T:  (3·n_nodes, 3·nZones) transformation matrix
    """
    zone_names = list(subdomains.keys())
    n_zones   = len(zone_names)
    n_dof_full = 3 * n_nodes
    n_dof_red  = 3 * n_zones

    # Build T as sparse CSC — avoids densifying large M_t/K_t for BIW-scale models
    rows, cols, vals = [], [], []
    for s, name in enumerate(zone_names):
        nodes_s = np.asarray(subdomains[name])
        ns = len(nodes_s)
        for d in range(3):   # Ux, Uy, Uz blocks
            red_dof   = s * 3 + d
            full_dofs = nodes_s + d * n_nodes   # block layout
            rows.extend(full_dofs.tolist())
            cols.extend([red_dof] * len(nodes_s))
            vals.extend([1.0 / ns] * len(nodes_s))

    T_sp = sp.csc_matrix((vals, (rows, cols)), shape=(n_dof_full, n_dof_red))

    if sp.issparse(M_t):
        Mr = (T_sp.T @ M_t @ T_sp).toarray()
        Kr = (T_sp.T @ K_t @ T_sp).toarray()
    else:
        T = T_sp.toarray()
        Mr = T.T @ M_t @ T
        Kr = T.T @ K_t @ T

    return Mr, Kr, T_sp.toarray()
