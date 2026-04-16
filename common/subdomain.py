"""
Subdomain averaging and reduction utilities.
Equivalent to averageZones.m, averageModeSubdomain.m, reduce_MK_by_subdomains.m
"""

import numpy as np
from typing import Dict, List, Tuple


def average_zones(modes: np.ndarray, subdomains: Dict[str, List[int]],
                  n_dof_per_node: int = 6) -> np.ndarray:
    """
    Reduce mode shapes to one 3-DOF vector per subdomain (translational average).

    Args:
        modes: Full mode matrix (GDof, nModes)
        subdomains: zone name → list of node indices (0-based)
        n_dof_per_node: DOFs per node in the full model

    Returns:
        modes_red: (3·nZones, nModes) reduced modes
    """
    raise NotImplementedError


def average_mode_subdomain(mode: np.ndarray, node_indices: List[int],
                            n_dof_per_node: int = 6) -> np.ndarray:
    """
    Average translational DOFs (Ux, Uy, Uz) over all nodes in a zone for one mode.

    Returns:
        phi_avg: (3,) averaged vector
    """
    raise NotImplementedError


def reduce_mk_by_subdomains(M: np.ndarray, K: np.ndarray,
                             subdomains: Dict[str, List[int]],
                             n_nodes: int,
                             n_dof_per_node: int = 6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build reduced mass and stiffness matrices via subdomain Galerkin reduction.

    Constructs transformation T (GDof × nDOF_red) where T(i,j) = 1/ns
    if node i belongs to zone j, then Mr = Tᵀ M T, Kr = Tᵀ K T.

    Returns:
        Mr: Reduced mass matrix
        Kr: Reduced stiffness matrix
        T:  Transformation matrix
    """
    raise NotImplementedError
