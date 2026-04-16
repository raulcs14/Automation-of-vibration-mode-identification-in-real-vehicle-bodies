"""
FE solvers: generalized eigenvalue problem and inertia relief static solver.
Equivalent to eigenvalue.m and inertia_relief_FE.m
"""

import numpy as np
from typing import Optional, Tuple


def solve_eigenvalue(K: np.ndarray, M: np.ndarray,
                     prescribed_dofs: np.ndarray,
                     n_modes: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the generalized eigenvalue problem: K φ = λ M φ

    Args:
        K: Global stiffness matrix (GDof x GDof)
        M: Global mass matrix (GDof x GDof)
        prescribed_dofs: DOF indices to remove (0-based)
        n_modes: Number of modes to extract. None → all modes (dense solver).

    Returns:
        modes: (GDof, n_modes) eigenvectors, zeros at prescribed DOFs
        eigenvalues: (n_modes,) eigenvalues λ = ω²
    """
    raise NotImplementedError


def inertia_relief(K: np.ndarray, M: np.ndarray,
                   f: np.ndarray, r_set: np.ndarray) -> np.ndarray:
    """
    Solve a static FE problem under free-free BCs using inertia relief.

    Partitions DOFs into r_set (reference) and l_set (free), computes
    constraint modes, and balances applied loads with inertial forces.

    Args:
        K: Global stiffness matrix
        M: Global mass matrix
        f: Applied force vector (GDof,)
        r_set: Reference DOF indices (0-based)

    Returns:
        u: Full displacement vector (GDof,)
    """
    raise NotImplementedError
