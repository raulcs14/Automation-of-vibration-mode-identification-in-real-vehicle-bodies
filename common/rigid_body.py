"""
Rigid-body utilities: basis construction and component removal from mode shapes.
Equivalent to removeRigidBodyComponent.m
"""

import numpy as np


def build_rigid_body_basis(node_xyz: np.ndarray) -> np.ndarray:
    """
    Build the rigid-body mode basis R (6*N x 6).

    Columns: Tx, Ty, Tz, Rx (rot about X), Ry (rot about Y), Rz (rot about Z).
    DOF order per node: [Ux, Uy, Uz, Rx, Ry, Rz].
    """
    n_nodes = node_xyz.shape[0]
    R = np.zeros((6 * n_nodes, 6))
    for i, (x, y, z) in enumerate(node_xyz):
        R[6*i + 0, 0] = 1.0;  R[6*i + 1, 1] = 1.0;  R[6*i + 2, 2] = 1.0
        R[6*i + 1, 3] = -z;   R[6*i + 2, 3] =  y
        R[6*i + 0, 4] =  z;   R[6*i + 2, 4] = -x
        R[6*i + 0, 5] = -y;   R[6*i + 1, 5] =  x
    return R


def remove_rigid_body_component(modes: np.ndarray, M: np.ndarray,
                                 R: np.ndarray) -> np.ndarray:
    """
    Project out rigid-body contributions using a mass-weighted projector.

    P = I - M R (Rᵀ M R)⁻¹ Rᵀ
    modes_proj = P · modes

    Args:
        modes: Mode matrix (GDof, nModes)
        M: Global mass matrix (GDof, GDof)
        R: Rigid-body basis (GDof, 6)

    Returns:
        modes_proj: Projected modes with rigid-body component removed
    """
    MR    = M @ R                                    # (GDof, 6)
    G     = R.T @ MR                                 # (6, 6)  Gram matrix
    alpha = np.linalg.solve(G, R.T @ (M @ modes))   # (6, nVectors)
    return modes - R @ alpha
