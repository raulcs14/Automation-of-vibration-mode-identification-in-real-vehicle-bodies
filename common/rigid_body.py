"""
Rigid-body component removal from mode shapes.
Equivalent to removeRigidBodyComponent.m
"""

import numpy as np


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
    raise NotImplementedError
