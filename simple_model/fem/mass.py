"""
Global consistent mass matrix assembly for 3D frame elements.
Equivalent to formMass3Dframe.m
"""

import numpy as np
from simple_model.geometry.chassis import ChassisGeometry


def form_mass(geometry: ChassisGeometry, rho: float, A: float,
              Iy: float, Iz: float) -> np.ndarray:
    """
    Assemble the global consistent mass matrix M (GDof x GDof).

    Uses the consistent mass formulation with translational and rotational
    inertia terms. Same coordinate transformation as stiffness assembly.
    """
    raise NotImplementedError
