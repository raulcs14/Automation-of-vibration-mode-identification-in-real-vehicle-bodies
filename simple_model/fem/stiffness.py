"""
Global stiffness matrix assembly for 3D frame elements.
Equivalent to formStiffness3Dframe.m
"""

import numpy as np
from simple_model.geometry.chassis import ChassisGeometry


def form_stiffness(geometry: ChassisGeometry, E: float, G: float,
                   A: float, Iy: float, Iz: float, J: float) -> np.ndarray:
    """
    Assemble the global stiffness matrix K (GDof x GDof).

    Each 3D beam element contributes a 12x12 local stiffness matrix
    (axial EA/L, bending EI/L³, torsion GJ/L) transformed to global frame.
    """
    raise NotImplementedError
