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
    nc = geometry.node_coordinates
    en = geometry.element_nodes
    GDof = 6 * nc.shape[0]
    K = np.zeros((GDof, GDof))

    for i_node, j_node in en:
        x1, y1, z1 = nc[i_node]
        x2, y2, z2 = nc[j_node]
        L = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

        k1  = E*A/L
        k2  = 12*E*Iz / L**3;  k3 = 6*E*Iz / L**2
        k4  = 4*E*Iz  / L;     k5 = 2*E*Iz / L
        k6  = 12*E*Iy / L**3;  k7 = 6*E*Iy / L**2
        k8  = 4*E*Iy  / L;     k9 = 2*E*Iy / L
        k10 = G*J / L

        k_loc = np.array([
            [ k1,   0,   0,    0,    0,   0,  -k1,   0,   0,    0,    0,   0],
            [  0,  k2,   0,    0,    0,  k3,    0, -k2,   0,    0,    0,  k3],
            [  0,   0,  k6,    0,  -k7,   0,    0,   0, -k6,    0,  -k7,   0],
            [  0,   0,   0,  k10,    0,   0,    0,   0,   0, -k10,    0,   0],
            [  0,   0, -k7,    0,   k8,   0,    0,   0,  k7,    0,   k9,   0],
            [  0,  k3,   0,    0,    0,  k4,    0, -k3,   0,    0,    0,  k5],
            [-k1,   0,   0,    0,    0,   0,   k1,   0,   0,    0,    0,   0],
            [  0, -k2,   0,    0,    0, -k3,    0,  k2,   0,    0,    0, -k3],
            [  0,   0, -k6,    0,   k7,   0,    0,   0,  k6,    0,   k7,   0],
            [  0,   0,   0, -k10,    0,   0,    0,   0,   0,  k10,    0,   0],
            [  0,   0, -k7,    0,   k9,   0,    0,   0,  k7,    0,   k8,   0],
            [  0,  k3,   0,    0,    0,  k5,    0, -k3,   0,    0,    0,  k4],
        ])

        R = _rotation_matrix(x1, y1, z1, x2, y2, z2, L)
        dofs = _element_dofs(i_node, j_node)
        K[np.ix_(dofs, dofs)] += R.T @ k_loc @ R

    return K


def _rotation_matrix(x1, y1, z1, x2, y2, z2, L) -> np.ndarray:
    if x1 == x2 and y1 == y2:
        lam = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float) if z2 > z1 \
              else np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], float)
    else:
        CXx = (x2-x1)/L;  CYx = (y2-y1)/L;  CZx = (z2-z1)/L
        D   = np.sqrt(CXx**2 + CYx**2)
        lam = np.array([
            [CXx,        CYx,        CZx],
            [-CYx/D,     CXx/D,      0.0],
            [-CXx*CZx/D, -CYx*CZx/D, D  ],
        ])
    z3 = np.zeros((3, 3))
    return np.block([
        [lam, z3,  z3,  z3],
        [z3,  lam, z3,  z3],
        [z3,  z3,  lam, z3],
        [z3,  z3,  z3,  lam],
    ])


def _element_dofs(i: int, j: int) -> np.ndarray:
    return np.array([6*i, 6*i+1, 6*i+2, 6*i+3, 6*i+4, 6*i+5,
                     6*j, 6*j+1, 6*j+2, 6*j+3, 6*j+4, 6*j+5])
