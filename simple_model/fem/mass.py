"""
Global consistent mass matrix assembly for 3D frame elements.
Equivalent to formMass3Dframe.m
"""

import numpy as np
from simple_model.geometry.chassis import ChassisGeometry
from simple_model.fem.stiffness import _rotation_matrix, _element_dofs


def form_mass(geometry: ChassisGeometry, rho: float, A: float,
              Iy: float, Iz: float) -> np.ndarray:
    """
    Assemble the global consistent mass matrix M (GDof x GDof).

    Uses the consistent mass formulation with translational and rotational
    inertia terms. Same coordinate transformation as stiffness assembly.
    """
    nc = geometry.node_coordinates
    en = geometry.element_nodes
    GDof = 6 * nc.shape[0]
    M = np.zeros((GDof, GDof))

    for i_node, j_node in en:
        x1, y1, z1 = nc[i_node]
        x2, y2, z2 = nc[j_node]
        L = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

        p = (Iz + Iy) / A   # rotational inertia factor

        m_loc = rho*A*L/420 * np.array([
            [140,    0,      0,      0,         0,        0,      70,    0,      0,      0,         0,        0    ],
            [  0,  156,      0,      0,         0,    22*L,       0,    54,      0,      0,         0,   -13*L    ],
            [  0,    0,    156,      0,    -22*L,        0,       0,     0,     54,      0,      13*L,        0    ],
            [  0,    0,      0,  140*p,         0,        0,      0,     0,      0,   70*p,         0,        0    ],
            [  0,    0,  -22*L,      0,     4*L**2,       0,      0,     0,  -13*L,      0,  -3*L**2,        0    ],
            [  0,  22*L,     0,      0,         0,   4*L**2,      0,  13*L,     0,      0,         0,  -3*L**2   ],
            [ 70,    0,      0,      0,         0,        0,    140,    0,      0,      0,         0,        0    ],
            [  0,   54,      0,      0,         0,    13*L,       0,   156,     0,      0,         0,   -22*L    ],
            [  0,    0,     54,      0,    -13*L,        0,       0,     0,    156,     0,      22*L,        0    ],
            [  0,    0,      0,   70*p,         0,        0,      0,     0,      0,  140*p,        0,        0    ],
            [  0,    0,   13*L,      0,    -3*L**2,       0,      0,     0,   22*L,     0,   4*L**2,        0    ],
            [  0, -13*L,     0,      0,         0,  -3*L**2,      0, -22*L,     0,      0,         0,   4*L**2  ],
        ])

        R = _rotation_matrix(x1, y1, z1, x2, y2, z2, L)
        dofs = _element_dofs(i_node, j_node)
        M[np.ix_(dofs, dofs)] += R.T @ m_loc @ R

    return M
