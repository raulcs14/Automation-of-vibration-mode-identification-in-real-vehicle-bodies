"""
Shear internal forces for 3D beam elements.

For each structural element (inertia-relief elements excluded) computes the
transverse shear forces Fy and Fz at both end-nodes from the element
displacement vector.  The result is a compact vector that can be used as
input to compute_mac() in place of nodal displacements.
"""

import numpy as np
from simple_model.geometry.chassis import ChassisGeometry


def build_shear_projection_matrix(
    geometry: ChassisGeometry,
    E: float, G: float,
    A: float, Iy: float, Iz: float, J: float,
) -> np.ndarray:
    """
    Build the explicit shear projection matrix B (4*n_struct_elems, nDOF).

    Each block of 4 rows corresponds to one structural element and contains
    the rows [1, 2, 7, 8] of (k_local @ R_elem) — i.e. the linear maps from
    global displacements to [Fy_i, Fz_i, Fy_j, Fz_j].

    Inertia-relief elements are excluded (same criterion as
    project_to_shear_forces).
    """
    nc           = geometry.node_coordinates
    en           = geometry.element_nodes
    inertia_node = geometry.inertia_node
    nDOF         = 6 * nc.shape[0]

    struct_mask  = ~((en[:, 0] == inertia_node) | (en[:, 1] == inertia_node))
    struct_elems = en[struct_mask]
    n_elems      = len(struct_elems)

    B = np.zeros((4 * n_elems, nDOF))

    for e_idx, (i_node, j_node) in enumerate(struct_elems):
        x1, y1, z1 = nc[i_node]
        x2, y2, z2 = nc[j_node]
        L = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

        kR   = _local_stiffness(E, G, A, Iy, Iz, J, L) @ _rotation_matrix(x1, y1, z1, x2, y2, z2, L)
        dofs = _element_dofs(i_node, j_node)

        row = 4 * e_idx
        B[row,     dofs] = kR[1, :]   # Fy_i
        B[row + 1, dofs] = kR[2, :]   # Fz_i
        B[row + 2, dofs] = kR[7, :]   # Fy_j
        B[row + 3, dofs] = kR[8, :]   # Fz_j

    return B


def project_mk_to_shear(
    B: np.ndarray,
    M: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project M and K into the shear-force space via Galerkin congruence.

    M_tau = B @ M @ Bᵀ      (4*n_struct_elems, 4*n_struct_elems)
    K_tau = B @ K @ Bᵀ      (4*n_struct_elems, 4*n_struct_elems)
    """
    BM    = B @ M
    BK    = B @ K
    M_tau = BM @ B.T
    K_tau = BK @ B.T
    return M_tau, K_tau


def project_to_shear_forces(
    shape_matrix: np.ndarray,
    geometry: ChassisGeometry,
    E: float, G: float,
    A: float, Iy: float, Iz: float, J: float,
) -> np.ndarray:
    """
    Project mode/reference shapes onto the shear-force space via B @ shape_matrix.

    Equivalent to computing [Fy_i, Fz_i, Fy_j, Fz_j] per structural element
    from f_local = k_local @ R @ u_global, stacked for all structural elements.

    Args:
        shape_matrix: (nDOF, nShapes) — mode or reference displacement vectors.
        geometry:     ChassisGeometry with node_coordinates, element_nodes,
                      and inertia_node.
        E, G, A, Iy, Iz, J: material and section properties.

    Returns:
        tau: (4 * n_structural_elements, nShapes) — shear force vectors.
    """
    B = build_shear_projection_matrix(geometry, E, G, A, Iy, Iz, J)
    return B @ shape_matrix


# ---------------------------------------------------------------------------
# Local helpers (mirror of stiffness.py, kept private here)
# ---------------------------------------------------------------------------

def _local_stiffness(E, G, A, Iy, Iz, J, L) -> np.ndarray:
    k1  = E*A/L
    k2  = 12*E*Iz / L**3;  k3 = 6*E*Iz / L**2
    k4  = 4*E*Iz  / L;     k5 = 2*E*Iz / L
    k6  = 12*E*Iy / L**3;  k7 = 6*E*Iy / L**2
    k8  = 4*E*Iy  / L;     k9 = 2*E*Iy / L
    k10 = G*J / L

    return np.array([
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
    ], dtype=float)


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
