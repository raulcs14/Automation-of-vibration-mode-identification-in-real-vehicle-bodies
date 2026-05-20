"""
FE solvers: inertia relief static solver.
Equivalent to inertia_relief_FE.m
"""

import numpy as np


def inertia_relief(K: np.ndarray, M: np.ndarray,
                   f: np.ndarray, r_set: np.ndarray) -> np.ndarray:
    """
    Solve a static FE problem under free-free BCs using inertia relief.

    Partitions DOFs into r_set (reference) and l_set (free), computes
    constraint modes Psi, and balances applied loads with inertial forces.

    Args:
        K: Global stiffness matrix
        M: Global mass matrix
        f: Applied force vector (GDof,)
        r_set: Reference DOF indices (0-based)

    Returns:
        u: Full displacement vector (GDof,)
    """
    GDof = K.shape[0]
    l_set = np.setdiff1d(np.arange(GDof), r_set)
    n_r = len(r_set)

    K_ll = K[np.ix_(l_set, l_set)]
    K_lr = K[np.ix_(l_set, r_set)]
    M_ll = M[np.ix_(l_set, l_set)]
    M_lr = M[np.ix_(l_set, r_set)]
    f_l  = f[l_set]

    # Step 1: constraint modes Psi (static condensation)
    K_product = -np.linalg.solve(K_ll, K_lr)          # (n_l, n_r)
    Psi = np.zeros((GDof, n_r))
    Psi[r_set, :] = np.eye(n_r)
    Psi[l_set, :] = K_product

    # Step 2: rigid-body accelerations that balance applied loads
    M_phi     = Psi.T @ M @ Psi                        # (n_r, n_r)
    alpha_vec = -np.linalg.solve(M_phi, Psi.T @ f)    # (n_r,)

    # Step 3: solve reduced static problem with inertia-relief correction
    M_low  = np.hstack([M_lr, M_ll])                   # (n_l, n_r + n_l)
    f_ir   = M_low @ (Psi @ alpha_vec)
    u_l    = np.linalg.solve(K_ll, f_l + f_ir)

    u = np.zeros(GDof)
    u[l_set] = u_l
    return u
