"""
Generic MAC computation, model-agnostic.
Equivalent to MAC_calculator.m
"""

import numpy as np
from typing import Optional


def compute_mac(Phi: np.ndarray, Psi: np.ndarray,
                W: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute the weighted Modal Assurance Criterion matrix.

    MAC(i,j) = |φᵢᵀ W ψⱼ|² / (φᵢᵀ W φᵢ · ψⱼᵀ W ψⱼ)

    Args:
        Phi: Dynamic modes (nDOF, nModes)
        Psi: Reference shapes (nDOF, nRefs)
        W:   Weighting matrix (nDOF, nDOF). None → identity.

    Returns:
        mac: MAC matrix (nModes, nRefs), values in [0, 1]
    """
    if W is None:
        W = np.eye(Phi.shape[0])

    n_modes = Phi.shape[1]
    n_refs  = Psi.shape[1]
    mac = np.zeros((n_modes, n_refs))

    WPhi = W @ Phi   # (nDOF, nModes)
    WPsi = W @ Psi   # (nDOF, nRefs)

    phi_norms = np.einsum("ij,ij->j", Phi, WPhi)   # φᵢᵀ W φᵢ
    psi_norms = np.einsum("ij,ij->j", Psi, WPsi)   # ψⱼᵀ W ψⱼ

    for i in range(n_modes):
        for j in range(n_refs):
            num   = (Phi[:, i] @ WPsi[:, j]) ** 2
            denom = phi_norms[i] * psi_norms[j]
            mac[i, j] = num / denom if denom > 0 else 0.0

    return mac
