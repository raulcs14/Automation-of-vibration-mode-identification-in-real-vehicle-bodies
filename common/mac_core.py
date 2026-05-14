"""
Generic MAC computation, model-agnostic.
Equivalent to MAC_calculator.m
"""

import numpy as np
import scipy.sparse as sp
from typing import Optional


def compute_mac(Phi: np.ndarray, Psi: np.ndarray,
                W=None) -> np.ndarray:
    """
    Compute the weighted Modal Assurance Criterion matrix.

    MAC(i,j) = |φᵢᵀ W ψⱼ|² / (φᵢᵀ W φᵢ · ψⱼᵀ W ψⱼ)

    Args:
        Phi: Dynamic modes (nDOF, nModes)
        Psi: Reference shapes (nDOF, nRefs)
        W:   Weighting matrix (nDOF, nDOF), dense or sparse. None → identity.

    Returns:
        mac: MAC matrix (nModes, nRefs), values in [0, 1]
    """
    n_modes = Phi.shape[1]
    n_refs  = Psi.shape[1]
    mac = np.zeros((n_modes, n_refs))

    if W is None:
        WPhi = Phi
        WPsi = Psi
    else:
        WPhi = np.asarray(W @ Phi)   # (nDOF, nModes) — works for sparse or dense W
        WPsi = np.asarray(W @ Psi)   # (nDOF, nRefs)

    phi_norms = np.einsum("ij,ij->j", Phi, WPhi)   # φᵢᵀ W φᵢ
    psi_norms = np.einsum("ij,ij->j", Psi, WPsi)   # ψⱼᵀ W ψⱼ

    cross = WPsi.T @ Phi   # (nRefs, nModes)

    denom = np.outer(psi_norms, phi_norms)   # (nRefs, nModes)
    with np.errstate(invalid="ignore", divide="ignore"):
        mac_T = np.where(denom > 0, cross ** 2 / denom, 0.0)

    return mac_T.T   # (nModes, nRefs)


def best_mac_per_mode(mac: np.ndarray) -> np.ndarray:
    """Return (nModes,) array of max MAC value over all references."""
    return mac.max(axis=1)
