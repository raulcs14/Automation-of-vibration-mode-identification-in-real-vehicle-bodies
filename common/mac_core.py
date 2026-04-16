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
    raise NotImplementedError
