import numpy as np


def translational_dof_indices(gdof: int) -> np.ndarray:
    """Return indices of translational DOFs (Ux, Uy, Uz) in a 6-DOF-per-node layout."""
    return np.concatenate([np.arange(d, gdof, 6) for d in range(3)])


def densify(mat):
    """Convert a sparse matrix to dense ndarray; pass-through if already dense."""
    return mat.toarray() if hasattr(mat, "toarray") else np.asarray(mat)
