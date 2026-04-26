import numpy as np


def save_csv(matrix, path):
    np.savetxt(path, matrix, delimiter=',')
    print(f"Saved {path}  {matrix.shape}")


def interleave_6dof(trans: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """
    Interleave translational and rotational results into a (6*nNodes, nCols) matrix.

    trans, rot: (nNodes*3, nCols) — translational and rotational components.
    Returns   : (nNodes*6, nCols) — rows ordered as [Ux, Uy, Uz, Rx, Ry, Rz] per node.
    """
    DOF, n_cols = trans.shape
    n_nodes = DOF // 3
    T = trans.reshape(n_nodes, 3, n_cols)
    R = rot.reshape(n_nodes, 3, n_cols)
    return np.concatenate([T, R], axis=1).reshape(6 * n_nodes, n_cols)


def extract_deformation_matrix(item_list, grids_group) -> np.ndarray:
    """
    Extract deformation vectors for each item in item_list and stack into a matrix.

    Works for both mode lists (eigenvectors) and load case lists (displacements).
    Returns: (nDOF, nItems) if more than one item, else (nDOF, 1).
    """
    vectors = []
    for item in item_list:
        vec = grids_group.get_deformations(item, 'all', numpy='xyz')
        vec = np.array(vec).reshape(-1)
        vectors.append(vec)
    return np.column_stack(vectors) if len(vectors) > 1 else vectors[0].reshape(-1, 1)
