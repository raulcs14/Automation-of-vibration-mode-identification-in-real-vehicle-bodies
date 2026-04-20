"""
Export modal eigenvectors from a Nastran SOL103 result to CSV.

Run from inside META post-processor:
    File > Execute Script > export_modes.py

Outputs (in data/ansa_model/<variant>/):
    modal_trans_results.csv   (nNodes*3, nModes)  translational eigenvectors
    modal_rot_results.csv     (nNodes*3, nModes)  rotational eigenvectors
    modal_total_results.csv   (nNodes*6, nModes)  full [trans+rot] interleaved

DOF layout in modal_total_results.csv:
    rows ordered as [Ux0,Uy0,Uz0,Rx0,Ry0,Rz0, Ux1,...] (interleaved, Nastran G-set order)
"""

import numpy as np
from meta import models, results, groups
from config import INPUT_MODAL_DAT, INPUT_MODAL_OP2, OUTPUT_DIR


def get_eigenvectors(mode_list, grids_group):
    """Extract eigenvectors for all modes into a (nNodes*3, nModes) matrix."""
    matrix = []
    for mode in mode_list:
        x = np.asarray(grids_group.get_deformations(mode, 'all', numpy='x')).reshape(-1)
        y = np.asarray(grids_group.get_deformations(mode, 'all', numpy='y')).reshape(-1)
        z = np.asarray(grids_group.get_deformations(mode, 'all', numpy='z')).reshape(-1)
        n_nodes = len(x)
        # interleave: [x0,y0,z0, x1,y1,z1, ...]
        vec = np.empty(n_nodes * 3)
        vec[0::3] = x
        vec[1::3] = y
        vec[2::3] = z
        matrix.append(vec)
    return np.column_stack(matrix)


def interleave_trans_rot(trans, rot):
    """
    Combine (nNodes*3, nModes) trans and rot into (nNodes*6, nModes) interleaved.
    Both trans and rot are already in interleaved node order [x0,y0,z0, x1,...].
    Result row order: [Ux0,Uy0,Uz0,Rx0,Ry0,Rz0, Ux1,...] — matches Nastran G-set USET.
    """
    n_nodes   = trans.shape[0] // 3
    n_modes   = trans.shape[1]
    T = trans.reshape(n_nodes, 3, n_modes)   # (nNodes, 3, nModes)
    R = rot.reshape(n_nodes, 3, n_modes)     # (nNodes, 3, nModes)
    return np.concatenate([T, R], axis=1).reshape(6 * n_nodes, n_modes)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = models.LoadModel('MetaPost', str(INPUT_MODAL_DAT), 'NASTRAN')
    if model is None:
        raise RuntimeError(f"Could not load model from:\n  {INPUT_MODAL_DAT}")

    trans_modes = results.LoadDeformations(model.id, str(INPUT_MODAL_OP2),
                                           'NASTRAN', 'all',
                                           'Eigenvectors,Translational')
    rot_modes   = results.LoadDeformations(model.id, str(INPUT_MODAL_OP2),
                                           'NASTRAN', 'all',
                                           'Eigenvectors,Rotational')

    if not trans_modes or not rot_modes:
        raise ValueError(f"Could not load eigenvectors from:\n  {INPUT_MODAL_OP2}")

    all_grids = model.get_nodes('all')
    grids_grp = groups.CreateGroupFromNodes(model.id, 'all_grids', all_grids)

    print(f"Exporting {len(trans_modes)} modes...")

    trans_mat = get_eigenvectors(trans_modes, grids_grp)
    rot_mat   = get_eigenvectors(rot_modes,   grids_grp)
    print(f"trans_mat shape: {trans_mat.shape}")
    print(f"rot_mat shape  : {rot_mat.shape}")
    total_mat = interleave_trans_rot(trans_mat, rot_mat)

    np.savetxt(OUTPUT_DIR / 'modal_trans_results.csv',  trans_mat, delimiter=',')
    np.savetxt(OUTPUT_DIR / 'modal_rot_results.csv',    rot_mat,   delimiter=',')
    np.savetxt(OUTPUT_DIR / 'modal_total_results.csv',  total_mat, delimiter=',')

    print(f"Saved to {OUTPUT_DIR}")
    print(f"  modal_trans_results.csv : {trans_mat.shape}")
    print(f"  modal_rot_results.csv   : {rot_mat.shape}")
    print(f"  modal_total_results.csv : {total_mat.shape}")


if __name__ == '__main__':
    main()
