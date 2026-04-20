"""
Export static reference displacements from a Nastran SOL101 result to CSV.

Run from inside META post-processor:
    File > Execute Script > export_static_reference.py

Outputs (in data/ansa_model/<variant>/):
    ref_trans_results.csv    (nNodes*3, nCases)  translational displacements
    ref_rot_results.csv      (nNodes*3, nCases)  rotational displacements
    ref_total_results.csv    (nNodes*6, nCases)  full [trans+rot] interleaved

DOF layout matches modal_total_results.csv (interleaved G-set order).
"""

import numpy as np
from meta import models, results, groups
from config import INPUT_STATIC_DAT, INPUT_STATIC_OP2, OUTPUT_DIR


def get_displacement_matrix(case_list, grids_group):
    """Extract displacements for all load cases into a (nNodes*3, nCases) matrix."""
    matrix = []
    for case in case_list:
        vec = grids_group.get_deformations(case, 'all', numpy='xyz')
        matrix.append(np.asarray(vec).reshape(-1))
    return np.column_stack(matrix) if len(matrix) > 1 else np.asarray(matrix[0]).reshape(-1, 1)


def interleave_trans_rot(trans, rot):
    """Combine (nNodes*3, nCases) trans and rot into (nNodes*6, nCases) interleaved."""
    if trans.ndim == 1:
        trans = trans.reshape(-1, 1)
    if rot.ndim == 1:
        rot = rot.reshape(-1, 1)
    n_dof, n_cases = trans.shape
    n_nodes = n_dof // 3
    T = trans.reshape(n_nodes, 3, n_cases)
    R = rot.reshape(n_nodes, 3, n_cases)
    return np.concatenate([T, R], axis=1).reshape(6 * n_nodes, n_cases)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = models.LoadModel('MetaPost', str(INPUT_STATIC_DAT), 'NASTRAN')
    if model is None:
        raise RuntimeError(f"Could not load model from:\n  {INPUT_STATIC_DAT}")

    trans_cases = results.LoadDeformations(model.id, str(INPUT_STATIC_OP2),
                                           'NASTRAN', 'all',
                                           'Displacements,Translational')
    rot_cases   = results.LoadDeformations(model.id, str(INPUT_STATIC_OP2),
                                           'NASTRAN', 'all',
                                           'Displacements,Rotational')

    if not trans_cases or not rot_cases:
        raise ValueError(f"Could not load displacements from:\n  {INPUT_STATIC_OP2}")

    all_grids = model.get_nodes('all')
    grids_grp = groups.CreateGroupFromNodes(model.id, 'all_grids', all_grids)

    print(f"Exporting {len(trans_cases)} load cases...")

    trans_mat = get_displacement_matrix(trans_cases, grids_grp)
    rot_mat   = get_displacement_matrix(rot_cases,   grids_grp)
    total_mat = interleave_trans_rot(trans_mat, rot_mat)

    np.savetxt(OUTPUT_DIR / 'ref_trans_results.csv',  trans_mat, delimiter=',')
    np.savetxt(OUTPUT_DIR / 'ref_rot_results.csv',    rot_mat,   delimiter=',')
    np.savetxt(OUTPUT_DIR / 'ref_total_results.csv',  total_mat, delimiter=',')

    print(f"Saved to {OUTPUT_DIR}")
    print(f"  ref_trans_results.csv : {trans_mat.shape}")
    print(f"  ref_rot_results.csv   : {rot_mat.shape}")
    print(f"  ref_total_results.csv : {total_mat.shape}")


if __name__ == '__main__':
    main()
