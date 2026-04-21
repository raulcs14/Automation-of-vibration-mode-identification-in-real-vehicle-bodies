"""
Export modal eigenvectors from a Nastran SOL103 result to CSV.

Run from inside META post-processor:
    File > Execute Script > export_modes.py

Outputs (in data/ansa_model/<variant>/):
    modal_trans_results.csv   (nNodes*3, nModes)
    modal_rot_results.csv     (nNodes*3, nModes)
    modal_total_results.csv   (nNodes*6, nModes)  full [trans+rot] interleaved
"""

import numpy as np
from meta import models, results, groups
from dataclasses import dataclass
from config import INPUT_MODAL_DAT, INPUT_MODAL_OP2, OUTPUT_DIR


@dataclass
class ModalAnalysis:
    inputfile: str
    out_103: str

    def __post_init__(self):
        self.model = models.LoadModel('MetaPost', self.inputfile, 'NASTRAN')

    @property
    def all_GRIDs(self):
        return self.model.get_nodes('all')

    def get_grp(self, nodes):
        return groups.CreateGroupFromNodes(self.model.id, 'grp_TB', nodes)

    @property
    def get_sol103_translational(self):
        sol103 = results.LoadDeformations(self.model.id, self.out_103,
                                          'NASTRAN', 'all',
                                          'Eigenvectors,Translational')
        if not sol103:
            raise ValueError('Empty translational eigenvectors')
        return sol103

    @property
    def get_sol103_rotational(self):
        sol103 = results.LoadDeformations(self.model.id, self.out_103,
                                          'NASTRAN', 'all',
                                          'Eigenvectors,Rotational')
        if not sol103:
            raise ValueError('Empty rotational eigenvectors')
        return sol103


def get_modes_eigvec(mode_list, grids_group):
    print(f"Number of modes: {len(mode_list)}")
    modes_matrix = []
    for mode in mode_list:
        eigvec = grids_group.get_deformations(mode, 'all', numpy='xyz')
        eigvec = np.array(eigvec).reshape(-1)
        modes_matrix.append(eigvec)
    print(f"Eigenvector shape: {eigvec.shape}")
    return np.column_stack(modes_matrix)


def get_total_eigenvectors(trans_modes, rot_modes):
    DOF, n_modes = trans_modes.shape
    n_nodes = DOF // 3
    T = trans_modes.reshape(n_nodes, 3, n_modes)
    R = rot_modes.reshape(n_nodes, 3, n_modes)
    return np.concatenate([T, R], axis=1).reshape(6 * n_nodes, n_modes)


def save_csv(matrix, path):
    np.savetxt(path, matrix, delimiter=',')
    print(f"Saved {path}  {matrix.shape}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mod_trans = ModalAnalysis(str(INPUT_MODAL_DAT), str(INPUT_MODAL_OP2))
    mod_rot   = ModalAnalysis(str(INPUT_MODAL_DAT), str(INPUT_MODAL_OP2))

    trans_modes = mod_trans.get_sol103_translational
    rot_modes   = mod_rot.get_sol103_rotational

    grids     = mod_trans.all_GRIDs
    grp_trans = mod_trans.get_grp(grids)
    grp_rot   = mod_rot.get_grp(grids)

    print(f"Modes: {len(trans_modes)}  |  Nodes: {len(grids)}")

    trans_mat = get_modes_eigvec(trans_modes, grp_trans)
    rot_mat   = get_modes_eigvec(rot_modes,   grp_rot)
    total_mat = get_total_eigenvectors(trans_mat, rot_mat)

    save_csv(trans_mat, OUTPUT_DIR / 'modal_trans_results.csv')
    save_csv(rot_mat,   OUTPUT_DIR / 'modal_rot_results.csv')
    save_csv(total_mat, OUTPUT_DIR / 'modal_total_results.csv')


if __name__ == '__main__':
    main()
