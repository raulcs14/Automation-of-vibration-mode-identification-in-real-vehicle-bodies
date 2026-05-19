"""
Export static reference displacements from a Nastran SOL101 result to CSV.

Run from inside META post-processor:
    File > Execute Script > export_static_reference.py

Outputs (in data/ansa_model/<variant>/):
    ref_trans_results.csv    (nNodes*3, nCases)
    ref_rot_results.csv      (nNodes*3, nCases)
    ref_total_results.csv    (nNodes*6, nCases)  full [trans+rot] interleaved
"""

import numpy as np
from meta import models, results, groups
from dataclasses import dataclass
from config import INPUT_STATIC_DAT, INPUT_STATIC_OP2, OUTPUT_DIR_STATIC as OUTPUT_DIR
from utils import save_csv, interleave_6dof, extract_deformation_matrix


@dataclass
class StaticAnalysis:
    inputfile: str
    out_101: str

    def __post_init__(self):
        self.model = models.LoadModel('MetaPost', self.inputfile, 'NASTRAN')

    @property
    def all_GRIDs(self):
        return self.model.get_nodes('all')

    def get_grp(self, nodes):
        return groups.CreateGroupFromNodes(self.model.id, 'grp', nodes)

    @property
    def get_translational(self):
        sol101 = results.LoadDeformations(self.model.id, self.out_101,
                                          'NASTRAN', 'all',
                                          'Displacements,Translational')
        if not sol101:
            raise ValueError('Empty translational displacements')
        return sol101

    @property
    def get_rotational(self):
        sol101 = results.LoadDeformations(self.model.id, self.out_101,
                                          'NASTRAN', 'all',
                                          'Displacements,Rotational')
        if not sol101:
            raise ValueError('Empty rotational displacements')
        return sol101


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mod_trans = StaticAnalysis(str(INPUT_STATIC_DAT), str(INPUT_STATIC_OP2))
    mod_rot   = StaticAnalysis(str(INPUT_STATIC_DAT), str(INPUT_STATIC_OP2))

    displacements = mod_trans.get_translational
    rotations     = mod_rot.get_rotational

    grids       = mod_trans.all_GRIDs
    grp_trans   = mod_trans.get_grp(grids)
    grp_rot     = mod_rot.get_grp(grids)

    print(f"Load cases: {len(displacements)} trans, {len(rotations)} rot")
    print(f"Nodes: {len(grids)}")

    trans_mat = extract_deformation_matrix(displacements, grp_trans)
    rot_mat   = extract_deformation_matrix(rotations[:1],  grp_rot)
    # Replicate rot across all translational cases so shapes match
    rot_mat   = np.repeat(rot_mat, trans_mat.shape[1], axis=1)
    total_mat = interleave_6dof(trans_mat, rot_mat)

    save_csv(trans_mat, OUTPUT_DIR / 'ref_trans_results.csv')
    save_csv(rot_mat,   OUTPUT_DIR / 'ref_rot_results.csv')
    save_csv(total_mat, OUTPUT_DIR / 'ref_total_results.csv')


if __name__ == '__main__':
    main()
