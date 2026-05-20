"""
Export modal eigenvectors from a Nastran SOL103 result to CSV.

Invoked by meta_runner/run_postprocess.py via:
    meta_post64.bat -b -s export_modes.py

Reads paths from environment variables set by the launcher:
    META_MODAL_DAT, META_MODAL_OP2, META_OUTPUT_DIR
"""

import os
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from meta import models, results, groups
from utils import save_csv, interleave_6dof, extract_deformation_matrix

INPUT_MODAL_DAT = Path(os.environ["META_MODAL_DAT"])
INPUT_MODAL_OP2 = Path(os.environ["META_MODAL_OP2"])
OUTPUT_DIR      = Path(os.environ["META_OUTPUT_DIR"])


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
        return groups.CreateGroupFromNodes(self.model.id, 'grp_modal', nodes)

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

    trans_mat = extract_deformation_matrix(trans_modes, grp_trans)
    rot_mat   = extract_deformation_matrix(rot_modes,   grp_rot)
    total_mat = interleave_6dof(trans_mat, rot_mat)

    save_csv(trans_mat, OUTPUT_DIR / 'modal_trans_results.csv')
    save_csv(rot_mat,   OUTPUT_DIR / 'modal_rot_results.csv')
    save_csv(total_mat, OUTPUT_DIR / 'modal_total_results.csv')


main()
