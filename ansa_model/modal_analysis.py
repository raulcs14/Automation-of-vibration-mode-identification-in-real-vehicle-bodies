"""
Load pre-computed modal results for the ANSA Trimmed-Body model.

Primary data source (inside the repo, gitignored):
    data/ansa_model/modal_total_results.csv   eigenvectors (nDOF x nModes)
    data/ansa_model/frequencies.csv           frequencies [Hz] (1 x nModes)

Fallback for M/K matrices and node coordinates (outside the repo, confidential):
    META/Test_Epilysis/dummycar_TB_matrices/000_Header_TB_getKM.h5

To regenerate the CSVs, run export_modes.py from inside META post-processor.
"""

from pathlib import Path
import numpy as np
from ansa_model.reader import (read_modes_csv, read_frequencies_csv,
                                read_frequencies_f06, read_h5)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR   = _REPO_ROOT / "data" / "ansa_model"

_H5_FILE  = DATA_DIR / "matrices.h5"
_F06_FILE = Path(r"C:\Users\raulc\Documents\ProyectosGit\TFM\META\Test_Epilysis\dummycar_TB\000_Header_TB_modal.f06")

N_RIGID_BODY_MODES = 6


def build_rigid_body_basis(node_xyz: np.ndarray) -> np.ndarray:
    """Build rigid-body basis R (6*N x 6). DOF order: [Ux,Uy,Uz,Rx,Ry,Rz]."""
    n_nodes = node_xyz.shape[0]
    R = np.zeros((6 * n_nodes, 6))
    for i, (x, y, z) in enumerate(node_xyz):
        R[6*i + 0, 0] = 1.0;  R[6*i + 1, 1] = 1.0;  R[6*i + 2, 2] = 1.0
        R[6*i + 1, 3] = -z;   R[6*i + 2, 3] =  y
        R[6*i + 0, 4] =  z;   R[6*i + 2, 4] = -x
        R[6*i + 0, 5] = -y;   R[6*i + 1, 5] =  x
    return R


def run_modal_analysis(skip_rigid: bool = True) -> dict:
    """
    Load ANSA modal results and return a dict compatible with the simple_model
    pipeline:

      modes            : (GDof, nModes)   elastic modes (rigid skipped if skip_rigid)
      freq             : (nModes,)        frequencies [Hz]
      K                : (GDof, GDof)     sparse stiffness matrix
      M                : (GDof, GDof)     sparse mass matrix
      R                : (GDof, 6)        rigid-body basis
      node_coordinates : (nNodes, 3)
      node_ids         : (nNodes,)
    """
    print("Loading ANSA modal data...")

    # --- Eigenvectors --------------------------------------------------------
    modes_csv = DATA_DIR / "modal_total_results.csv"
    if not modes_csv.exists():
        raise FileNotFoundError(
            f"{modes_csv} not found.\n"
            "Run ansa_model/meta_scripts/export_modes.py from META to generate it."
        )
    modes_all = read_modes_csv(modes_csv)       # (GDof, nModes_total)

    # --- Frequencies ---------------------------------------------------------
    freq_csv = DATA_DIR / "frequencies.csv"
    if freq_csv.exists():
        freq_all = read_frequencies_csv(freq_csv)
    elif _F06_FILE.exists():
        print(f"  (frequencies.csv not found, reading from {_F06_FILE.name})")
        freq_all = read_frequencies_f06(_F06_FILE)
    else:
        raise FileNotFoundError(
            f"Neither {freq_csv} nor {_F06_FILE} found.\n"
            "Re-run export_modes.py from META or provide the .f06 file."
        )

    # --- M, K matrices and node coordinates (from H5) ------------------------
    if not _H5_FILE.exists():
        raise FileNotFoundError(
            f"{_H5_FILE} not found.\n"
            "Copy the Nastran H5 file to data/ansa_model/matrices.h5"
        )
    h5data = read_h5(_H5_FILE)

    GDof, n_total = modes_all.shape
    n_nodes = GDof // 6
    print(f"  Modes loaded : {n_total}  ({GDof} DOFs, {n_nodes} nodes)")
    print(f"  Frequencies  : {freq_all[0]:.4f} ... {freq_all[-1]:.2f} Hz")

    if skip_rigid:
        modes = modes_all[:, N_RIGID_BODY_MODES:]
        freq  = freq_all[N_RIGID_BODY_MODES:]
        print(f"  Elastic modes kept: {modes.shape[1]}  "
              f"({freq[0]:.2f} – {freq[-1]:.2f} Hz)")
    else:
        modes = modes_all
        freq  = freq_all

    node_xyz = h5data["node_xyz"]
    node_ids = h5data["node_ids"]
    R = build_rigid_body_basis(node_xyz)

    return dict(
        modes            = modes,
        freq             = freq,
        K                = h5data["K"],
        M                = h5data["M"],
        R                = R,
        node_coordinates = node_xyz,
        node_ids         = node_ids,
    )
