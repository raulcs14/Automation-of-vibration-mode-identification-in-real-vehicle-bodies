"""
Load pre-computed modal results for the ANSA vehicle models.

Variants:
    "BIW" — Body in White (no lumped masses)
    "TB"  — Trimmed Body (with lumped masses)

Primary data source (inside the repo, gitignored):
    data/ansa_model/<variant>/modal_total_results.csv   eigenvectors (nDOF x nModes)
    data/ansa_model/<variant>/frequencies.csv           frequencies [Hz] (1 x nModes)

Fallback for frequencies:
    data/ansa_model/<variant>/<variant>_modal.f06

M/K matrices:
    data/ansa_model/<variant>/matrices.h5

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
_ANSA_ROOT = _REPO_ROOT / "data" / "ansa_model"
VARIANTS   = ["BIW", "TB"]

_F06_FILES = {
    "BIW": Path(r"C:\Users\raulc\Documents\ProyectosGit\TFM\META\Test_Epilysis\BodyInWhite\dummycar_BIW_modal\output\000_Header_BIW_modal.f06"),
    "TB":  Path(r"C:\Users\raulc\Documents\ProyectosGit\TFM\META\Test_Epilysis\TrimmedBody\dummycar_TB\000_Header_TB_modal.f06"),
}

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


def run_modal_analysis(variant: str = "BIW", skip_rigid: bool = True) -> dict:
    """
    Load ANSA modal results and return a dict compatible with the pipeline:

      modes            : (GDof, nModes)   elastic modes (rigid skipped if skip_rigid)
      freq             : (nModes,)        frequencies [Hz]
      K                : (GDof, GDof)     sparse stiffness matrix
      M                : (GDof, GDof)     sparse mass matrix
      R                : (GDof, 6)        rigid-body basis
      node_coordinates : (nNodes, 3)
      node_ids         : (nNodes,)

    variant: "BIW" | "TB"
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got '{variant}'")

    DATA_DIR = _ANSA_ROOT / variant
    print(f"Loading ANSA modal data [{variant}]...")

    # --- Eigenvectors --------------------------------------------------------
    modes_csv = DATA_DIR / "modal_total_results.csv"
    if not modes_csv.exists():
        raise FileNotFoundError(
            f"{modes_csv} not found.\n"
            f"Run ansa_model/meta_scripts/export_modes.py with VARIANT='{variant}' from META."
        )
    modes_all = read_modes_csv(modes_csv)       # (GDof, nModes_total)
    if modes_all.ndim != 2 or modes_all.size == 0:
        raise ValueError(
            f"modal_total_results.csv for variant '{variant}' is empty or has wrong shape {modes_all.shape}.\n"
            f"Re-run ansa_model/meta_scripts/export_modes.py with VARIANT='{variant}' from META."
        )

    # --- Frequencies ---------------------------------------------------------
    freq_csv = DATA_DIR / "frequencies.csv"
    f06_file = _F06_FILES.get(variant)
    if freq_csv.exists():
        freq_all = read_frequencies_csv(freq_csv)
    elif f06_file and f06_file.exists():
        print(f"  (frequencies.csv not found, reading from {f06_file.name})")
        freq_all = read_frequencies_f06(f06_file)
    else:
        raise FileNotFoundError(
            f"Neither {freq_csv} nor the .f06 fallback found for variant '{variant}'.\n"
            "Re-run export_modes.py from META or provide the .f06 file."
        )

    # --- M, K matrices and node coordinates (from H5) ------------------------
    h5_file = DATA_DIR / "matrices.h5"
    if not h5_file.exists():
        raise FileNotFoundError(
            f"{h5_file} not found.\n"
            f"Copy the Nastran H5 file to data/ansa_model/{variant}/matrices.h5\n"
            f"or run ansa_model/meta_scripts/export_matrices.py with VARIANT='{variant}'."
        )
    h5data = read_h5(h5_file)

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
