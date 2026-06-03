"""
Load modal results for the ANSA vehicle models directly from Epilysis H5 output.

Variants:
    "BIW" — Body in White (no lumped masses)
    "TB"  — Trimmed Body (with lumped masses)

Modal data (eigenvectors, frequencies, nodes):
    data/seat_model/<variant>/ansa/modal/output/000_Header_<variant>_modal_run.h5

K/M matrices:
    data/seat_model/<variant>/meta/matrices/matrices.h5  (PARAM,MAT2HDF)
"""

from pathlib import Path
import numpy as np
from seat_model.reader import read_h5, read_hdf5_modal, read_hdf5_conm2_node_ids
from common.rigid_body import build_rigid_body_basis

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SEAT_ROOT = _REPO_ROOT / "data" / "seat_model"
VARIANTS   = ["BIW", "TB"]

_H5_MODAL = {
    "BIW": _SEAT_ROOT / "BIW" / "ansa" / "modal" / "output" / "000_Header_BIW_modal_run.h5",
    "TB":  _SEAT_ROOT / "TB"  / "ansa" / "modal" / "output" / "000_Header_TB_modal_run.h5",
}

_H5_MATRICES = {
    "BIW": _SEAT_ROOT / "BIW" / "ansa" / "matrices" / "output" / "000_Header_BIW_getKM.h5",
    "TB":  _SEAT_ROOT / "TB"  / "ansa" / "matrices" / "output" / "000_Header_TB_getKM.h5",
}

N_RIGID_BODY_MODES = 6


def run_modal_analysis(variant: str = "BIW", skip_rigid: bool = True) -> dict:
    """
    Load ANSA modal results from the Epilysis H5 output file.

    Returns:
      modes            : (GDof, nModes)   elastic modes (rigid skipped if skip_rigid)
      freq             : (nModes,)        frequencies [Hz]
      K                : (GDof, GDof)     sparse stiffness matrix
      M                : (GDof, GDof)     sparse mass matrix
      R                : (GDof, 6)        rigid-body basis
      node_coordinates : (nNodes, 3)      coordinates
      node_ids         : (nNodes,)        GRID IDs
      conm2_node_ids   : (nCONM2,) int | None   TB only
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got '{variant}'")

    h5_modal    = _H5_MODAL[variant]
    h5_matrices = _H5_MATRICES[variant]

    for p in (h5_modal, h5_matrices):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found.")

    print(f"Loading ANSA modal data [{variant}] from {h5_modal.name}...")

    data     = read_hdf5_modal(h5_modal)
    modes_all = data["modes"]
    freq_all  = data["freq"]
    node_ids  = data["node_ids"]
    node_xyz  = data["node_xyz"]

    conm2_ids = read_hdf5_conm2_node_ids(h5_modal) if variant == "TB" else None
    if conm2_ids is not None:
        print(f"  CONM2 nodes  : {len(conm2_ids)}")

    km = read_h5(h5_matrices)
    K  = km["K"]
    M  = km["M"]

    GDof_total, n_total = modes_all.shape
    print(f"  Modes loaded : {n_total}  ({GDof_total} DOFs = {GDof_total // 6} nodes)")
    print(f"  Frequencies  : {freq_all[0]:.4f} ... {freq_all[-1]:.2f} Hz")

    if skip_rigid:
        modes = modes_all[:, N_RIGID_BODY_MODES:]
        freq  = freq_all[N_RIGID_BODY_MODES:]
        print(f"  Elastic modes: {modes.shape[1]}  ({freq[0]:.2f} - {freq[-1]:.2f} Hz)")
    else:
        modes = modes_all
        freq  = freq_all

    return dict(
        modes            = modes,
        freq             = freq,
        K                = K,
        M                = M,
        R                = build_rigid_body_basis(node_xyz),
        node_coordinates = node_xyz,
        node_ids         = node_ids,
        conm2_node_ids   = conm2_ids,
    )
