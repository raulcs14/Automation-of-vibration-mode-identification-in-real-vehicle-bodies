"""
Load pre-computed modal results for the ANSA vehicle models.

Variants:
    "BIW" — Body in White (no lumped masses)
    "TB"  — Trimmed Body (with lumped masses)

Primary data source (inside the repo, gitignored):
    data/seat_model/<variant>/meta/modal/modal_total_results.csv   eigenvectors (nDOF x nModes)
    data/seat_model/<variant>/meta/modal/frequencies.csv           frequencies [Hz] (1 x nModes)

Fallback for frequencies:
    data/seat_model/<variant>/ansa/modal/output/000_Header_<variant>_modal_run.f06

M/K matrices (both variants, G-set):
    data/seat_model/<variant>/meta/matrices/matrices.h5
    (copied from ansa/matrices/output/ by meta_runner/scripts/export_matrices.py)

DOF space after loading:
    Both BIW and TB are returned in the full G-set from the Epilysis H5.
    TB: conm2_node_ids provided for optional CONM2 DOF removal.

To regenerate the CSVs, run seat_model/meta_runner/scripts/export_modes.py from META.
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp
from seat_model.reader import (read_csv, read_frequencies_f06, read_h5)
from common.rigid_body import build_rigid_body_basis

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT  = Path(__file__).resolve().parents[1]
_SEAT_ROOT  = _REPO_ROOT / "data" / "seat_model"
VARIANTS    = ["BIW", "TB"]

_F06_FILES = {
    "BIW": _SEAT_ROOT / "BIW" / "ansa" / "modal"    / "output" / "000_Header_BIW_modal_run.f06",
    "TB":  _SEAT_ROOT / "TB"  / "ansa" / "modal"    / "output" / "000_Header_TB_modal_run.f06",
}

# getKM .f06 — source of A-set node IDs (both variants)
_GETKM_F06 = {
    "BIW": _SEAT_ROOT / "BIW" / "ansa" / "matrices" / "output" / "000_Header_BIW_getKM.f06",
    "TB":  _SEAT_ROOT / "TB"  / "ansa" / "matrices" / "output" / "000_Header_TB_getKM.f06",
}

# Directories containing the BDF includes (.bdf/.nas files with GRID cards)
_BDF_DIR = {
    "BIW": _SEAT_ROOT / "BIW" / "ansa" / "matrices" / "input",
    "TB":  _SEAT_ROOT / "TB"  / "ansa" / "matrices" / "input",
}

N_RIGID_BODY_MODES = 6


def run_modal_analysis(variant: str = "BIW", skip_rigid: bool = True) -> dict:
    """
    Load ANSA modal results and return a dict compatible with the pipeline.

    Returns a dict with:
      modes            : (GDof, nModes)   elastic modes (rigid skipped if skip_rigid)
      freq             : (nModes,)        frequencies [Hz]
      K                : (GDof, GDof)     sparse stiffness matrix
      M                : (GDof, GDof)     sparse mass matrix
      R                : (GDof, 6)        rigid-body basis
      node_coordinates : (nNodes, 3)      coordinates matching DOF space
      node_ids         : (nNodes,)        GRID IDs matching DOF space
      conm2_keep_mask  : (GDof,) bool | None
                         TB only: True = keep this DOF (CONM2 DOFs are False).
                         Apply with apply_dof_mask() to remove CONM2 nodes.
                         None for BIW (already at A-set, no CONM2).

    variant: "BIW" | "TB"
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got '{variant}'")

    DATA_DIR = _SEAT_ROOT / variant / "meta"
    print(f"Loading ANSA modal data [{variant}]...")

    # --- Eigenvectors --------------------------------------------------------
    modes_csv = DATA_DIR / "modal" / "modal_total_results.csv"
    if not modes_csv.exists():
        raise FileNotFoundError(
            f"{modes_csv} not found.\n"
            f"Run seat_model/meta_runner/scripts/export_modes.py with VARIANT='{variant}' from META."
        )
    modes_all = read_csv(modes_csv)              # (GDof_csv, nModes_total)
    if modes_all.ndim != 2 or modes_all.size == 0:
        raise ValueError(
            f"modal_total_results.csv for variant '{variant}' is empty or has wrong shape {modes_all.shape}.\n"
            f"Re-run seat_model/meta_runner/scripts/export_modes.py with VARIANT='{variant}' from META."
        )

    # --- Frequencies ---------------------------------------------------------
    freq_csv = DATA_DIR / "modal" / "frequencies.csv"
    f06_file = _F06_FILES.get(variant)
    if freq_csv.exists():
        freq_all = read_csv(freq_csv, flatten=True)
    elif f06_file and f06_file.exists():
        print(f"  (frequencies.csv not found, reading from {f06_file.name})")
        freq_all = read_frequencies_f06(f06_file)
    else:
        raise FileNotFoundError(
            f"Neither {freq_csv} nor the .f06 fallback found for variant '{variant}'.\n"
            "Re-run export_modes.py from META or provide the .f06 file."
        )

    # --- Align freq to modes (f06 may have trailing zeros beyond CSV columns) -
    GDof_csv, n_total = modes_all.shape
    freq_all = freq_all[:n_total]
    print(f"  Modes loaded : {n_total}  ({GDof_csv} DOFs = {GDof_csv // 6} nodes, G-set)")
    print(f"  Frequencies  : {freq_all[0]:.4f} ... {freq_all[-1]:.2f} Hz")

    # --- Load K, M and build DOF mask ----------------------------------------
    _NAS_EXTS = ("*.bdf", "*.nas", "*.dat", "*.inc")
    bdf_files = sorted({f for ext in _NAS_EXTS for f in _BDF_DIR[variant].glob(ext)})
    conm2_ids = None

    # Both BIW and TB: K/M come from the H5. Node IDs and coordinates are
    # read directly from the H5 (Epilysis A-set = G-set for both variants).
    # No F06 USET parsing needed — Epilysis does not print USET tables.
    h5_file = DATA_DIR / "matrices" / "matrices.h5"
    if not h5_file.exists():
        raise FileNotFoundError(
            f"{h5_file} not found.\n"
            f"Run the matrices task from meta_runner/run_postprocess.py for {variant}."
        )
    h5data   = read_h5(h5_file)
    K        = h5data["K"]
    M        = h5data["M"]
    node_ids = h5data["node_ids"]
    node_xyz = h5data["node_xyz"]

    if variant == "TB":
        # TB only: read CONM2 node IDs from BDF files for optional removal.
        from seat_model.reader import read_bdf_conm2_node_ids
        conm2_ids = read_bdf_conm2_node_ids(bdf_files)
        print(f"  CONM2 nodes  : {len(conm2_ids)}  (removable via conm2_node_ids)")

    # --- Select elastic modes -------------------------------------------------
    if skip_rigid:
        modes = modes_all[:, N_RIGID_BODY_MODES:]
        freq  = freq_all[N_RIGID_BODY_MODES:]
        print(f"  Elastic modes kept: {modes.shape[1]}  "
              f"({freq[0]:.2f} - {freq[-1]:.2f} Hz)")
    else:
        modes = modes_all
        freq  = freq_all

    R = build_rigid_body_basis(node_xyz)

    return dict(
        modes            = modes,
        freq             = freq,
        K                = K,
        M                = M,
        R                = R,
        node_coordinates = node_xyz,
        node_ids         = node_ids,
        conm2_node_ids   = conm2_ids if variant == "TB" else None,
    )
