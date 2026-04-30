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
    BIW -> K.npz, M.npz  (node IDs and coords from getKM .f06 + BDF files)
    TB  -> matrices.h5   (node IDs, coords and CONM2 IDs from getKM .f06 + BDF files)

DOF space after loading:
    BIW: G-set minus SPC/singular nodes = A-set  (22818 DOFs, 3803 nodes)
    TB:  G-set minus CONM2 nodes                 (22968 DOFs, 3828 nodes)

To regenerate the CSVs, run export_modes.py from inside META post-processor.
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp
from ansa_model.reader import (read_csv, read_frequencies_f06, read_h5,
                                keep_mask_from_nodes, read_f06_biw, read_f06_tb)
from common.rigid_body import build_rigid_body_basis

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_ANSA_ROOT = _REPO_ROOT / "data" / "ansa_model"
VARIANTS   = ["BIW", "TB"]

_META_ROOT = Path(r"C:\Users\raulc\Documents\ProyectosGit\TFM\META\Test_Epilysis")

_F06_FILES = {
    "BIW": _META_ROOT / "BodyInWhite" / "dummycar_BIW_modal" / "output" / "000_Header_BIW_modal.f06",
    "TB":  _META_ROOT / "TrimmedBody" / "dummycar_TB" / "000_Header_TB_modal.f06",
}

# getKM .f06 — source of A-set node IDs (both variants)
_GETKM_F06 = {
    "BIW": _META_ROOT / "BodyInWhite" / "dummycar_BIW_matrices" / "output" / "000_Header_BIW_getKM.f06",
    "TB":  _META_ROOT / "TrimmedBody" / "dummycar_TB_matrices"  / "output" / "000_Header_TB_getKM.f06",
}

# Directories containing the BDF includes (same folder as the getKM .dat)
_BDF_DIR = {
    "BIW": _META_ROOT / "BodyInWhite" / "dummycar_BIW_matrices",
    "TB":  _META_ROOT / "TrimmedBody" / "dummycar_TB_matrices",
}

N_RIGID_BODY_MODES = 6


def run_modal_analysis(variant: str = "BIW", skip_rigid: bool = True) -> dict:
    """
    Load ANSA modal results and return a dict compatible with the pipeline.

    All arrays share the same DOF space (GDof rows):
      BIW: A-set DOFs  (G-set minus SPC/singular nodes)
      TB:  G-set minus CONM2 nodes

      modes            : (GDof, nModes)   elastic modes (rigid skipped if skip_rigid)
      freq             : (nModes,)        frequencies [Hz]
      K                : (GDof, GDof)     sparse stiffness matrix
      M                : (GDof, GDof)     sparse mass matrix
      R                : (GDof, 6)        rigid-body basis
      node_coordinates : (nNodes, 3)      coordinates matching DOF space
      node_ids         : (nNodes,)        GRID IDs matching DOF space

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
    modes_all = read_csv(modes_csv)              # (GDof_csv, nModes_total)
    if modes_all.ndim != 2 or modes_all.size == 0:
        raise ValueError(
            f"modal_total_results.csv for variant '{variant}' is empty or has wrong shape {modes_all.shape}.\n"
            f"Re-run ansa_model/meta_scripts/export_modes.py with VARIANT='{variant}' from META."
        )

    # --- Frequencies ---------------------------------------------------------
    freq_csv = DATA_DIR / "frequencies.csv"
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

    if variant == "BIW":
        # BIW: K/M are A-set sized. Filter modes from G-set to A-set.
        K = sp.load_npz(DATA_DIR / "K.npz")
        M = sp.load_npz(DATA_DIR / "M.npz")

        f06data       = read_f06_biw(_GETKM_F06["BIW"], bdf_files)
        node_ids      = f06data["node_ids"]      # A-set GRID IDs (3803,)
        node_xyz      = f06data["node_xyz"]      # A-set coords   (3803, 3)
        aset_dof_mask = f06data["aset_dof_mask"] # G-set -> A-set bool mask

        n_excl = int(np.sum(~aset_dof_mask))
        print(f"  G-set DOFs   : {GDof_csv}  ->  A-set DOFs: {int(np.sum(aset_dof_mask))}"
              f"  ({n_excl} excluded)")
        modes_all = modes_all[aset_dof_mask, :]

    else:
        # TB: K/M from H5 are G-set sized. Remove CONM2 DOFs from K, M and modes.
        h5_file = DATA_DIR / "matrices.h5"
        if not h5_file.exists():
            raise FileNotFoundError(
                f"{h5_file} not found.\n"
                f"Copy the Nastran H5 file to data/ansa_model/{variant}/matrices.h5\n"
                f"or run ansa_model/meta_scripts/export_matrices.py with VARIANT='{variant}'."
            )
        h5data = read_h5(h5_file)
        K      = h5data["K"]
        M      = h5data["M"]

        f06data   = read_f06_tb(_GETKM_F06["TB"], bdf_files)
        conm2_ids = f06data["conm2_node_ids"]

        conm2_keep_mask = keep_mask_from_nodes(conm2_ids, h5data["uset_g"])
        n_removed = int(np.sum(~conm2_keep_mask))
        print(f"  CONM2 nodes  : {len(conm2_ids)}  ({n_removed} DOFs removed)")

        # Apply mask to K, M and modes — all consistently in post-CONM2 G-set
        idx   = np.where(conm2_keep_mask)[0]
        K     = K[np.ix_(idx, idx)]
        M     = M[np.ix_(idx, idx)]
        modes_all = modes_all[conm2_keep_mask, :]

        # node_ids and node_xyz for the kept G-set nodes (one bool per node)
        node_mask = conm2_keep_mask[::6]
        node_ids  = h5data["node_ids"][node_mask]
        node_xyz  = h5data["node_xyz"][node_mask]

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
    )
