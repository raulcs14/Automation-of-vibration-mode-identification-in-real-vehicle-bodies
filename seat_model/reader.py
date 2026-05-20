"""
Readers for ANSA/Nastran outputs:
  - H5 file  -> M, K matrices + node coordinates (Epilysis G-set, both BIW and TB)
  - F06 file -> eigenfrequencies (fallback when frequencies.csv is absent)
  - BDF      -> CONM2 node IDs (TB only, for optional mass-node removal)
  - CSV      -> mode shapes and reference displacements
"""

import re
import h5py
import numpy as np
import scipy.sparse as sp
from pathlib import Path


# ---------------------------------------------------------------------------
# H5 reader (TB: M, K, nodes, USET)
# ---------------------------------------------------------------------------

def _read_csr_symmetric(assembly, info_row) -> sp.csr_matrix:
    """Read a CSR SYMMETRIC UPPER matrix from the EPILYSIS HDF5 layout and
    symmetrize it."""
    n    = int(info_row["NROW"])
    ia_p = int(info_row["IA_POS"]); ia_l = int(info_row["IA_LEN"])
    ja_p = int(info_row["JA_POS"]); ja_l = int(info_row["JA_LEN"])
    va_p = int(info_row["VA_POS"]); va_l = int(info_row["VA_LEN"])

    ia = np.asarray(assembly["MATRIX/IA"]["VALUE"][ia_p : ia_p + ia_l])
    ja = np.asarray(assembly["MATRIX/JA"]["VALUE"][ja_p : ja_p + ja_l])
    va = np.asarray(assembly["MATRIX/VA"]["VALUE"][va_p : va_p + va_l])

    upper = sp.csr_matrix((va, ja, ia), shape=(n, n))
    mat   = upper + upper.T - sp.diags(upper.diagonal())
    return mat.tocsr()


def read_h5(h5_path: Path) -> dict:
    """
    Read the EPILYSIS HDF5 file and return:
      node_ids   : (nNodes,)     original Nastran GRID IDs
      node_xyz   : (nNodes, 3)   coordinates
      uset_g     : (nDOF, 2)     [(node_id, dof_component)] for G-set
      K          : (nDOF, nDOF)  sparse CSR stiffness matrix (G-set)
      M          : (nDOF, nDOF)  sparse CSR mass matrix (G-set)
    """
    with h5py.File(h5_path, "r") as f:
        assembly = f["EPILYSIS/ASSEMBLY"]

        grid     = f["EPILYSIS/INPUT/NODE/GRID"][:]
        node_ids = grid["ID"].astype(int)
        node_xyz = np.array([row["X"] for row in grid], dtype=float)

        uset_info = assembly["USET/INFO"][:]
        uset_data = assembly["USET/DATA"][:]

        g_row = None
        for row in uset_info:
            dof_set = row["DOF_SET"]
            if isinstance(dof_set, bytes):
                dof_set = dof_set.decode()
            if dof_set.strip() == "G":
                g_row = row
                break
        if g_row is None:
            raise ValueError("G-set not found in USET/INFO")

        pos    = int(g_row["DATA_POS"])
        length = int(g_row["DATA_LEN"])
        uset_g = uset_data[pos : pos + length]

        mat_info = assembly["MATRIX/INFO"][:]
        K_mat = M_mat = None
        for row in mat_info:
            dof_set = row["DOF_SET"]
            if isinstance(dof_set, bytes):
                dof_set = dof_set.decode()
            name = row["MATRIX"]
            if isinstance(name, bytes):
                name = name.decode()
            if dof_set.strip() != "G":
                continue
            if name.strip() == "K":
                K_mat = _read_csr_symmetric(assembly, row)
            elif name.strip() == "M":
                M_mat = _read_csr_symmetric(assembly, row)

    if K_mat is None or M_mat is None:
        raise ValueError("Could not find K and/or M matrices for G-set in H5 file")

    return dict(node_ids=node_ids, node_xyz=node_xyz, uset_g=uset_g,
                K=K_mat, M=M_mat)


# ---------------------------------------------------------------------------
# DOF-set mask helper (used in tests and for G-set -> A-set filtering)
# ---------------------------------------------------------------------------

def aset_dof_mask_from_gset(aset_node_ids: np.ndarray,
                             gset_node_ids: np.ndarray) -> np.ndarray:
    """
    Build a boolean DOF-level keep-mask aligned with the G-set.

    Returns a 1-D bool array of length len(gset_node_ids)*6 where True means
    the DOF belongs to a node in the A-set (free structural node that enters
    K and M). Assumes 6 DOFs per node, nodes sorted ascending.
    """
    node_in_aset = np.isin(gset_node_ids, aset_node_ids)
    return np.repeat(node_in_aset, 6)


# ---------------------------------------------------------------------------
# BDF readers (node coordinates and CONM2 IDs)
# ---------------------------------------------------------------------------

_NASTRAN_FLOAT = re.compile(r"([+-]?\d+\.?\d*)([+-]\d+)$")


def _parse_nastran_float(s: str) -> float:
    """Parse a Nastran real, handling implicit-exponent form (e.g. '1.5-3' = 1.5e-3)."""
    s = s.strip()
    m = _NASTRAN_FLOAT.match(s)
    if m:
        return float(m.group(1) + "e" + m.group(2))
    return float(s)


def read_bdf_node_coords(bdf_paths: list[Path]) -> dict[int, np.ndarray]:
    """
    Parse Nastran BDF files and return a dict mapping GRID ID -> xyz (shape (3,)).

    Uses Nastran small-field fixed-format (8-char columns):
        cols  1- 8: GRID
        cols  9-16: ID
        cols 17-24: CP (may be blank)
        cols 25-32: X1
        cols 33-40: X2
        cols 41-48: X3
    """
    coords: dict[int, np.ndarray] = {}
    for bdf_path in bdf_paths:
        for line in bdf_path.read_text(encoding="latin-1").splitlines():
            if not line.upper().startswith("GRID"):
                continue
            line = line.ljust(48)
            try:
                nid = int(line[8:16])
                xyz = np.array([_parse_nastran_float(line[24:32]),
                                _parse_nastran_float(line[32:40]),
                                _parse_nastran_float(line[40:48])])
            except ValueError:
                continue
            coords[nid] = xyz
    return coords


def read_bdf_conm2_node_ids(bdf_paths: list[Path]) -> np.ndarray:
    """
    Parse Nastran BDF files and return the GRID IDs referenced by CONM2 elements,
    sorted ascending.

    CONM2 small-field format:
        cols  1- 8: CONM2
        cols  9-16: EID  (element ID)
        cols 17-24: G    (GRID ID the mass is attached to)
    """
    node_ids: set[int] = set()
    for bdf_path in bdf_paths:
        for line in bdf_path.read_text(encoding="latin-1").splitlines():
            if not line.upper().startswith("CONM2"):
                continue
            line = line.ljust(24)
            try:
                node_ids.add(int(line[8:16]))  # EID, not used
                node_ids.discard(int(line[8:16]))
                node_ids.add(int(line[16:24]))  # G — the attached GRID
            except ValueError:
                continue
    return np.array(sorted(node_ids), dtype=int)


# ---------------------------------------------------------------------------
# F06 reader (frequencies)
# ---------------------------------------------------------------------------

def read_frequencies_f06(f06_path: Path) -> np.ndarray:
    """
    Parse a Nastran .f06 file and return an array of frequencies [Hz]
    for each mode (including rigid-body modes with ~0 Hz).
    """
    pattern = re.compile(
        r"^\s+(\d+)\s+"          # mode number
        r"[\deE+\-\.]+\s+"       # eigenvalue
        r"[\deE+\-\.]+\s+"       # radians/s
        r"([\deE+\-\.]+)"        # CYCLES (Hz)
    )
    freqs = {}
    with open(f06_path, "r", encoding="latin-1") as fh:
        for line in fh:
            m = pattern.match(line)
            if m:
                mode_n = int(m.group(1))
                hz     = float(m.group(2))
                freqs[mode_n] = hz

    if not freqs:
        raise ValueError(f"No frequency data found in {f06_path}")

    n_modes = max(freqs.keys())
    freq = np.array([freqs.get(i, 0.0) for i in range(1, n_modes + 1)])
    return freq


# ---------------------------------------------------------------------------
# Generic CSV reader
# ---------------------------------------------------------------------------

def read_csv(csv_path: Path, dtype=float, flatten: bool = False,
             ensure_2d: bool = False) -> np.ndarray:
    """
    Load a numeric CSV produced by the META export scripts.

    dtype     : numpy dtype for the array (default float)
    flatten   : return a 1-D array (for frequencies, node ID lists)
    ensure_2d : if the loaded array is 1-D, reshape to (n, 1) (for reference vectors)
    """
    data = np.loadtxt(csv_path, delimiter=",", dtype=dtype)
    if flatten:
        return data.flatten()
    if ensure_2d and data.ndim == 1:
        data = data.reshape(-1, 1)
    return data

