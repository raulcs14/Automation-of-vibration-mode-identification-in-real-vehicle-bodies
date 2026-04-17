"""
Readers for ANSA/Nastran outputs:
  - H5 file  → M, K matrices + node coordinates + DOF order (USET)
  - F06 file → eigenfrequencies
  - CSV files → mode shapes and reference displacements
"""

import re
import h5py
import numpy as np
import scipy.sparse as sp
from pathlib import Path


# ---------------------------------------------------------------------------
# H5 reader (M, K, nodes, USET)
# ---------------------------------------------------------------------------

def _read_csr_symmetric(assembly, info_row) -> sp.csr_matrix:
    """Read a CSR SYMMETRIC UPPER matrix from the EPILYSIS HDF5 layout and
    symmetrize it."""
    n    = int(info_row["NROW"])
    nnz  = int(info_row["NNZ"])
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

        # Node coordinates
        grid = f["EPILYSIS/INPUT/NODE/GRID"][:]
        node_ids = grid["ID"].astype(int)
        node_xyz = np.array([row["X"] for row in grid], dtype=float)

        # USET for G-set (full DOF set)
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

        pos = int(g_row["DATA_POS"])
        length = int(g_row["DATA_LEN"])
        uset_g = uset_data[pos : pos + length]   # structured array with ID, C

        # M and K matrices for G-set
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
    with open(f06_path, "r") as fh:
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


def read_frequencies_csv(csv_path: Path) -> np.ndarray:
    """Load frequencies [Hz] from a single-row CSV (exported by export_modes.py)."""
    data = np.loadtxt(csv_path, delimiter=",")
    return data.flatten()


# ---------------------------------------------------------------------------
# CSV readers (modes, reference displacement)
# ---------------------------------------------------------------------------

def read_modes_csv(csv_path: Path) -> np.ndarray:
    """
    Load modal eigenvectors from a CSV (nDOF × nModes).
    Rows are DOFs in interleaved layout: [Ux0,Uy0,Uz0,Rx0,Ry0,Rz0, Ux1,...].
    """
    return np.loadtxt(csv_path, delimiter=",")


def read_reference_csv(csv_path: Path) -> np.ndarray:
    """
    Load reference displacement vector(s) from a CSV.
    Returns (nDOF, nRefs) — works for both 1-D and 2-D CSV files.
    """
    data = np.loadtxt(csv_path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return data
