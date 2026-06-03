"""
Readers for ANSA/Nastran outputs:
  - H5 (Epilysis modal/static output) -> eigenvectors, displacements, nodes, CONM2 IDs
  - H5 (PARAM,MAT2HDF matrices)       -> K, M sparse matrices
"""

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
# HDF5 reader (MDLPRM,HDF5,1) — eigenvectors, displacements, nodes, CONM2
# ---------------------------------------------------------------------------

def _hdf5_nodes(f) -> tuple:
    """Return (node_ids, node_xyz) from the Epilysis INPUT/NODE/GRID dataset."""
    grid = f["EPILYSIS/INPUT/NODE/GRID"][:]
    node_ids = grid["ID"].astype(int)
    node_xyz = np.array([row["X"] for row in grid], dtype=float)   # (nNodes, 3)
    return node_ids, node_xyz


def _hdf5_split_by_domain(flat_ds, domains_ds, node_ids: np.ndarray) -> np.ndarray:
    """
    Reconstruct a (nNodes, 6, nDomains) array from a flat Epilysis result dataset.

    The flat dataset has one row per (node, domain) pair identified by DOMAIN_ID.
    INDEX/.../POSITION+LENGTH could be used, but iterating unique domain IDs is
    simpler and robust.

    Returns ndarray shape (nNodes, 6, nDomains), columns = [X,Y,Z,RX,RY,RZ].
    """
    flat  = flat_ds[:]
    n_nodes   = len(node_ids)
    node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    domain_ids = np.unique(flat["DOMAIN_ID"])
    n_domains  = len(domain_ids)
    result     = np.zeros((n_nodes, 6, n_domains), dtype=float)

    for col, did in enumerate(domain_ids):
        mask  = flat["DOMAIN_ID"] == did
        chunk = flat[mask]
        for row in chunk:
            idx = node_id_to_idx.get(int(row["ID"]))
            if idx is not None:
                result[idx, :, col] = [row["X"], row["Y"], row["Z"],
                                       row["RX"], row["RY"], row["RZ"]]
    return result, domain_ids


def read_hdf5_modal(hdf5_path: Path) -> dict:
    """
    Read modal results from an Epilysis/Nastran H5 output file (MDLPRM,HDF5,1).

    Structure used:
      EPILYSIS/INPUT/NODE/GRID                → node IDs + coordinates
      EPILYSIS/RESULT/SUMMARY/EIGENVALUE      → FREQ column [Hz], ordered by MODE
      EPILYSIS/RESULT/NODAL/EIGENVECTOR       → flat (ID, X,Y,Z,RX,RY,RZ, DOMAIN_ID)
      EPILYSIS/RESULT/DOMAINS                 → maps DOMAIN_ID → MODE number

    Returns:
      node_ids : (nNodes,)          GRID IDs in input order
      node_xyz : (nNodes, 3)        coordinates [model units]
      freq     : (nModes,)          frequencies [Hz], sorted by mode number
      modes    : (6*nNodes, nModes) interleaved [Ux,Uy,Uz,Rx,Ry,Rz] per node
    """
    with h5py.File(hdf5_path, "r") as f:
        node_ids, node_xyz = _hdf5_nodes(f)

        # Frequencies: SUMMARY/EIGENVALUE sorted by MODE field
        eig_ds = f["EPILYSIS/RESULT/SUMMARY/EIGENVALUE"][:]
        order  = np.argsort(eig_ds["MODE"])
        freq   = eig_ds["FREQ"][order].astype(float)          # (nModes,)

        # Eigenvectors: flat array, split by DOMAIN_ID
        ev_flat   = f["EPILYSIS/RESULT/NODAL/EIGENVECTOR"]
        domains   = f["EPILYSIS/RESULT/DOMAINS"][:]
        result, domain_ids = _hdf5_split_by_domain(ev_flat, domains, node_ids)
        # result: (nNodes, 6, nDomains) — domains ordered by DOMAIN_ID ascending

        # Map domain order to mode order using DOMAINS table
        # domains["ID"] → domains["MODE"] (1-based)
        dom_id_to_mode = {int(d["ID"]): int(d["MODE"]) for d in domains}
        mode_order = np.array([dom_id_to_mode[int(did)] for did in domain_ids])
        sort_idx   = np.argsort(mode_order)
        result     = result[:, :, sort_idx]                   # (nNodes, 6, nModes)

    n_nodes, _, n_modes = result.shape
    # Interleave to (6*nNodes, nModes): [Ux,Uy,Uz,Rx,Ry,Rz] per node
    modes = result.reshape(6 * n_nodes, n_modes, order='C')

    # Trim freq to actual number of domains (SUMMARY may include DOMAIN 0 summary row)
    freq = freq[:n_modes]

    return dict(node_ids=node_ids, node_xyz=node_xyz, freq=freq, modes=modes)


def read_hdf5_static(hdf5_path: Path) -> dict:
    """
    Read static displacement results from an Epilysis/Nastran H5 output file.

    Structure used:
      EPILYSIS/INPUT/NODE/GRID             → node IDs + coordinates
      EPILYSIS/RESULT/NODAL/DISPLACEMENT   → flat (ID, X,Y,Z,RX,RY,RZ, DOMAIN_ID)
      EPILYSIS/RESULT/DOMAINS              → maps DOMAIN_ID → SUBCASE number

    Returns:
      node_ids : (nNodes,)         GRID IDs in input order
      node_xyz : (nNodes, 3)       coordinates
      refs     : (6*nNodes, nRefs) interleaved [Ux,Uy,Uz,Rx,Ry,Rz] per node,
                                   one column per subcase sorted by SUBCASE number
    """
    with h5py.File(hdf5_path, "r") as f:
        node_ids, node_xyz = _hdf5_nodes(f)

        disp_flat = f["EPILYSIS/RESULT/NODAL/DISPLACEMENT"]
        domains   = f["EPILYSIS/RESULT/DOMAINS"][:]
        result, domain_ids = _hdf5_split_by_domain(disp_flat, domains, node_ids)

        # Sort columns by SUBCASE number
        dom_id_to_sub = {int(d["ID"]): int(d["SUBCASE"]) for d in domains}
        sub_order  = np.array([dom_id_to_sub[int(did)] for did in domain_ids])
        sort_idx   = np.argsort(sub_order)
        result     = result[:, :, sort_idx]                   # (nNodes, 6, nRefs)

    n_nodes, _, n_refs = result.shape
    refs = result.reshape(6 * n_nodes, n_refs, order='C')
    return dict(node_ids=node_ids, node_xyz=node_xyz, refs=refs)


def read_hdf5_conm2_node_ids(hdf5_path: Path) -> np.ndarray:
    """
    Extract GRID IDs referenced by CONM2 elements from an Epilysis H5 file.
    Returns sorted unique (nCONM2,) int array. Empty array if none present.
    """
    with h5py.File(hdf5_path, "r") as f:
        conm2_path = "EPILYSIS/INPUT/ELEMENT/CONM2"
        if conm2_path not in f:
            return np.array([], dtype=int)
        grid_ids = np.asarray(f[conm2_path]["G"], dtype=int)
    return np.unique(grid_ids)


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



