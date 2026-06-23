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


def read_hdf5_element_stress(hdf5_path: Path, mode: int) -> dict:
    """
    Read CQUAD4 and CTRIA3 in-plane stresses for a given mode number (1-based).

    For each shell element the H5 stores two through-thickness fibre results:
      QUAD4 fields : EID, FD1, X1, Y1, XY1, FD2, X2, Y2, XY2, DOMAIN_ID
      TRIA3 fields : EID, FD1, X1, Y1, TXY1, FD2, X2, Y2, TXY2, DOMAIN_ID
    where X/Y are in-plane normal stresses, XY/TXY is in-plane shear (τ_xy),
    FD1/FD2 are the fibre distances (e.g. ±T/2).

    The element centroid is computed as the mean of its corner node coordinates.

    Returns a dict with keys:
      eid        : (nElem,)    element IDs
      centroid   : (nElem, 3)  centroid coordinates in model units (mm)
      pid        : (nElem,)    property IDs
      elem_type  : (nElem,)    element type string ('QUAD4' or 'TRIA3')
      sigma_x1   : (nElem,)   σ_x  top fibre
      sigma_y1   : (nElem,)   σ_y  top fibre
      tau_xy1    : (nElem,)   τ_xy top fibre
      sigma_x2   : (nElem,)   σ_x  bot fibre
      sigma_y2   : (nElem,)   σ_y  bot fibre
      tau_xy2    : (nElem,)   τ_xy bot fibre
      tau_xy_avg : (nElem,)   mean(|τ_xy1|, |τ_xy2|) — scalar summary per element
    """
    with h5py.File(hdf5_path, "r") as f:
        # --- node coordinates ---
        grid = f["EPILYSIS/INPUT/NODE/GRID"][:]
        nid_arr = grid["ID"].astype(int)
        xyz_arr = np.array([row["X"] for row in grid], dtype=float)  # (nNodes, 3)
        nid_to_idx = {nid: i for i, nid in enumerate(nid_arr)}

        # --- domain_id for requested mode ---
        domains = f["EPILYSIS/RESULT/DOMAINS"][:]
        candidates = [int(d["ID"]) for d in domains if int(d["MODE"]) == mode]
        if not candidates:
            raise ValueError(f"Mode {mode} not found in EPILYSIS/RESULT/DOMAINS")
        target_domain = candidates[0]

        # --- index tables: DOMAIN_ID -> (POSITION, LENGTH) ---
        def _get_slice(index_ds, domain_id):
            for row in index_ds:
                if int(row["DOMAIN_ID"]) == domain_id:
                    return int(row["POSITION"]), int(row["LENGTH"])
            raise ValueError(f"DOMAIN_ID {domain_id} not found in index")

        idx_q4  = f["INDEX/EPILYSIS/RESULT/ELEMENTAL/STRESS/QUAD4"][:]
        idx_t3  = f["INDEX/EPILYSIS/RESULT/ELEMENTAL/STRESS/TRIA3"][:]
        pos_q4, len_q4 = _get_slice(idx_q4, target_domain)
        pos_t3, len_t3 = _get_slice(idx_t3, target_domain)

        stress_q4 = f["EPILYSIS/RESULT/ELEMENTAL/STRESS/QUAD4"][pos_q4 : pos_q4 + len_q4]
        stress_t3 = f["EPILYSIS/RESULT/ELEMENTAL/STRESS/TRIA3"][pos_t3 : pos_t3 + len_t3]

        # --- element connectivity (DOMAIN_ID=1 for input) ---
        cquad4 = f["EPILYSIS/INPUT/ELEMENT/CQUAD4"][:]
        ctria3 = f["EPILYSIS/INPUT/ELEMENT/CTRIA3"][:]

    # --- build centroid lookup from connectivity ---
    def _centroids(elems, n_corners):
        """elems: structured array with EID, PID, G fields."""
        eid  = elems["EID"].astype(int)
        pid  = elems["PID"].astype(int)
        g    = elems["G"]                          # (nElem, n_corners) int
        cent = np.zeros((len(eid), 3), dtype=float)
        for i in range(len(eid)):
            coords = np.array([xyz_arr[nid_to_idx[nid]] for nid in g[i] if nid in nid_to_idx])
            if len(coords):
                cent[i] = coords.mean(axis=0)
        return eid, pid, cent

    q4_eid, q4_pid, q4_cent = _centroids(cquad4, 4)
    t3_eid, t3_pid, t3_cent = _centroids(ctria3, 3)
    q4_lookup = {eid: (pid, cent) for eid, pid, cent in zip(q4_eid, q4_pid, q4_cent)}
    t3_lookup = {eid: (pid, cent) for eid, pid, cent in zip(t3_eid, t3_pid, t3_cent)}

    # --- assemble result arrays ---
    all_eid, all_pid, all_cent, all_type = [], [], [], []
    all_x1, all_y1, all_xy1 = [], [], []
    all_x2, all_y2, all_xy2 = [], [], []

    for s in stress_q4:
        eid = int(s["EID"])
        pid, cent = q4_lookup.get(eid, (0, np.zeros(3)))
        all_eid.append(eid); all_pid.append(pid); all_cent.append(cent)
        all_type.append("QUAD4")
        all_x1.append(float(s["X1"]));  all_y1.append(float(s["Y1"]));  all_xy1.append(float(s["XY1"]))
        all_x2.append(float(s["X2"]));  all_y2.append(float(s["Y2"]));  all_xy2.append(float(s["XY2"]))

    for s in stress_t3:
        eid = int(s["EID"])
        pid, cent = t3_lookup.get(eid, (0, np.zeros(3)))
        all_eid.append(eid); all_pid.append(pid); all_cent.append(cent)
        all_type.append("TRIA3")
        all_x1.append(float(s["X1"]));  all_y1.append(float(s["Y1"]));  all_xy1.append(float(s["TXY1"]))
        all_x2.append(float(s["X2"]));  all_y2.append(float(s["Y2"]));  all_xy2.append(float(s["TXY2"]))

    eid_arr   = np.array(all_eid,  dtype=int)
    pid_arr   = np.array(all_pid,  dtype=int)
    cent_arr  = np.array(all_cent, dtype=float)
    type_arr  = np.array(all_type, dtype=object)
    xy1_arr   = np.array(all_xy1,  dtype=float)
    xy2_arr   = np.array(all_xy2,  dtype=float)

    return dict(
        eid        = eid_arr,
        centroid   = cent_arr,
        pid        = pid_arr,
        elem_type  = type_arr,
        sigma_x1   = np.array(all_x1,  dtype=float),
        sigma_y1   = np.array(all_y1,  dtype=float),
        tau_xy1    = xy1_arr,
        sigma_x2   = np.array(all_x2,  dtype=float),
        sigma_y2   = np.array(all_y2,  dtype=float),
        tau_xy2    = xy2_arr,
        tau_xy_avg = 0.5 * (np.abs(xy1_arr) + np.abs(xy2_arr)),
    )


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


def read_hdf5_rbe_node_ids(hdf5_path: Path) -> np.ndarray:
    """
    Extract GRID IDs referenced by RBE2/RBE3 rigid/interpolation elements.

    These are connector nodes (master/independent + dependent/weighted grids),
    not part of any shell or bar property, so they carry no PID.  Used to
    classify the nodes that fall outside the PID subdomains.

    Layout in the Epilysis H5 (both stored as groups):
      RBE2: RB.GN  (independent grid) + GM.ID  (dependent grids)
      RBE3: IDENTITY.REFG (reference grid) + G.ID (weighted grids)

    Returns sorted unique (nRBE,) int array; empty if no RBE elements present.
    """
    ids: set[int] = set()
    with h5py.File(hdf5_path, "r") as f:
        rbe2 = f.get("EPILYSIS/INPUT/ELEMENT/RBE2")
        if rbe2 is not None:
            if "RB" in rbe2:
                ids.update(int(v) for v in np.asarray(rbe2["RB"]["GN"], dtype=int))
            if "GM" in rbe2:
                ids.update(int(v) for v in np.asarray(rbe2["GM"]["ID"], dtype=int))
        rbe3 = f.get("EPILYSIS/INPUT/ELEMENT/RBE3")
        if rbe3 is not None:
            if "IDENTITY" in rbe3:
                ids.update(int(v) for v in np.asarray(rbe3["IDENTITY"]["REFG"], dtype=int))
            if "G" in rbe3:
                ids.update(int(v) for v in np.asarray(rbe3["G"]["ID"], dtype=int))
    return np.array(sorted(ids), dtype=int)


def read_hdf5_pid_subdomains(hdf5_path: Path,
                             element_types=("CQUAD4", "CTRIA3", "CBAR")) -> dict:
    """
    Build subdomains keyed by property ID (PID) directly from the element
    connectivity stored in an Epilysis H5 file.

    Each structural element carries a PID and its corner GRID IDs (field ``G``).
    Grouping the GRID IDs by PID gives one zone per property — the same
    PID->GRID mapping that ``export_biw_subdomains.py`` produces in META, but
    read straight from the H5 so it works for BIW and TB without a separate
    JSON export.  The structural mesh is identical across variants (TB is BIW
    plus lumped masses), so the zones match.

    Parameters
    ----------
    hdf5_path     : path to the Epilysis modal/static H5 file
    element_types : element groups under EPILYSIS/INPUT/ELEMENT to scan
                    (shells by default; CBAR included so beam-only properties
                    are not dropped — absent groups are silently skipped)

    Returns
    -------
    dict  {f"pid_{PID}": sorted list of unique GRID IDs}
        Zones are ordered by ascending PID.  Empty if no elements are found.
    """
    pid_to_grids: dict[int, set] = {}
    with h5py.File(hdf5_path, "r") as f:
        for et in element_types:
            path = f"EPILYSIS/INPUT/ELEMENT/{et}"
            if path not in f:
                continue
            elems  = f[path][:]
            fields = elems.dtype.names
            pids   = np.asarray(elems["PID"], dtype=int)

            # Connectivity field differs by element: shells store all corners in
            # ``G`` (nElem, nCorners); CBAR stores its two ends in ``GA``/``GB``.
            if "G" in fields:
                node_cols = [np.asarray(elems["G"], dtype=int)]
            else:
                node_cols = [np.asarray(elems[c], dtype=int)[:, None]
                             for c in ("GA", "GB") if c in fields]
            if not node_cols:
                continue
            grids = np.hstack(node_cols)                     # (nElem, nNodesPerElem)

            for pid, g in zip(pids, grids):
                pid_to_grids.setdefault(int(pid), set()).update(int(n) for n in g if n > 0)

    return {
        f"pid_{pid}": sorted(grids)
        for pid, grids in sorted(pid_to_grids.items())
    }


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



