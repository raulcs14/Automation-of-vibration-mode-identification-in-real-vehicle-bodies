"""
DofSpace — stateful DOF reduction with persistent Nastran node-ID tracking.

The object holds all analysis arrays (modes, refs, M, K, R, node_ids,
node_xyz) and a `dof_node_ids` vector that maps every current DOF row to
its original Nastran GRID ID.  That mapping is updated on every reduction,
so masks are always built from node IDs — never from positional indices.

Usage
-----
    space = DofSpace(modes, refs, M, K, R, node_ids, node_xyz)

    # Remove CONM2 nodes by their Nastran IDs
    space.remove_nodes(conm2_node_ids)

    # Remove rotational DOFs (keep only Ux Uy Uz)
    space.keep_dof_components([0, 1, 2])   # 0=Ux 1=Uy 2=Uz 3=Rx 4=Ry 5=Rz

    # Check current dimensions
    print(space.n_dof, space.n_nodes)

    # Unpack for downstream code
    modes, refs, M, K, R = space.modes, space.refs, space.M, space.K, space.R
"""

import numpy as np
import scipy.sparse as sp


class DofSpace:
    """
    Container for all DOF-aligned analysis arrays with built-in reduction.

    Every DOF row is tagged with its original Nastran GRID ID via
    `dof_node_ids`.  Reductions are expressed in terms of node IDs or DOF
    components so the caller never needs to track positional indices manually.

    Parameters
    ----------
    modes     : (nDOF, nModes)
    refs      : (nDOF, nRefs) or (nDOF,)
    M, K      : (nDOF, nDOF) sparse or dense
    R         : (nDOF, 6)
    node_ids  : (nNodes,)   Nastran GRID IDs in current DOF order
    node_xyz  : (nNodes, 3) coordinates in same order
    dofs_per_node : int     6 for full 6-DOF, 3 for translational-only, etc.
    """

    def __init__(self, modes, refs, M, K, R, node_ids, node_xyz,
                 dofs_per_node: int = 6):
        self.modes  = modes
        self.refs   = refs
        self.M      = M
        self.K      = K
        self.R      = R
        self.node_ids  = np.asarray(node_ids, dtype=int)
        self.node_xyz  = np.asarray(node_xyz, dtype=float)
        self.dofs_per_node = dofs_per_node

        # dof_node_ids[i] = Nastran GRID ID of DOF row i
        self.dof_node_ids = np.repeat(self.node_ids, dofs_per_node)

        self._validate()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_dof(self) -> int:
        return self.modes.shape[0]

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    # ------------------------------------------------------------------
    # Public reduction API
    # ------------------------------------------------------------------

    def remove_nodes(self, node_ids_to_remove) -> "DofSpace":
        """
        Remove all DOFs belonging to the given Nastran GRID IDs.

        node_ids_to_remove : array-like of int — original Nastran IDs.
        Works regardless of how many reductions have already been applied.
        """
        node_ids_to_remove = np.asarray(node_ids_to_remove, dtype=int)
        keep_mask = ~np.isin(self.dof_node_ids, node_ids_to_remove)
        n_removed = int(np.sum(~keep_mask))
        self._apply_dof_mask(keep_mask)
        print(f"  [DofSpace] remove_nodes: {len(node_ids_to_remove)} nodes"
              f" -> {n_removed} DOFs removed  ({self.n_dof} DOFs remaining)")
        return self

    def keep_nodes(self, node_ids_to_keep) -> "DofSpace":
        """Keep only DOFs belonging to the given Nastran GRID IDs."""
        node_ids_to_keep = np.asarray(node_ids_to_keep, dtype=int)
        keep_mask = np.isin(self.dof_node_ids, node_ids_to_keep)
        n_removed = int(np.sum(~keep_mask))
        self._apply_dof_mask(keep_mask)
        print(f"  [DofSpace] keep_nodes: {len(node_ids_to_keep)} nodes requested"
              f" -> {n_removed} DOFs removed  ({self.n_dof} DOFs remaining)")
        return self

    def keep_dof_components(self, components) -> "DofSpace":
        """
        Keep only specific DOF components per node.

        components : list of int, subset of range(dofs_per_node)
                     e.g. [0,1,2] for Ux/Uy/Uz in a 6-DOF layout.

        After this call dofs_per_node is updated to len(components).
        node_ids and node_xyz are unchanged (same nodes, fewer DOFs each).
        """
        dpn = self.dofs_per_node
        components = sorted(set(int(c) for c in components))
        if any(c >= dpn or c < 0 for c in components):
            raise ValueError(
                f"components must be in 0..{dpn-1}, got {components}"
            )
        # Build a per-DOF boolean mask: True for DOFs whose intra-node offset
        # is in `components`
        offsets   = np.tile(np.arange(dpn), self.n_nodes)
        keep_mask = np.isin(offsets, components)

        n_removed = int(np.sum(~keep_mask))
        # node_xyz / node_ids don't change — same nodes, fewer DOFs
        self._apply_dof_mask(keep_mask, update_nodes=False)
        self.dofs_per_node = len(components)
        # Rebuild dof_node_ids with new dofs_per_node
        self.dof_node_ids = np.repeat(self.node_ids, self.dofs_per_node)
        print(f"  [DofSpace] keep_dof_components {components}: "
              f"{n_removed} DOFs removed  ({self.n_dof} DOFs remaining)")
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate(self):
        n = self.modes.shape[0]
        for name, arr in [("refs", self.refs), ("R", self.R)]:
            if arr.shape[0] != n:
                raise ValueError(f"{name} has {arr.shape[0]} rows, expected {n}")
        for name, mat in [("M", self.M), ("K", self.K)]:
            if mat.shape != (n, n):
                raise ValueError(f"{name} has shape {mat.shape}, expected ({n},{n})")
        if len(self.dof_node_ids) != n:
            raise ValueError(f"dof_node_ids length {len(self.dof_node_ids)} != {n}")

    def _apply_dof_mask(self, mask: np.ndarray, update_nodes: bool = True):
        """Apply a boolean DOF-level keep-mask to all arrays."""
        self._check_mask(mask)
        idx = np.where(mask)[0]

        self.modes = self.modes[idx, :]
        self.refs  = self.refs[idx, :] if self.refs.ndim == 2 else self.refs[idx]
        self.R     = self.R[idx, :]

        if sp.issparse(self.M):
            self.M = self.M[idx, :][:, idx]
            self.K = self.K[idx, :][:, idx]
        else:
            self.M = self.M[np.ix_(idx, idx)]
            self.K = self.K[np.ix_(idx, idx)]

        self.dof_node_ids = self.dof_node_ids[idx]

        if update_nodes:
            node_mask      = np.isin(self.node_ids, self.dof_node_ids)
            self.node_ids  = self.node_ids[node_mask]
            self.node_xyz  = self.node_xyz[node_mask, :]

    def _check_mask(self, mask: np.ndarray):
        if mask.shape != (self.n_dof,):
            raise ValueError(
                f"Mask size {mask.shape[0]} does not match current DOF space "
                f"({self.n_dof} DOFs). "
                f"Build masks from space.dof_node_ids or space.node_ids, "
                f"not from original indices."
            )
        if mask.dtype != bool:
            raise TypeError(f"Mask must be boolean, got {mask.dtype}")
