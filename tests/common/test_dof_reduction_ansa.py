# -*- coding: utf-8 -*-
"""
Mathematical correctness checks for the ANSA DOF-reduction pipeline.

Run directly:
    py -3 tests/common/test_dof_reduction_ansa.py

Also works with pytest:
    py -3 -m pytest tests/common/test_dof_reduction_ansa.py -v

Uses synthetic data — no real Nastran files required.
"""

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.dof_reduction import DofSpace
from common.utils import translational_dof_indices
from common.subdomain import average_zones
from seat_model.subdomains import grid_ids_to_node_indices
from seat_model.reader import aset_dof_mask_from_gset


# ---------------------------------------------------------------------------
# Runner infrastructure
# ---------------------------------------------------------------------------

_passed = 0
_failed = 0


def check(label, condition, detail=""):
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  [OK]  {label}")
    else:
        _failed += 1
        msg = f"  [FAIL] {label}"
        if detail:
            msg += f"\n         -> {detail}"
        print(msg)


def section(title):
    print(f"\n{'-'*60}")
    print(f"  {title}")
    print(f"{'-'*60}")


# ---------------------------------------------------------------------------
# Helper: synthetic DofSpace
# ---------------------------------------------------------------------------

def _make_space(n_nodes, n_modes=4, n_refs=2, dofs_per_node=6, seed=0):
    """
    DofSpace with node_ids = [1001 .. 1000+n_nodes] (Nastran-style IDs, not 0-based).
    M and K are sparse identity matrices (positive definite, self-consistent).
    """
    rng = np.random.default_rng(seed)
    node_ids = np.arange(1001, 1001 + n_nodes, dtype=int)
    node_xyz = np.column_stack([
        np.arange(n_nodes, dtype=float),
        np.arange(n_nodes, dtype=float) * 2,
        np.arange(n_nodes, dtype=float) * 3,
    ])
    n_dof = n_nodes * dofs_per_node
    modes = rng.standard_normal((n_dof, n_modes))
    refs  = rng.standard_normal((n_dof, n_refs))
    R     = rng.standard_normal((n_dof, 6))
    A     = rng.standard_normal((n_dof, n_dof))
    M = sp.csr_matrix(A @ A.T + np.eye(n_dof) * 0.1)
    K = sp.csr_matrix(A @ A.T + np.eye(n_dof) * 0.5)
    return DofSpace(modes, refs, M, K, R, node_ids, node_xyz,
                    dofs_per_node=dofs_per_node)


# ---------------------------------------------------------------------------
# Block 1 -- remove_nodes removes exactly the requested GRID IDs
# ---------------------------------------------------------------------------

def test_remove_nodes():
    section("1. remove_nodes -- node removal by GRID ID")

    # 1a: correct node and DOF count after removal
    space = _make_space(10)
    ids_rm = np.array([1001, 1003, 1005])
    print(f"\n  Initial space: {space.n_nodes} nodes, {space.n_dof} DOFs")
    print(f"  Removing GRID IDs: {ids_rm.tolist()}")
    space.remove_nodes(ids_rm)
    print(f"  Resulting space: {space.n_nodes} nodes, {space.n_dof} DOFs")
    check("3 nodes removed -> 7 nodes remaining", space.n_nodes == 7,
          f"n_nodes={space.n_nodes}")
    check("3 nodes x 6 DOFs removed -> 42 DOFs remaining", space.n_dof == 42,
          f"n_dof={space.n_dof}")

    # 1b: removed IDs must not appear in dof_node_ids
    for nid in ids_rm:
        present = nid in space.dof_node_ids
        check(f"GRID {nid} absent from dof_node_ids after removal", not present,
              f"GRID {nid} still present")

    # 1c: non-removed IDs must still be present
    kept = [1002, 1004, 1006, 1007, 1008, 1009, 1010]
    all_kept = all(nid in space.dof_node_ids for nid in kept)
    check(f"The {len(kept)} remaining nodes are still in dof_node_ids", all_kept)

    # 1d: each node appears exactly 6 times in dof_node_ids
    counts = {int(nid): int(np.sum(space.dof_node_ids == nid)) for nid in space.node_ids}
    all_six = all(v == 6 for v in counts.values())
    wrong = [(k, v) for k, v in counts.items() if v != 6]
    print(f"\n  Occurrences per GRID ID in dof_node_ids (expected 6): "
          f"{list(counts.values())}")
    check("Each node appears exactly 6 times in dof_node_ids", all_six,
          f"Nodes with wrong count: {wrong}")

    # 1e: matrices are square and match new size
    n = space.n_dof
    check(f"M.shape == ({n},{n}) after removal", space.M.shape == (n, n),
          f"M.shape={space.M.shape}")
    check(f"K.shape == ({n},{n}) after removal", space.K.shape == (n, n),
          f"K.shape={space.K.shape}")
    check("modes.shape[0] == n_dof after removal",
          space.modes.shape[0] == n, f"modes.shape={space.modes.shape}")

    # 1f: node_xyz is updated
    space2 = _make_space(5)
    print(f"\n  Removing GRID 1003 from a 5-node space")
    space2.remove_nodes([1003])
    check("node_xyz.shape == (4, 3) after removing 1 node",
          space2.node_xyz.shape == (4, 3), f"shape={space2.node_xyz.shape}")
    check("GRID 1003 not in node_ids", 1003 not in space2.node_ids)

    # 1g: unknown IDs cause no change
    space3 = _make_space(5)
    dof_before = space3.n_dof
    print(f"\n  Attempting to remove non-existent GRIDs [9999, 8888]")
    space3.remove_nodes([9999, 8888])
    check("Non-existent IDs: n_dof unchanged", space3.n_dof == dof_before,
          f"n_dof before={dof_before}, after={space3.n_dof}")

    # 1h: len(dof_node_ids) == n_dof always
    space4 = _make_space(10)
    space4.remove_nodes([1001, 1005, 1010])
    check("len(dof_node_ids) == n_dof after multiple removals",
          len(space4.dof_node_ids) == space4.n_dof,
          f"len={len(space4.dof_node_ids)} vs n_dof={space4.n_dof}")


# ---------------------------------------------------------------------------
# Block 2 -- translational_dof_indices: blocked layout [Ux|Uy|Uz]
# ---------------------------------------------------------------------------

def test_translational_dof_indices():
    section("2. translational_dof_indices -- blocked layout [Ux_all | Uy_all | Uz_all]")

    n_nodes = 5
    gdof = n_nodes * 6
    t_idx = translational_dof_indices(gdof)

    print(f"\n  n_nodes={n_nodes}, gdof={gdof}")
    print(f"  t_idx = {t_idx.tolist()}")
    print(f"  Ux block (first {n_nodes}): {t_idx[:n_nodes].tolist()}")
    print(f"  Uy block (next  {n_nodes}): {t_idx[n_nodes:2*n_nodes].tolist()}")
    print(f"  Uz block (last  {n_nodes}): {t_idx[2*n_nodes:].tolist()}")

    check(f"Length = {n_nodes}x3 = {n_nodes*3}", len(t_idx) == n_nodes * 3,
          f"len={len(t_idx)}")

    ux_expected = np.arange(0, gdof, 6)
    check("Ux block = [0, 6, 12, ...]",
          np.array_equal(t_idx[:n_nodes], ux_expected),
          f"got={t_idx[:n_nodes].tolist()}, expected={ux_expected.tolist()}")

    uy_expected = np.arange(1, gdof, 6)
    check("Uy block = [1, 7, 13, ...]",
          np.array_equal(t_idx[n_nodes:2*n_nodes], uy_expected),
          f"got={t_idx[n_nodes:2*n_nodes].tolist()}, expected={uy_expected.tolist()}")

    uz_expected = np.arange(2, gdof, 6)
    check("Uz block = [2, 8, 14, ...]",
          np.array_equal(t_idx[2*n_nodes:], uz_expected),
          f"got={t_idx[2*n_nodes:].tolist()}, expected={uz_expected.tolist()}")

    rotational = {3, 4, 5, 9, 10, 11, 15, 16, 17, 21, 22, 23}
    overlap = set(t_idx.tolist()) & rotational
    check("No rotational index (Rx/Ry/Rz) selected", len(overlap) == 0,
          f"rotational indices in t_idx: {overlap}")

    # Exact case with 2 nodes: expect [0, 6, 1, 7, 2, 8]
    t2 = translational_dof_indices(12)
    expected_2 = np.array([0, 6, 1, 7, 2, 8])
    print(f"\n  For 2 nodes (gdof=12): t_idx={t2.tolist()}, expected={expected_2.tolist()}")
    check("2-node case: t_idx == [0, 6, 1, 7, 2, 8]",
          np.array_equal(t2, expected_2), f"got={t2.tolist()}")


# ---------------------------------------------------------------------------
# Block 3 -- average_zones receives correct slice after remove_nodes + t_idx
# ---------------------------------------------------------------------------

def test_average_zones_slice():
    section("3. average_zones -- correct slice after remove_nodes + t_idx")

    # Deterministic modes: mode k, node i, component c -> value = (i+1)*10 + c + k*100
    # Ux node 0 mode 0 = 10,  Ux node 1 mode 0 = 20,  Ux node 2 mode 0 = 30
    n_nodes = 5
    n_modes = 2
    n_dof   = n_nodes * 6
    modes   = np.zeros((n_dof, n_modes))
    for i in range(n_nodes):
        for c in range(6):
            row = i * 6 + c
            for k in range(n_modes):
                modes[row, k] = (i + 1) * 10.0 + c + k * 100.0

    rng  = np.random.default_rng(42)
    refs = rng.standard_normal((n_dof, 1))
    R    = rng.standard_normal((n_dof, 6))
    I    = sp.csr_matrix(np.eye(n_dof))
    node_ids = np.arange(1001, 1001 + n_nodes, dtype=int)
    space = DofSpace(modes, refs, I, I, R, node_ids, np.zeros((n_nodes, 3)))

    t_idx   = translational_dof_indices(space.n_dof)
    modes_t = space.modes[t_idx, :]

    # Zone = positional nodes 0, 1, 2
    subdomains = {"zone_A": [0, 1, 2]}
    result = average_zones(modes_t, subdomains, space.n_nodes)

    ux_node0 = modes[0,  0]   # = 10
    ux_node1 = modes[6,  0]   # = 20
    ux_node2 = modes[12, 0]   # = 30
    expected = (ux_node0 + ux_node1 + ux_node2) / 3.0
    obtained = result[0, 0]

    print(f"\n  Deterministic modes: Ux node0={ux_node0}, node1={ux_node1}, node2={ux_node2}")
    print(f"  Expected Ux zone mean (mode 0) = {expected:.4f}")
    print(f"  average_zones returns Ux zone  = {obtained:.4f}")
    check("average_zones(Ux zone) == arithmetic mean of Ux for the nodes",
          np.isclose(obtained, expected),
          f"got={obtained:.6f}, expected={expected:.6f}")

    # Correct shape: 3 zones x 3 DOFs, 3 modes
    n_nodes2 = 10
    n_modes2 = 3
    n_dof2   = n_nodes2 * 6
    modes2   = rng.standard_normal((n_dof2, n_modes2))
    refs2    = rng.standard_normal((n_dof2, 1))
    I2       = sp.csr_matrix(np.eye(n_dof2))
    node_ids2 = np.arange(1001, 1001 + n_nodes2, dtype=int)
    space2 = DofSpace(modes2, refs2, I2, I2, rng.standard_normal((n_dof2, 6)),
                      node_ids2, np.zeros((n_nodes2, 3)))
    t_idx2   = translational_dof_indices(space2.n_dof)
    modes_t2 = space2.modes[t_idx2, :]
    subs2 = {"z1": [0,1,2], "z2": [3,4,5], "z3": [6,7,8,9]}
    r2 = average_zones(modes_t2, subs2, space2.n_nodes)
    print(f"\n  3 zones, {n_modes2} modes -> expected shape (9, 3), got {r2.shape}")
    check(f"average_zones shape == (9, {n_modes2}) with 3 zones",
          r2.shape == (9, n_modes2), f"got={r2.shape}")

    # After remove_nodes, t_idx of the reduced space covers 3*n_nodes DOFs
    space3 = _make_space(8)
    space3.remove_nodes([1003])
    t3 = translational_dof_indices(space3.n_dof)
    print(f"\n  After remove_nodes([1003]): n_nodes={space3.n_nodes}, "
          f"n_dof={space3.n_dof}, len(t_idx)={len(t3)}")
    check("len(t_idx) == n_nodes*3 after remove_nodes",
          len(t3) == space3.n_nodes * 3,
          f"len(t3)={len(t3)}, n_nodes*3={space3.n_nodes*3}")
    check("t_idx.max() < n_dof (all indices within space)",
          t3.max() < space3.n_dof,
          f"t3.max()={t3.max()}, n_dof={space3.n_dof}")


# ---------------------------------------------------------------------------
# Block 4 -- Subdomain positional indices remain valid after remove_nodes
# ---------------------------------------------------------------------------

def test_subdomain_indices_after_removal():
    section("4. Subdomains -- valid positional indices after remove_nodes")

    space = _make_space(10)
    print(f"\n  Initial node_ids: {space.node_ids.tolist()}")
    space.remove_nodes([1005])
    print(f"  node_ids after removing 1005: {space.node_ids.tolist()}")

    # Fictitious JSON that includes the removed node in a zone
    subdomains_grid = {
        "front": [1001, 1002, 1003],
        "mid":   [1004, 1005, 1006],   # 1005 no longer exists, ignored
        "rear":  [1008, 1009, 1010],
    }
    subdomains_pos = grid_ids_to_node_indices(subdomains_grid, space.node_ids)

    print(f"\n  Zones after conversion (1005 removed from 'mid'):")
    for name, idxs in subdomains_pos.items():
        node_ids_in_zone = [space.node_ids[i] for i in idxs]
        print(f"    {name}: indices={idxs} -> GRIDs={node_ids_in_zone}")

    all_valid = all(
        0 <= idx < space.n_nodes
        for idxs in subdomains_pos.values()
        for idx in idxs
    )
    check("All positional indices are in [0, n_nodes)", all_valid)

    mid_ids = [space.node_ids[i] for i in subdomains_pos.get("mid", [])]
    check("GRID 1005 absent from 'mid' zone after removal",
          1005 not in mid_ids, f"mid contains: {mid_ids}")

    # Zone with removed node when building subdomains from GRIDs
    space2 = _make_space(10)
    space2.remove_nodes([1005])
    subs2 = {"zone": [1001, 1005, 1009]}
    pos2  = grid_ids_to_node_indices(subs2, space2.node_ids)
    ids_in_zone = [space2.node_ids[i] for i in pos2["zone"]]
    print(f"\n  Zone [1001, 1005, 1009] with 1005 removed -> GRIDs in zone: {ids_in_zone}")
    check("GRID 1005 not in subdomain after conversion to positional indices",
          1005 not in ids_in_zone)


# ---------------------------------------------------------------------------
# Block 5 -- keep_dof_components updates dofs_per_node and layout
# ---------------------------------------------------------------------------

def test_keep_dof_components():
    section("5. keep_dof_components -- reduction to translational DOFs")

    space = _make_space(6)
    n_nodes = space.n_nodes
    print(f"\n  Initial space: {n_nodes} nodes, {space.n_dof} DOFs, "
          f"dofs_per_node={space.dofs_per_node}")
    space.keep_dof_components([0, 1, 2])
    print(f"  After keep_dof_components([0,1,2]): {space.n_dof} DOFs, "
          f"dofs_per_node={space.dofs_per_node}")

    check("dofs_per_node == 3 after keep_dof_components([0,1,2])",
          space.dofs_per_node == 3, f"got={space.dofs_per_node}")
    check(f"n_dof == {n_nodes}*3 = {n_nodes*3}",
          space.n_dof == n_nodes * 3, f"got={space.n_dof}")

    for nid in space.node_ids:
        count = int(np.sum(space.dof_node_ids == nid))
        if count != 3:
            check(f"GRID {nid} appears 3 times in dof_node_ids", False,
                  f"appears {count} times")
            break
    else:
        check("Each node appears exactly 3 times in dof_node_ids", True)

    # Out-of-range component must raise ValueError
    space2 = _make_space(3)
    try:
        space2.keep_dof_components([0, 1, 6])
        check("Component 6 (out of range) raises ValueError", False,
              "no exception was raised")
    except ValueError as e:
        print(f"\n  Expected ValueError: {e}")
        check("Component 6 (out of range) raises ValueError", True)


# ---------------------------------------------------------------------------
# Block 6 -- aset_dof_mask_from_gset (G-set -> A-set filtering)
# ---------------------------------------------------------------------------

def test_aset_dof_mask():
    section("6. aset_dof_mask_from_gset -- G-set -> A-set filtering (BIW)")

    aset = np.array([10, 30])
    gset = np.array([10, 20, 30])
    mask = aset_dof_mask_from_gset(aset, gset)

    print(f"\n  G-set: {gset.tolist()}, A-set: {aset.tolist()}")
    print(f"  Mask (1=A-set): {mask.astype(int).tolist()}")
    print(f"  Node 10 -> rows 0-5  : {mask[:6].astype(int).tolist()}")
    print(f"  Node 20 -> rows 6-11 : {mask[6:12].astype(int).tolist()}  (excluded)")
    print(f"  Node 30 -> rows 12-17: {mask[12:18].astype(int).tolist()}")

    check("Mask length = len(gset)*6",
          len(mask) == len(gset) * 6, f"len={len(mask)}")
    check("DOFs of node 10 (A-set) -> True",
          mask[:6].all(), f"mask[:6]={mask[:6].tolist()}")
    check("DOFs of node 20 (excluded) -> False",
          not mask[6:12].any(), f"mask[6:12]={mask[6:12].tolist()}")
    check("DOFs of node 30 (A-set) -> True",
          mask[12:18].all(), f"mask[12:18]={mask[12:18].tolist()}")
    check("Mask sum = len(A-set)*6",
          int(mask.sum()) == len(aset) * 6, f"sum={mask.sum()}")


# ---------------------------------------------------------------------------
# Block 7 -- Full BIW pipeline: mask -> DofSpace -> t_idx -> average_zones
# ---------------------------------------------------------------------------

def test_full_biw_pipeline():
    section("7. Full BIW pipeline: aset_mask -> DofSpace -> t_idx -> average_zones")

    n_gset   = 10
    aset_ids = np.array([1001, 1002, 1003, 1005, 1007, 1008, 1009])   # 7 nodes
    gset_ids = np.arange(1001, 1001 + n_gset, dtype=int)

    mask = aset_dof_mask_from_gset(aset_ids, gset_ids)
    excluded = gset_ids[~np.isin(gset_ids, aset_ids)].tolist()
    print(f"\n  G-set: {n_gset} nodes ({n_gset*6} DOFs)")
    print(f"  A-set: {len(aset_ids)} nodes ({int(mask.sum())} DOFs in mask)")
    print(f"  Excluded nodes (SPC/singular): {excluded}")

    rng    = np.random.default_rng(7)
    n_dof_g = n_gset * 6
    modes_g = rng.standard_normal((n_dof_g, 5))
    refs_g  = rng.standard_normal((n_dof_g, 2))
    R_g     = rng.standard_normal((n_dof_g, 6))
    I_g     = sp.csr_matrix(np.eye(n_dof_g))

    modes_a = modes_g[mask, :]
    refs_a  = refs_g[mask, :]
    R_a     = R_g[mask, :]
    M_a     = I_g[mask, :][:, mask]
    K_a     = I_g[mask, :][:, mask]

    node_xyz = rng.standard_normal((len(aset_ids), 3))
    space = DofSpace(modes_a, refs_a, M_a, K_a, R_a, aset_ids, node_xyz)

    print(f"\n  DofSpace created: {space.n_nodes} nodes, {space.n_dof} DOFs")
    check(f"DofSpace.n_nodes == {len(aset_ids)} (A-set)",
          space.n_nodes == len(aset_ids), f"got={space.n_nodes}")
    check(f"DofSpace.n_dof == {len(aset_ids)*6}",
          space.n_dof == len(aset_ids) * 6, f"got={space.n_dof}")

    t_idx = translational_dof_indices(space.n_dof)
    print(f"  t_idx: {len(t_idx)} translational DOFs ({space.n_nodes} nodes x 3)")
    check(f"len(t_idx) == {space.n_nodes*3}",
          len(t_idx) == space.n_nodes * 3, f"got={len(t_idx)}")

    modes_t = space.modes[t_idx, :]
    subdomains = {"front": [0, 1, 2], "rear": [3, 4, 5, 6]}
    result = average_zones(modes_t, subdomains, space.n_nodes)
    print(f"  average_zones with 2 zones -> shape={result.shape} (expected (6, 5))")
    check("average_zones shape == (6, 5): 2 zones x 3 DOFs, 5 modes",
          result.shape == (6, 5), f"got={result.shape}")


# ---------------------------------------------------------------------------
# Block 8 -- Full TB pipeline: DofSpace(G-set) -> remove_nodes -> t_idx -> average_zones
# ---------------------------------------------------------------------------

def test_full_tb_pipeline():
    section("8. Full TB pipeline: DofSpace(G-set) -> remove_nodes -> t_idx -> average_zones")

    n_total   = 12
    conm2_ids = np.array([1010, 1011, 1012])
    node_ids  = np.arange(1001, 1001 + n_total, dtype=int)
    node_xyz  = np.zeros((n_total, 3))

    rng   = np.random.default_rng(99)
    n_dof = n_total * 6
    modes = rng.standard_normal((n_dof, 4))
    refs  = rng.standard_normal((n_dof, 1))
    R     = rng.standard_normal((n_dof, 6))
    I     = sp.csr_matrix(np.eye(n_dof))

    print(f"\n  G-set DofSpace: {n_total} nodes, {n_dof} DOFs")
    print(f"  CONM2 to remove: {conm2_ids.tolist()}")
    space = DofSpace(modes, refs, I, I, R, node_ids, node_xyz)
    space.remove_nodes(conm2_ids)
    print(f"  After remove_nodes: {space.n_nodes} nodes, {space.n_dof} DOFs")

    n_expected = n_total - len(conm2_ids)
    check(f"n_nodes == {n_expected} after removing CONM2",
          space.n_nodes == n_expected, f"got={space.n_nodes}")
    check(f"n_dof == {n_expected * 6}",
          space.n_dof == n_expected * 6, f"got={space.n_dof}")

    t_idx   = translational_dof_indices(space.n_dof)
    modes_t = space.modes[t_idx, :]
    print(f"  t_idx: {len(t_idx)} translational DOFs (expected {space.n_nodes*3})")
    check(f"len(t_idx) == {space.n_nodes*3}",
          len(t_idx) == space.n_nodes * 3, f"got={len(t_idx)}")

    subdomains_grid = {
        "z1": [1001, 1002, 1003],
        "z2": [1004, 1005, 1006],
        "z3": [1007, 1008, 1009],
    }
    subdomains_pos = grid_ids_to_node_indices(subdomains_grid, space.node_ids)
    result = average_zones(modes_t, subdomains_pos, space.n_nodes)
    print(f"  average_zones with 3 zones -> shape={result.shape} (expected (9, 4))")
    check("average_zones shape == (9, 4): 3 zones x 3 DOFs, 4 modes",
          result.shape == (9, 4), f"got={result.shape}")

    # CONM2 nodes must not appear in any subdomain
    subdomains_all = {"all": node_ids.tolist()}
    pos_all = grid_ids_to_node_indices(subdomains_all, space.node_ids)
    ids_in_zone = [space.node_ids[i] for i in pos_all["all"]]
    conm2_in_zone = [int(nid) for nid in conm2_ids if nid in ids_in_zone]
    print(f"  CONM2 GRIDs in 'all' subdomain after conversion: {conm2_in_zone} (expected [])")
    check("No CONM2 node appears in subdomains after remove_nodes",
          len(conm2_in_zone) == 0, f"found: {conm2_in_zone}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 64)
    print("  DOF reduction verification -- ANSA pipeline")
    print("  Synthetic data, no real Nastran files")
    print("=" * 64)

    test_remove_nodes()
    test_translational_dof_indices()
    test_average_zones_slice()
    test_subdomain_indices_after_removal()
    test_keep_dof_components()
    test_aset_dof_mask()
    test_full_biw_pipeline()
    test_full_tb_pipeline()

    print(f"\n{'='*60}")
    total = _passed + _failed
    print(f"  Result: {_passed}/{total} checks passed", end="")
    if _failed:
        print(f"  ({_failed} FAILED)")
    else:
        print("  -- all correct")
    print(f"{'='*60}")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
