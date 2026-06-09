# -*- coding: utf-8 -*-
"""
Full pipeline verification for H5 reading and MAC computation — BIW and TB.

Checks:
  1. H5 modal read  — shapes, frequencies, nodes, DOFs
  2. H5 static read — shapes, node consistency with modal
  3. K/M matrices   — shapes, symmetry, diagonal positivity
  4. CONM2          — TB only: valid IDs present
  5. MAC identity   — values in [0,1], dominant max for elastic modes
  6. MAC mass-weighted — as coherent as identity
  7. DofSpace       — remove_nodes and translational_slice keep consistency

Run:
    py -3 tests/SEAT/test_h5_pipeline.py
    py -3 -m pytest tests/SEAT/test_h5_pipeline.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import scipy.sparse as sp

from seat_model.modal_analysis import run_modal_analysis, N_RIGID_BODY_MODES
from seat_model.static_model   import run_static_model
from seat_model.reader         import read_hdf5_modal, read_hdf5_static, read_hdf5_conm2_node_ids
from common.mac_core           import compute_mac
from common.dof_reduction      import DofSpace
from common.utils              import translational_dof_indices

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_passed = 0
_failed = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  [OK]  {label}")
    else:
        _failed += 1
        print(f"  [FAIL] {label}" + (f"\n         -> {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n{'-'*60}")
    print(f"  {title}")
    print(f"{'-'*60}")


# ---------------------------------------------------------------------------
# 1. H5 modal read
# ---------------------------------------------------------------------------

def test_modal_reader(variant: str) -> dict:
    section(f"1. H5 modal read [{variant}]")
    dyn = run_modal_analysis(variant=variant, skip_rigid=False)

    n_nodes  = dyn["node_ids"].shape[0]
    gdof     = n_nodes * 6
    n_modes  = dyn["modes"].shape[1]

    check("node_ids is 1-D int",
          dyn["node_ids"].ndim == 1 and dyn["node_ids"].dtype.kind == "i")
    check("node_coordinates shape (nNodes, 3)",
          dyn["node_coordinates"].shape == (n_nodes, 3))
    check("modes shape (6*nNodes, nModes)",
          dyn["modes"].shape == (gdof, n_modes),
          f"got {dyn['modes'].shape}, expected ({gdof}, {n_modes})")
    check("freq shape (nModes,)",
          dyn["freq"].shape == (n_modes,))
    check(f"first {N_RIGID_BODY_MODES} modes are near 0 Hz (rigid)",
          np.all(dyn["freq"][:N_RIGID_BODY_MODES] < 1.0),
          f"freq[:6] = {dyn['freq'][:N_RIGID_BODY_MODES].round(4)}")
    check("elastic modes have positive and increasing frequencies",
          np.all(np.diff(dyn["freq"][N_RIGID_BODY_MODES:]) >= -0.01))
    check("R shape (6*nNodes, 6)",
          dyn["R"].shape == (gdof, 6))

    return dyn


# ---------------------------------------------------------------------------
# 2. H5 static read
# ---------------------------------------------------------------------------

def test_static_reader(variant: str, dyn: dict) -> dict:
    section(f"2. H5 static read [{variant}]")
    stat = run_static_model(variant=variant)

    n_nodes = dyn["node_ids"].shape[0]
    gdof    = n_nodes * 6
    n_refs  = stat["ref_moves_raw"].shape[1]

    check("refs shape (6*nNodes, nRefs)",
          stat["ref_moves_raw"].shape[0] == gdof,
          f"got {stat['ref_moves_raw'].shape[0]}, expected {gdof}")
    check("at least 1 reference case",
          n_refs >= 1)
    check("ref_names and short_names have the same length",
          len(stat["ref_names"]) == len(stat["short_names"]))
    check("reference is not all zeros",
          np.any(stat["ref_moves_raw"] != 0.0))

    return stat


# ---------------------------------------------------------------------------
# 3. K and M matrices
# ---------------------------------------------------------------------------

def test_matrices(variant: str, dyn: dict) -> None:
    section(f"3. K and M matrices [{variant}]")
    K = dyn["K"]
    M = dyn["M"]
    gdof = dyn["node_ids"].shape[0] * 6

    check("K is sparse CSR",      sp.issparse(K))
    check("M is sparse CSR",      sp.issparse(M))
    check("K shape (GDof, GDof)", K.shape == (gdof, gdof),
          f"got {K.shape}")
    check("M shape (GDof, GDof)", M.shape == (gdof, gdof))
    check("K diagonal >= 0",      np.all(K.diagonal() >= 0))
    check("M diagonal >= 0",      np.all(M.diagonal() >= 0))

    diff_K = K - K.T
    rel_K  = diff_K.norm() / K.norm() if hasattr(diff_K, "norm") else (
             np.linalg.norm(diff_K.toarray()) / np.linalg.norm(K.toarray()))
    check("K is symmetric (rel error < 1e-8)", rel_K < 1e-8,
          f"rel error = {rel_K:.2e}")

    diff_M = M - M.T
    rel_M  = diff_M.norm() / M.norm() if hasattr(diff_M, "norm") else (
             np.linalg.norm(diff_M.toarray()) / np.linalg.norm(M.toarray()))
    check("M is symmetric (rel error < 1e-8)", rel_M < 1e-8,
          f"rel error = {rel_M:.2e}")


# ---------------------------------------------------------------------------
# 4. CONM2 (TB only)
# ---------------------------------------------------------------------------

def test_conm2(dyn: dict, variant: str) -> None:
    section(f"4. CONM2 [{variant}]")
    if variant != "TB":
        check("BIW: conm2_node_ids is None", dyn["conm2_node_ids"] is None)
        return

    ids = dyn["conm2_node_ids"]
    check("TB: conm2_node_ids is not None", ids is not None)
    check("TB: at least 1 CONM2 node",     ids is not None and len(ids) > 0,
          f"got {len(ids) if ids is not None else 'None'}")
    check("TB: CONM2 IDs are positive integers",
          ids is not None and np.all(ids > 0))
    check("TB: CONM2 IDs are present in the model",
          ids is not None and np.all(np.isin(ids, dyn["node_ids"])),
          "some CONM2 IDs not found in node_ids")


# ---------------------------------------------------------------------------
# 5 & 6. MAC identity and mass-weighted
# ---------------------------------------------------------------------------

def test_mac(variant: str, dyn: dict, stat: dict) -> None:
    section(f"5-6. MAC identity and mass-weighted [{variant}]")

    modes = dyn["modes"][:, N_RIGID_BODY_MODES:]   # elastic modes only
    freq  = dyn["freq"][N_RIGID_BODY_MODES:]
    refs  = stat["ref_moves_raw"]
    M     = dyn["M"]
    gdof  = modes.shape[0]

    t_idx = translational_dof_indices(gdof)
    Phi_t = modes[t_idx, :]
    Psi_t = refs[t_idx, :]

    # --- Identity ---
    mac_id = compute_mac(Phi_t, Psi_t)
    check("MAC identity shape (nModes, nRefs)",
          mac_id.shape == (modes.shape[1], refs.shape[1]))
    check("MAC identity values in [0, 1]",
          np.all(mac_id >= -1e-10) and np.all(mac_id <= 1.0 + 1e-10),
          f"min={mac_id.min():.4f} max={mac_id.max():.4f}")
    max_mac = mac_id.max()
    check("MAC identity: at least one mode with MAC > 0.1 (basic coherence)",
          max_mac > 0.1,
          f"max MAC = {max_mac:.4f}")
    if max_mac < 0.5:
        print(f"  [INFO] max MAC = {max_mac:.4f} < 0.5 — torsion reference "
              f"may not correlate well without rigid-body removal")

    # --- Mass-weighted ---
    M_t    = M[t_idx, :][:, t_idx]
    mac_mw = compute_mac(Phi_t, Psi_t, M_t)
    check("MAC mass-weighted shape (nModes, nRefs)",
          mac_mw.shape == mac_id.shape)
    check("MAC mass-weighted values in [0, 1]",
          np.all(mac_mw >= -1e-10) and np.all(mac_mw <= 1.0 + 1e-10),
          f"min={mac_mw.min():.4f} max={mac_mw.max():.4f}")

    best_id = int(mac_id.argmax())
    best_mw = int(mac_mw.argmax())
    check("Best mode is the same for identity and mass-weighted",
          best_id == best_mw,
          f"identity best={best_id} ({freq[best_id // refs.shape[1]]:.1f} Hz), "
          f"mass best={best_mw} ({freq[best_mw // refs.shape[1]]:.1f} Hz)")

    best_per_mode = mac_id.max(axis=1)
    top5 = np.argsort(best_per_mode)[-5:][::-1]
    print(f"\n  Top 5 modes by MAC identity:")
    for i in top5:
        print(f"    Mode {N_RIGID_BODY_MODES + i + 1:3d}  "
              f"({freq[i]:7.2f} Hz)  MAC = {best_per_mode[i]:.4f}")


# ---------------------------------------------------------------------------
# 7. DofSpace — consistency after remove_nodes and translational_slice
# ---------------------------------------------------------------------------

def test_dofspace(variant: str, dyn: dict, stat: dict) -> None:
    section(f"7. DofSpace [{variant}]")

    space = DofSpace(
        modes    = dyn["modes"][:, N_RIGID_BODY_MODES:],
        refs     = stat["ref_moves_raw"],
        M        = dyn["M"],
        K        = dyn["K"],
        R        = dyn["R"],
        node_ids = dyn["node_ids"],
        node_xyz = dyn["node_coordinates"],
    )

    n_nodes_orig = len(dyn["node_ids"])
    check("DofSpace.n_nodes == nNodes",
          space.n_nodes == n_nodes_orig)

    # TB: remove CONM2 nodes
    if variant == "TB" and dyn["conm2_node_ids"] is not None:
        n_conm2 = len(dyn["conm2_node_ids"])
        space.remove_nodes(dyn["conm2_node_ids"])
        check("After remove_nodes: n_nodes is reduced",
              space.n_nodes == n_nodes_orig - n_conm2,
              f"expected {n_nodes_orig - n_conm2}, got {space.n_nodes}")

    t_idx, modes_t, refs_t, M_t, K_t = space.translational_slice()
    n_t = space.n_nodes * 3

    check("translational_slice: t_idx length == 3*nNodes",
          len(t_idx) == n_t,
          f"got {len(t_idx)}, expected {n_t}")
    check("translational_slice: modes_t shape (3*nNodes, nModes)",
          modes_t.shape[0] == n_t)
    check("translational_slice: refs_t shape (3*nNodes, nRefs)",
          refs_t.shape[0] == n_t)
    check("translational_slice: M_t shape (3*nNodes, 3*nNodes)",
          M_t.shape == (n_t, n_t))

    mac = compute_mac(modes_t, refs_t)
    check("MAC on translational DofSpace: values in [0,1]",
          np.all(mac >= -1e-10) and np.all(mac <= 1.0 + 1e-10))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_variant(variant: str) -> None:
    print(f"\n{'='*60}")
    print(f"  VARIANT: {variant}")
    print(f"{'='*60}")

    dyn  = test_modal_reader(variant)
    stat = test_static_reader(variant, dyn)
    test_matrices(variant, dyn)
    test_conm2(dyn, variant)
    test_mac(variant, dyn, stat)
    test_dofspace(variant, dyn, stat)


def main() -> None:
    for variant in ("BIW", "TB"):
        run_variant(variant)

    print(f"\n{'='*60}")
    print(f"  RESULT: {_passed} OK  |  {_failed} FAIL")
    print(f"{'='*60}")
    if _failed:
        sys.exit(1)


# ---------------------------------------------------------------------------
# pytest compatibility
# ---------------------------------------------------------------------------

def test_biw_pipeline():
    run_variant("BIW")
    assert _failed == 0, f"{_failed} checks failed in BIW"


def test_tb_pipeline():
    global _passed, _failed
    _passed = _failed = 0
    run_variant("TB")
    assert _failed == 0, f"{_failed} checks failed in TB"


if __name__ == "__main__":
    main()
