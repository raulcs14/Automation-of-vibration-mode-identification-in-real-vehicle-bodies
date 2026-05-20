# -*- coding: utf-8 -*-
"""
Verificacion del rigor matematico en la reduccion de DOFs del pipeline ANSA.

Ejecutar directamente:
    py -3 tests/test_dof_reduction_ansa.py

Tambien funciona con pytest:
    py -3 -m pytest tests/test_dof_reduction_ansa.py -v

Usa datos sinteticos -- no necesita ficheros Nastran reales.
"""

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
# Helper: DofSpace sintetico
# ---------------------------------------------------------------------------

def _make_space(n_nodes, n_modes=4, n_refs=2, dofs_per_node=6, seed=0):
    """
    DofSpace con node_ids = [1001 .. 1000+n_nodes] (IDs estilo Nastran, no 0-based).
    M y K son matrices identidad sparse (positivas definidas, consistentes).
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
# Bloque 1 -- remove_nodes elimina exactamente los GRID IDs pedidos
# ---------------------------------------------------------------------------

def test_remove_nodes():
    section("1. remove_nodes -- eliminacion de nodos por GRID ID")

    # 1a: cuenta correcta de nodos y DOFs eliminados
    space = _make_space(10)
    ids_rm = np.array([1001, 1003, 1005])
    print(f"\n  Espacio inicial: {space.n_nodes} nodos, {space.n_dof} DOFs")
    print(f"  Eliminando GRID IDs: {ids_rm.tolist()}")
    space.remove_nodes(ids_rm)
    print(f"  Espacio resultante: {space.n_nodes} nodos, {space.n_dof} DOFs")
    check("3 nodos eliminados -> 7 nodos restantes", space.n_nodes == 7,
          f"n_nodes={space.n_nodes}")
    check("3 nodos x 6 DOFs eliminados -> 42 DOFs restantes", space.n_dof == 42,
          f"n_dof={space.n_dof}")

    # 1b: los IDs eliminados no aparecen en dof_node_ids
    for nid in ids_rm:
        present = nid in space.dof_node_ids
        check(f"GRID {nid} ausente de dof_node_ids tras eliminacion", not present,
              f"GRID {nid} sigue presente")

    # 1c: los IDs no eliminados si aparecen
    kept = [1002, 1004, 1006, 1007, 1008, 1009, 1010]
    all_kept = all(nid in space.dof_node_ids for nid in kept)
    check(f"Los {len(kept)} nodos restantes siguen presentes en dof_node_ids", all_kept)

    # 1d: cada nodo aparece exactamente 6 veces en dof_node_ids
    counts = {int(nid): int(np.sum(space.dof_node_ids == nid)) for nid in space.node_ids}
    all_six = all(v == 6 for v in counts.values())
    wrong = [(k, v) for k, v in counts.items() if v != 6]
    print(f"\n  Repeticiones de cada GRID ID en dof_node_ids (esperado 6): "
          f"{list(counts.values())}")
    check("Cada nodo aparece exactamente 6 veces en dof_node_ids", all_six,
          f"Nodos con cuenta incorrecta: {wrong}")

    # 1e: matrices cuadradas y del tamano correcto
    n = space.n_dof
    check(f"M.shape == ({n},{n}) tras eliminacion", space.M.shape == (n, n),
          f"M.shape={space.M.shape}")
    check(f"K.shape == ({n},{n}) tras eliminacion", space.K.shape == (n, n),
          f"K.shape={space.K.shape}")
    check("modes.shape[0] == n_dof tras eliminacion",
          space.modes.shape[0] == n, f"modes.shape={space.modes.shape}")

    # 1f: node_xyz se actualiza
    space2 = _make_space(5)
    print(f"\n  Eliminando GRID 1003 de espacio con 5 nodos")
    space2.remove_nodes([1003])
    check("node_xyz.shape == (4, 3) tras eliminar 1 nodo",
          space2.node_xyz.shape == (4, 3), f"shape={space2.node_xyz.shape}")
    check("GRID 1003 no esta en node_ids", 1003 not in space2.node_ids)

    # 1g: IDs desconocidos no causan cambios
    space3 = _make_space(5)
    dof_antes = space3.n_dof
    print(f"\n  Intentando eliminar GRIDs inexistentes [9999, 8888]")
    space3.remove_nodes([9999, 8888])
    check("IDs inexistentes: n_dof no cambia", space3.n_dof == dof_antes,
          f"n_dof antes={dof_antes}, despues={space3.n_dof}")

    # 1h: len(dof_node_ids) == n_dof siempre
    space4 = _make_space(10)
    space4.remove_nodes([1001, 1005, 1010])
    check("len(dof_node_ids) == n_dof tras eliminacion multiple",
          len(space4.dof_node_ids) == space4.n_dof,
          f"len={len(space4.dof_node_ids)} vs n_dof={space4.n_dof}")


# ---------------------------------------------------------------------------
# Bloque 2 -- translational_dof_indices: layout bloqueado [Ux|Uy|Uz]
# ---------------------------------------------------------------------------

def test_translational_dof_indices():
    section("2. translational_dof_indices -- layout [Ux_all | Uy_all | Uz_all]")

    n_nodes = 5
    gdof = n_nodes * 6
    t_idx = translational_dof_indices(gdof)

    print(f"\n  n_nodes={n_nodes}, gdof={gdof}")
    print(f"  t_idx = {t_idx.tolist()}")
    print(f"  Bloque Ux (primeros {n_nodes}): {t_idx[:n_nodes].tolist()}")
    print(f"  Bloque Uy (siguientes {n_nodes}): {t_idx[n_nodes:2*n_nodes].tolist()}")
    print(f"  Bloque Uz (ultimos {n_nodes}): {t_idx[2*n_nodes:].tolist()}")

    check(f"Longitud = {n_nodes}x3 = {n_nodes*3}", len(t_idx) == n_nodes * 3,
          f"len={len(t_idx)}")

    ux_expected = np.arange(0, gdof, 6)
    check("Bloque Ux = [0, 6, 12, ...]",
          np.array_equal(t_idx[:n_nodes], ux_expected),
          f"obtenido={t_idx[:n_nodes].tolist()}, esperado={ux_expected.tolist()}")

    uy_expected = np.arange(1, gdof, 6)
    check("Bloque Uy = [1, 7, 13, ...]",
          np.array_equal(t_idx[n_nodes:2*n_nodes], uy_expected),
          f"obtenido={t_idx[n_nodes:2*n_nodes].tolist()}, esperado={uy_expected.tolist()}")

    uz_expected = np.arange(2, gdof, 6)
    check("Bloque Uz = [2, 8, 14, ...]",
          np.array_equal(t_idx[2*n_nodes:], uz_expected),
          f"obtenido={t_idx[2*n_nodes:].tolist()}, esperado={uz_expected.tolist()}")

    rotational = {3, 4, 5, 9, 10, 11, 15, 16, 17, 21, 22, 23}
    overlap = set(t_idx.tolist()) & rotational
    check("Ningun indice rotacional (Rx/Ry/Rz) seleccionado", len(overlap) == 0,
          f"indices rotacionales en t_idx: {overlap}")

    # Caso exacto con 2 nodos: esperamos [0, 6, 1, 7, 2, 8]
    t2 = translational_dof_indices(12)
    expected_2 = np.array([0, 6, 1, 7, 2, 8])
    print(f"\n  Para 2 nodos (gdof=12): t_idx={t2.tolist()}, esperado={expected_2.tolist()}")
    check("Caso 2 nodos: t_idx == [0, 6, 1, 7, 2, 8]",
          np.array_equal(t2, expected_2), f"obtenido={t2.tolist()}")


# ---------------------------------------------------------------------------
# Bloque 3 -- average_zones recibe el slice correcto tras remove_nodes + t_idx
# ---------------------------------------------------------------------------

def test_average_zones_slice():
    section("3. average_zones -- slice correcto tras remove_nodes + t_idx")

    # Modos deterministas: modo k, nodo i, componente c -> valor = (i+1)*10 + c + k*100
    # Ux nodo 0 modo 0 = 10,  Ux nodo 1 modo 0 = 20,  Ux nodo 2 modo 0 = 30
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

    # Zona = nodos posicionales 0, 1, 2
    subdomains = {"zona_A": [0, 1, 2]}
    result = average_zones(modes_t, subdomains, space.n_nodes)

    ux_nodo0 = modes[0,  0]   # = 10
    ux_nodo1 = modes[6,  0]   # = 20
    ux_nodo2 = modes[12, 0]   # = 30
    esperado = (ux_nodo0 + ux_nodo1 + ux_nodo2) / 3.0
    obtenido = result[0, 0]

    print(f"\n  Modos determin.: Ux nodo0={ux_nodo0}, nodo1={ux_nodo1}, nodo2={ux_nodo2}")
    print(f"  Media Ux zona (modo 0) esperada = {esperado:.4f}")
    print(f"  average_zones devuelve Ux zona  = {obtenido:.4f}")
    check("average_zones(Ux zona) == media aritmetica de Ux de los nodos",
          np.isclose(obtenido, esperado),
          f"obtenido={obtenido:.6f}, esperado={esperado:.6f}")

    # Shape correcta: 3 zonas x 3 DOFs, 3 modos
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
    print(f"\n  3 zonas, {n_modes2} modos -> shape esperada (9, 3), obtenida {r2.shape}")
    check(f"average_zones shape == (9, {n_modes2}) con 3 zonas",
          r2.shape == (9, n_modes2), f"obtenido={r2.shape}")

    # Tras remove_nodes, t_idx del espacio reducido cubre 3*n_nodes DOFs
    space3 = _make_space(8)
    space3.remove_nodes([1003])
    t3 = translational_dof_indices(space3.n_dof)
    print(f"\n  Tras remove_nodes([1003]): n_nodes={space3.n_nodes}, "
          f"n_dof={space3.n_dof}, len(t_idx)={len(t3)}")
    check("len(t_idx) == n_nodes*3 tras remove_nodes",
          len(t3) == space3.n_nodes * 3,
          f"len(t3)={len(t3)}, n_nodes*3={space3.n_nodes*3}")
    check("t_idx.max() < n_dof (todos los indices dentro del espacio)",
          t3.max() < space3.n_dof,
          f"t3.max()={t3.max()}, n_dof={space3.n_dof}")


# ---------------------------------------------------------------------------
# Bloque 4 -- Indices posicionales de subdominios validos tras remove_nodes
# ---------------------------------------------------------------------------

def test_subdomain_indices_after_removal():
    section("4. Subdominios -- indices posicionales validos tras remove_nodes")

    space = _make_space(10)
    print(f"\n  node_ids iniciales: {space.node_ids.tolist()}")
    space.remove_nodes([1005])
    print(f"  node_ids tras eliminar 1005: {space.node_ids.tolist()}")

    # JSON ficticio que incluye el nodo eliminado en una zona
    subdomains_grid = {
        "front": [1001, 1002, 1003],
        "mid":   [1004, 1005, 1006],   # 1005 ya no existe, se ignora
        "rear":  [1008, 1009, 1010],
    }
    subdomains_pos = grid_ids_to_node_indices(subdomains_grid, space.node_ids)

    print(f"\n  Zonas tras conversion (1005 eliminado de 'mid'):")
    for name, idxs in subdomains_pos.items():
        node_ids_in_zone = [space.node_ids[i] for i in idxs]
        print(f"    {name}: indices={idxs} -> GRIDs={node_ids_in_zone}")

    all_valid = all(
        0 <= idx < space.n_nodes
        for idxs in subdomains_pos.values()
        for idx in idxs
    )
    check("Todos los indices posicionales estan en [0, n_nodes)", all_valid)

    mid_ids = [space.node_ids[i] for i in subdomains_pos.get("mid", [])]
    check("GRID 1005 ausente de la zona 'mid' tras eliminacion",
          1005 not in mid_ids, f"mid contiene: {mid_ids}")

    # Zona con nodo eliminado al construir subdomains desde GRIDs
    space2 = _make_space(10)
    space2.remove_nodes([1005])
    subs2 = {"zone": [1001, 1005, 1009]}
    pos2  = grid_ids_to_node_indices(subs2, space2.node_ids)
    ids_in_zone = [space2.node_ids[i] for i in pos2["zone"]]
    print(f"\n  Zona [1001, 1005, 1009] con 1005 eliminado -> GRIDs en zona: {ids_in_zone}")
    check("GRID 1005 no aparece en subdomain convertido a indices posicionales",
          1005 not in ids_in_zone)


# ---------------------------------------------------------------------------
# Bloque 5 -- keep_dof_components actualiza dofs_per_node y layout
# ---------------------------------------------------------------------------

def test_keep_dof_components():
    section("5. keep_dof_components -- reduccion a DOFs translacionales")

    space = _make_space(6)
    n_nodes = space.n_nodes
    print(f"\n  Espacio inicial: {n_nodes} nodos, {space.n_dof} DOFs, "
          f"dofs_per_node={space.dofs_per_node}")
    space.keep_dof_components([0, 1, 2])
    print(f"  Tras keep_dof_components([0,1,2]): {space.n_dof} DOFs, "
          f"dofs_per_node={space.dofs_per_node}")

    check("dofs_per_node == 3 tras keep_dof_components([0,1,2])",
          space.dofs_per_node == 3, f"obtenido={space.dofs_per_node}")
    check(f"n_dof == {n_nodes}*3 = {n_nodes*3}",
          space.n_dof == n_nodes * 3, f"obtenido={space.n_dof}")

    for nid in space.node_ids:
        count = int(np.sum(space.dof_node_ids == nid))
        if count != 3:
            check(f"GRID {nid} aparece 3 veces en dof_node_ids", False,
                  f"aparece {count} veces")
            break
    else:
        check("Cada nodo aparece exactamente 3 veces en dof_node_ids", True)

    # Componente fuera de rango debe lanzar ValueError
    space2 = _make_space(3)
    try:
        space2.keep_dof_components([0, 1, 6])
        check("Componente 6 (fuera de rango) lanza ValueError", False,
              "no se lanzo ninguna excepcion")
    except ValueError as e:
        print(f"\n  ValueError esperado: {e}")
        check("Componente 6 (fuera de rango) lanza ValueError", True)


# ---------------------------------------------------------------------------
# Bloque 6 -- aset_dof_mask_from_gset (filtrado BIW G-set -> A-set)
# ---------------------------------------------------------------------------

def test_aset_dof_mask():
    section("6. aset_dof_mask_from_gset -- filtrado G-set -> A-set (BIW)")

    aset = np.array([10, 30])
    gset = np.array([10, 20, 30])
    mask = aset_dof_mask_from_gset(aset, gset)

    print(f"\n  G-set: {gset.tolist()}, A-set: {aset.tolist()}")
    print(f"  Mascara (1=A-set): {mask.astype(int).tolist()}")
    print(f"  Nodo 10 -> filas 0-5 : {mask[:6].astype(int).tolist()}")
    print(f"  Nodo 20 -> filas 6-11: {mask[6:12].astype(int).tolist()}  (excluido)")
    print(f"  Nodo 30 -> filas 12-17: {mask[12:18].astype(int).tolist()}")

    check("Longitud mascara = len(gset)*6",
          len(mask) == len(gset) * 6, f"len={len(mask)}")
    check("DOFs de nodo 10 (A-set) -> True",
          mask[:6].all(), f"mask[:6]={mask[:6].tolist()}")
    check("DOFs de nodo 20 (excluido) -> False",
          not mask[6:12].any(), f"mask[6:12]={mask[6:12].tolist()}")
    check("DOFs de nodo 30 (A-set) -> True",
          mask[12:18].all(), f"mask[12:18]={mask[12:18].tolist()}")
    check("Suma mascara = len(A-set)*6",
          int(mask.sum()) == len(aset) * 6, f"suma={mask.sum()}")


# ---------------------------------------------------------------------------
# Bloque 7 -- Pipeline BIW completo: mascara -> DofSpace -> t_idx -> average_zones
# ---------------------------------------------------------------------------

def test_full_biw_pipeline():
    section("7. Pipeline BIW completo: aset_mask -> DofSpace -> t_idx -> average_zones")

    n_gset   = 10
    aset_ids = np.array([1001, 1002, 1003, 1005, 1007, 1008, 1009])   # 7 nodos
    gset_ids = np.arange(1001, 1001 + n_gset, dtype=int)

    mask = aset_dof_mask_from_gset(aset_ids, gset_ids)
    excluded = gset_ids[~np.isin(gset_ids, aset_ids)].tolist()
    print(f"\n  G-set: {n_gset} nodos ({n_gset*6} DOFs)")
    print(f"  A-set: {len(aset_ids)} nodos ({int(mask.sum())} DOFs en mascara)")
    print(f"  Nodos excluidos (SPC/singular): {excluded}")

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

    print(f"\n  DofSpace creado: {space.n_nodes} nodos, {space.n_dof} DOFs")
    check(f"DofSpace.n_nodes == {len(aset_ids)} (A-set)",
          space.n_nodes == len(aset_ids), f"obtenido={space.n_nodes}")
    check(f"DofSpace.n_dof == {len(aset_ids)*6}",
          space.n_dof == len(aset_ids) * 6, f"obtenido={space.n_dof}")

    t_idx = translational_dof_indices(space.n_dof)
    print(f"  t_idx: {len(t_idx)} DOFs translacionales ({space.n_nodes} nodos x 3)")
    check(f"len(t_idx) == {space.n_nodes*3}",
          len(t_idx) == space.n_nodes * 3, f"obtenido={len(t_idx)}")

    modes_t = space.modes[t_idx, :]
    subdomains = {"front": [0, 1, 2], "rear": [3, 4, 5, 6]}
    result = average_zones(modes_t, subdomains, space.n_nodes)
    print(f"  average_zones con 2 zonas -> shape={result.shape} (esperado (6, 5))")
    check("average_zones shape == (6, 5): 2 zonas x 3 DOFs, 5 modos",
          result.shape == (6, 5), f"obtenido={result.shape}")


# ---------------------------------------------------------------------------
# Bloque 8 -- Pipeline TB completo: DofSpace(G-set) -> remove_nodes -> t_idx -> average_zones
# ---------------------------------------------------------------------------

def test_full_tb_pipeline():
    section("8. Pipeline TB completo: DofSpace(G-set) -> remove_nodes -> t_idx -> average_zones")

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

    print(f"\n  DofSpace G-set: {n_total} nodos, {n_dof} DOFs")
    print(f"  CONM2 a eliminar: {conm2_ids.tolist()}")
    space = DofSpace(modes, refs, I, I, R, node_ids, node_xyz)
    space.remove_nodes(conm2_ids)
    print(f"  Tras remove_nodes: {space.n_nodes} nodos, {space.n_dof} DOFs")

    n_expected = n_total - len(conm2_ids)
    check(f"n_nodes == {n_expected} tras eliminar CONM2",
          space.n_nodes == n_expected, f"obtenido={space.n_nodes}")
    check(f"n_dof == {n_expected * 6}",
          space.n_dof == n_expected * 6, f"obtenido={space.n_dof}")

    t_idx   = translational_dof_indices(space.n_dof)
    modes_t = space.modes[t_idx, :]
    print(f"  t_idx: {len(t_idx)} DOFs translacionales (esperado {space.n_nodes*3})")
    check(f"len(t_idx) == {space.n_nodes*3}",
          len(t_idx) == space.n_nodes * 3, f"obtenido={len(t_idx)}")

    subdomains_grid = {
        "z1": [1001, 1002, 1003],
        "z2": [1004, 1005, 1006],
        "z3": [1007, 1008, 1009],
    }
    subdomains_pos = grid_ids_to_node_indices(subdomains_grid, space.node_ids)
    result = average_zones(modes_t, subdomains_pos, space.n_nodes)
    print(f"  average_zones con 3 zonas -> shape={result.shape} (esperado (9, 4))")
    check("average_zones shape == (9, 4): 3 zonas x 3 DOFs, 4 modos",
          result.shape == (9, 4), f"obtenido={result.shape}")

    # CONM2 no deben aparecer en ningun subdomain
    subdomains_all = {"all": node_ids.tolist()}
    pos_all = grid_ids_to_node_indices(subdomains_all, space.node_ids)
    ids_in_zone = [space.node_ids[i] for i in pos_all["all"]]
    conm2_in_zone = [int(nid) for nid in conm2_ids if nid in ids_in_zone]
    print(f"  GRIDs CONM2 en subdomain 'all' tras conversion: {conm2_in_zone} (esperado [])")
    check("Ningun nodo CONM2 aparece en los subdominios tras remove_nodes",
          len(conm2_in_zone) == 0, f"encontrados: {conm2_in_zone}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 64)
    print("  Verificacion de reduccion de DOFs -- Pipeline ANSA")
    print("  Datos sinteticos, sin ficheros Nastran")
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
    print(f"  Resultado: {_passed}/{total} verificaciones correctas", end="")
    if _failed:
        print(f"  ({_failed} FALLARON)")
    else:
        print("  -- todo correcto")
    print(f"{'='*60}")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
