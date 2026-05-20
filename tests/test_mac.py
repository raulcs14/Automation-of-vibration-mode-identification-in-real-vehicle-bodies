"""
Visual test for MAC computation between dynamic modes and static reference shapes.

Run from anywhere:
    py -3 tests/test_mac.py

Interactive flow
----------------
  0. Model selection: Simple / ANSA BIW / ANSA TB
  1. Remove rigid-body component from reference shapes? (y/n)
  2. Use averaged subdomain vectors? (y/n)  [simple model only]
  3. Weighting: identity / mass / stiffness / total-energy
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt

from common.mac_core               import compute_mac
from common.subdomain              import average_zones, reduce_mk_by_subdomains
from common.rigid_body             import remove_rigid_body_component
from common.visualization.mac_plot import plot_mac_matrix
from common.utils                  import translational_dof_indices, densify
from test_helpers                  import ask_yn, ask_weighting

F0_ENERGY = 40.0


def _ask_model() -> tuple[str, str]:
    """Return (model_type, label): model_type is 'simple' or 'ansa', label for titles."""
    options = [
        "Simple model (beam FEM chassis)",
        "ANSA — Body in White (BIW)",
        "ANSA — Trimmed Body (TB)",
    ]
    print("\nSelecciona el modelo:")
    for i, opt in enumerate(options):
        print(f"  [{i+1}] {opt}")
    while True:
        raw = input("  > ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            idx = int(raw) - 1
            if idx == 0:
                return "simple", "Simple model"
            variant = ["BIW", "TB"][idx - 1]
            return "ansa", variant
        print(f"  Por favor introduce un número entre 1 y {len(options)}.")


def _load_simple():
    from simple_model.analysis.modal_analysis import run_modal_analysis
    from simple_model.analysis.static_model   import run_static_model, REF_NAMES
    from simple_model.geometry.chassis        import build_chassis_geometry

    dyn  = run_modal_analysis()
    stat = run_static_model()

    return dict(
        modes      = dyn["modes"],
        freq       = dyn["freq"],
        M          = dyn["M"],
        K          = dyn["K"],
        R          = dyn["R"],
        ref        = stat["ref_moves_raw"],
        ref_names  = REF_NAMES,
        subdomains = build_chassis_geometry("torsion").subdomains,
        mode_offset = 0,
    )


def _load_ansa(variant: str, remove_conm2: bool):
    from seat_model.modal_analysis import run_modal_analysis, N_RIGID_BODY_MODES
    from seat_model.static_model   import run_static_model
    from common.dof_reduction      import DofSpace

    dyn  = run_modal_analysis(variant=variant)
    stat = run_static_model(variant=variant)

    space = DofSpace(
        modes    = dyn["modes"],
        refs     = stat["ref_moves_raw"],
        M        = dyn["M"],
        K        = dyn["K"],
        R        = dyn["R"],
        node_ids = dyn["node_ids"],
        node_xyz = dyn["node_coordinates"],
    )

    conm2_node_ids = dyn.get("conm2_node_ids")
    if conm2_node_ids is not None and remove_conm2:
        print("Eliminando DOFs de nodos CONM2...")
        space.remove_nodes(conm2_node_ids)

    subdomains = None
    if variant == "BIW":
        from seat_model.subdomains import build_biw_subdomains
        subdomains = build_biw_subdomains(space.node_ids, space.node_xyz)
        print(f"  Subdominios BIW: {len(subdomains)} zonas")

    return dict(
        modes       = space.modes,
        freq        = dyn["freq"],
        M           = space.M,
        K           = space.K,
        R           = space.R,
        ref         = space.refs,
        ref_names   = stat["ref_names"],
        subdomains  = subdomains,
        n_nodes     = space.n_nodes,
        mode_offset = N_RIGID_BODY_MODES,
    )


def print_ranking(mac: np.ndarray, freq: np.ndarray, label: str,
                  ref_names: list, mode_offset: int = 0, top_n: int = 20) -> None:
    best_ref_idx = mac.argmax(axis=1)
    best_ref_val = mac.max(axis=1)

    # Sort modes by best MAC value descending, keep top_n
    order = np.argsort(best_ref_val)[::-1][:top_n]

    print(f"\n=== Top {top_n} modes by MAC  [{label}] ===")
    for rank, i in enumerate(order, 1):
        n = mode_offset + i + 1
        print(f"  {rank:2d}. Mode {n:3d} ({freq[i]:7.2f} Hz):  "
              f"{ref_names[best_ref_idx[i]]:<45s}  MAC = {best_ref_val[i]:.4f}")

    print(f"\n=== Best dynamic mode for each reference  [{label}] ===")
    best_mode_idx = mac.argmax(axis=0)
    best_mode_val = mac.max(axis=0)
    for j, rname in enumerate(ref_names):
        i = best_mode_idx[j]
        n = mode_offset + i + 1
        print(f"  {rname:<45s}  ->  Mode {n:3d} ({freq[i]:7.2f} Hz)  MAC = {best_mode_val[j]:.4f}")


def main():
    # --- Model selection ----------------------------------------------------
    model_type, model_label = _ask_model()
    print()

    if model_type == "simple":
        data = _load_simple()
    else:
        remove_conm2 = False
        if model_label == "TB":
            remove_conm2 = ask_yn("Remove CONM2 mass node DOFs before MAC computation?")
        data = _load_ansa(model_label, remove_conm2)

    modes       = data["modes"]
    freq        = data["freq"]
    M           = densify(data["M"])
    K           = densify(data["K"])
    R           = data["R"]
    ref         = data["ref"]
    ref_names   = data["ref_names"]
    subdomains  = data["subdomains"]
    mode_offset = data["mode_offset"]
    GDof        = modes.shape[0]
    n_nodes     = data.get("n_nodes", GDof // 6)

    # --- Interactive choices ------------------------------------------------
    use_rigid = ask_yn("Remove rigid-body component from reference shapes?")

    Psi = ref
    if use_rigid:
        Psi = remove_rigid_body_component(ref, M, R)

        import scipy.sparse as _sp
        t_idx_check = translational_dof_indices(GDof)
        M_t_check   = (M[t_idx_check, :][:, t_idx_check] if _sp.issparse(M)
                       else M[np.ix_(t_idx_check, t_idx_check)])
        e_before = np.einsum("ij,ij->j", ref[t_idx_check, :], densify(M_t_check) @ ref[t_idx_check, :])
        e_after  = np.einsum("ij,ij->j", Psi[t_idx_check, :], densify(M_t_check) @ Psi[t_idx_check, :])
        lost_pct = 100.0 * (e_before - e_after) / np.where(e_before > 0, e_before, 1.0)

        print("\nEnergy lost after rigid-body removal:")
        for j, name in enumerate(ref_names):
            print(f"  {j+1:2d}. {name:<45s}  {lost_pct[j]:6.1f} %")

        if not ask_yn("\nContinue with MAC computation?"):
            print("Aborted.")
            return

    use_zones = False
    if subdomains is not None:
        use_zones = ask_yn("Use averaged subdomain vectors?")

    w_idx, w_label = ask_weighting()

    title_parts = [model_label]
    if use_rigid: title_parts.append("rigid removed")
    if use_zones: title_parts.append("avg zones")
    title_parts.append(w_label)
    title = " | ".join(title_parts)

    # --- Translational extraction -------------------------------------------
    import scipy.sparse as _sp
    t_idx = translational_dof_indices(GDof)
    Phi_t = modes[t_idx, :]
    Psi_t = Psi[t_idx, :]
    if _sp.issparse(M):
        M_t = M[t_idx, :][:, t_idx]
        K_t = K[t_idx, :][:, t_idx]
    else:
        M_t = M[np.ix_(t_idx, t_idx)]
        K_t = K[np.ix_(t_idx, t_idx)]

    # --- Average zones (simple model only) ----------------------------------
    if use_zones:
        Phi_f = average_zones(Phi_t, subdomains, n_nodes)
        Psi_f = average_zones(Psi_t, subdomains, n_nodes)
        Mr, Kr, _ = reduce_mk_by_subdomains(M_t, K_t, subdomains, n_nodes)
        W_mass, W_stif = Mr, Kr
        W_ener = Mr * (2 * np.pi * F0_ENERGY) ** 2 + Kr
    else:
        Phi_f = Phi_t
        Psi_f = Psi_t
        W_mass, W_stif = M_t, K_t
        W_ener = M_t * (2 * np.pi * F0_ENERGY) ** 2 + K_t

    # --- Compute MAC --------------------------------------------------------
    W = {1: None, 2: W_mass, 3: W_stif, 4: W_ener}[w_idx]
    mac = compute_mac(Phi_f, Psi_f, W)

    # --- Print ranking & plot -----------------------------------------------
    TOP_N = 20
    print_ranking(mac, freq, title, ref_names, mode_offset, top_n=TOP_N)

    # Select top-N modes by best MAC, then sort by mode number for the plot
    best_mac = mac.max(axis=1)
    top_idx = np.sort(np.argsort(best_mac)[-TOP_N:])

    mode_labels = [f"Mode {mode_offset + i + 1} ({freq[i]:.2f} Hz)" for i in top_idx]
    plot_mac_matrix(mac[top_idx, :], mode_labels, ref_names, title=title)
    plt.show()


if __name__ == "__main__":
    main()
