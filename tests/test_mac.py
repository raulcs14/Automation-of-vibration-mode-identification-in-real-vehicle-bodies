"""
Visual test for MAC computation between dynamic modes and static reference shapes.

Run from anywhere:
    py -3 tests/test_mac.py

Interactive flow
----------------
  1. Remove rigid-body component from reference shapes? (y/n)
  2. Use averaged subdomain vectors? (y/n)
  3. Weighting: identity / mass / stiffness / total-energy
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt

from simple_model.analysis.modal_analysis import run_modal_analysis
from simple_model.analysis.static_model   import run_static_model, REF_NAMES
from simple_model.geometry.chassis        import build_chassis_geometry
from common.mac_core                      import compute_mac
from common.subdomain                     import average_zones, reduce_mk_by_subdomains
from common.rigid_body                    import remove_rigid_body_component
from common.visualization.mac_plot        import plot_mac_matrix

F0_ENERGY = 40.0


def translational_dof_indices(gdof: int) -> np.ndarray:
    """Block order: [Ux_0..Ux_N | Uy_0..Uy_N | Uz_0..Uz_N]."""
    return np.concatenate([np.arange(d, gdof, 6) for d in range(3)])


def ask_yn(prompt: str) -> bool:
    while True:
        raw = input(prompt + " (y/n): ").strip().lower()
        if raw in ("y", "n"):
            return raw == "y"
        print("  Please enter y or n.")


def ask_weighting() -> int:
    options = {
        1: "Identity (plain MAC)",
        2: "Mass-weighted",
        3: "Stiffness-weighted",
        4: f"Total-energy-weighted  (M·(2π·{F0_ENERGY})² + K)",
    }
    print("\nWeighting:")
    for k, name in options.items():
        print(f"  {k}. {name}")
    while True:
        raw = input("Select (1-4): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= 4:
            return int(raw), options[int(raw)]
        print("  Please enter a number between 1 and 4.")


def print_ranking(mac: np.ndarray, freq: np.ndarray, label: str) -> None:
    n_modes = mac.shape[0]
    print(f"\n=== Best reference for each dynamic mode  [{label}] ===")
    best_ref_idx = mac.argmax(axis=1)
    best_ref_val = mac.max(axis=1)
    for i in range(n_modes):
        print(f"  Mode {i+1:2d} ({freq[i]:7.2f} Hz):  "
              f"{REF_NAMES[best_ref_idx[i]]:<45s}  MAC = {best_ref_val[i]:.4f}")

    print(f"\n=== Best dynamic mode for each reference  [{label}] ===")
    best_mode_idx = mac.argmax(axis=0)
    best_mode_val = mac.max(axis=0)
    for j in range(mac.shape[1]):
        i = best_mode_idx[j]
        print(f"  {REF_NAMES[j]:<45s}  ->  Mode {i+1:2d} ({freq[i]:7.2f} Hz)  MAC = {best_mode_val[j]:.4f}")


def main():
    # --- Load data ----------------------------------------------------------
    dyn  = run_modal_analysis()
    stat = run_static_model()

    modes = dyn["modes"]          # (GDof, nModes)
    freq  = dyn["freq"]
    K     = dyn["K"]
    M     = dyn["M"]
    R     = dyn["R"]              # (GDof, 6) rigid-body basis
    ref   = stat["ref_moves_raw"] # (GDof, nRefs)
    GDof  = modes.shape[0]
    n_nodes = GDof // 6

    geo = build_chassis_geometry("torsion")
    subdomains = geo.subdomains

    # --- Interactive choices ------------------------------------------------
    print()
    use_rigid = ask_yn("Remove rigid-body component from reference shapes?")

    Phi = modes
    Psi = ref
    if use_rigid:
        Psi = remove_rigid_body_component(ref, M, R)

        # Show energy lost per case and ask whether to continue
        t_idx_check = translational_dof_indices(GDof)
        M_t_check   = M[np.ix_(t_idx_check, t_idx_check)]
        e_before = np.einsum("ij,ij->j", ref[t_idx_check, :],  M_t_check @ ref[t_idx_check, :])
        e_after  = np.einsum("ij,ij->j", Psi[t_idx_check, :],  M_t_check @ Psi[t_idx_check, :])
        lost_pct = 100.0 * (e_before - e_after) / np.where(e_before > 0, e_before, 1.0)

        print("\nEnergy lost after rigid-body removal:")
        for j, name in enumerate(REF_NAMES):
            print(f"  {j+1:2d}. {name:<45s}  {lost_pct[j]:6.1f} %")

        if not ask_yn("\nContinue with MAC computation?"):
            print("Aborted.")
            return

    use_zones  = ask_yn("Use averaged subdomain vectors?")
    w_idx, w_label = ask_weighting()

    title_parts = []
    if use_rigid:  title_parts.append("rigid removed")
    if use_zones:  title_parts.append("avg zones")
    title_parts.append(w_label)
    title = " | ".join(title_parts)

    # --- Translational extraction -------------------------------------------
    t_idx = translational_dof_indices(GDof)
    Phi_t = Phi[t_idx, :]
    Psi_t = Psi[t_idx, :]
    M_t   = M[np.ix_(t_idx, t_idx)]
    K_t   = K[np.ix_(t_idx, t_idx)]

    # --- Average zones (optional) -------------------------------------------
    if use_zones:
        Phi_f = average_zones(Phi_t, subdomains, n_nodes)
        Psi_f = average_zones(Psi_t, subdomains, n_nodes)
        Mr, Kr, _ = reduce_mk_by_subdomains(M_t, K_t, subdomains, n_nodes)
        W_mass = Mr
        W_stif = Kr
        W_ener = Mr * (2 * np.pi * F0_ENERGY) ** 2 + Kr
    else:
        Phi_f = Phi_t
        Psi_f = Psi_t
        W_mass = M_t
        W_stif = K_t
        W_ener = M_t * (2 * np.pi * F0_ENERGY) ** 2 + K_t

    # --- Compute MAC --------------------------------------------------------
    if w_idx == 1:
        mac = compute_mac(Phi_f, Psi_f)
    elif w_idx == 2:
        mac = compute_mac(Phi_f, Psi_f, W_mass)
    elif w_idx == 3:
        mac = compute_mac(Phi_f, Psi_f, W_stif)
    else:
        mac = compute_mac(Phi_f, Psi_f, W_ener)

    # --- Print ranking & plot -----------------------------------------------
    print_ranking(mac, freq, title)

    mode_labels = [f"Mode {i+1} ({freq[i]:.2f} Hz)" for i in range(len(freq))]
    plot_mac_matrix(mac, mode_labels, REF_NAMES, title=title)
    plt.show()


if __name__ == "__main__":
    main()
