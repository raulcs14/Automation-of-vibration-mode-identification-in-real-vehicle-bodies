"""
Visual test for MAC computation between dynamic modes and static reference shapes.

Run from anywhere:
    py -3 tests/test_mac.py

Asks which weighting to use, prints the best-match ranking, and shows the heatmap.

Weighting options
-----------------
  1. Identity (plain MAC)
  2. Mass-weighted
  3. Stiffness-weighted
  4. Total-energy-weighted  (M*(2π·f0)² + K,  f0=40 Hz)
  5. Reduced — identity     (averaged zones, W=I)
  6. Reduced — mass         (averaged zones, Mr)
  7. Reduced — stiffness    (averaged zones, Kr)
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
from common.visualization.mac_plot        import plot_mac_matrix

F0_ENERGY = 40.0   # reference frequency for total-energy weight [Hz]

WEIGHT_MENU = {
    1: "Identity (plain MAC)",
    2: "Mass-weighted",
    3: "Stiffness-weighted",
    4: f"Total-energy-weighted  (M·(2π·{F0_ENERGY})² + K)",
    5: "Reduced zones — identity",
    6: "Reduced zones — mass",
    7: "Reduced zones — stiffness",
}


def translational_dof_indices(gdof: int) -> np.ndarray:
    """
    Ux, Uy, Uz indices (0-based) in block order: [Ux_0..Ux_N | Uy_0..Uy_N | Uz_0..Uz_N].
    This matches the layout expected by average_zones and reduce_mk_by_subdomains.
    """
    return np.concatenate([np.arange(d, gdof, 6) for d in range(3)])


def ask_weighting() -> int:
    print("\nAvailable MAC weightings:")
    for k, name in WEIGHT_MENU.items():
        print(f"  {k}. {name}")
    while True:
        raw = input("Select weighting (1-7): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= 7:
            return int(raw)
        print("  Please enter a number between 1 and 7.")


def print_ranking(mac: np.ndarray, freq: np.ndarray, ref_names: list, label: str) -> None:
    n_modes, n_refs = mac.shape
    print(f"\n=== Best reference for each dynamic mode  [{label}] ===")
    best_ref_idx  = mac.argmax(axis=1)
    best_ref_val  = mac.max(axis=1)
    for i in range(n_modes):
        print(f"  Mode {i+1:2d} ({freq[i]:7.2f} Hz):  "
              f"{ref_names[best_ref_idx[i]]:<45s}  MAC = {best_ref_val[i]:.4f}")

    print(f"\n=== Best dynamic mode for each reference  [{label}] ===")
    best_mode_idx = mac.argmax(axis=0)
    best_mode_val = mac.max(axis=0)
    for j in range(n_refs):
        i = best_mode_idx[j]
        print(f"  {ref_names[j]:<45s}  ->  Mode {i+1:2d} ({freq[i]:7.2f} Hz)  MAC = {best_mode_val[j]:.4f}")


def main():
    # --- Load data ----------------------------------------------------------
    dyn   = run_modal_analysis()
    stat  = run_static_model()
    geo   = build_chassis_geometry("torsion")

    modes = dyn["modes"]        # (GDof, nModes)
    freq  = dyn["freq"]
    K     = dyn["K"]
    M     = dyn["M"]
    ref   = stat["ref_moves_raw"]   # (GDof, nRefs)  unnormalized
    GDof  = modes.shape[0]
    n_nodes = GDof // 6

    subdomains = geo.subdomains

    # Translational DOF subset (free-free → no prescribed DOFs to remove)
    t_idx = translational_dof_indices(GDof)

    Phi_t = modes[t_idx, :]
    Psi_t = ref[t_idx, :]
    M_t   = M[np.ix_(t_idx, t_idx)]
    K_t   = K[np.ix_(t_idx, t_idx)]

    # Reduced (averaged zones) — work entirely in translational space
    Phi_red = average_zones(Phi_t, subdomains, n_nodes)   # (3·nZones, nModes)
    Psi_red = average_zones(Psi_t, subdomains, n_nodes)   # (3·nZones, nRefs)
    Mr, Kr, _ = reduce_mk_by_subdomains(M_t, K_t, subdomains, n_nodes)

    # --- Ask user -----------------------------------------------------------
    choice = ask_weighting()
    label  = WEIGHT_MENU[choice]

    if choice == 1:
        mac = compute_mac(Phi_t, Psi_t)
    elif choice == 2:
        mac = compute_mac(Phi_t, Psi_t, M_t)
    elif choice == 3:
        mac = compute_mac(Phi_t, Psi_t, K_t)
    elif choice == 4:
        E_t = M_t * (2 * np.pi * F0_ENERGY) ** 2 + K_t
        mac = compute_mac(Phi_t, Psi_t, E_t)
    elif choice == 5:
        mac = compute_mac(Phi_red, Psi_red)
    elif choice == 6:
        mac = compute_mac(Phi_red, Psi_red, Mr)
    else:
        mac = compute_mac(Phi_red, Psi_red, Kr)

    # --- Print ranking ------------------------------------------------------
    print_ranking(mac, freq, REF_NAMES, label)

    # --- Plot ---------------------------------------------------------------
    mode_labels = [f"Mode {i+1} ({freq[i]:.2f} Hz)" for i in range(len(freq))]
    plot_mac_matrix(mac, mode_labels, REF_NAMES, title=label)
    plt.show()


if __name__ == "__main__":
    main()
