"""
MAC computation for the ANSA Trimmed-Body model.

Run from anywhere:
    py -3 tests/SEAT/mac/tb/test_mac_ansa.py

Interactive flow
----------------
  1. Remove rigid-body component from reference shapes? (y/n)
  2. Weighting: identity / mass / stiffness / total-energy
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # tests/

import numpy as np
import matplotlib.pyplot as plt

from seat_model.modal_analysis import run_modal_analysis, N_RIGID_BODY_MODES
from seat_model.static_model    import run_static_model, REF_NAMES
from common.mac_core            import compute_mac, best_mac_per_mode, select_top_modes
from common.rigid_body          import remove_rigid_body_component
from common.visualization.mac_plot import plot_mac_matrix
from common.utils               import translational_dof_indices, densify
from test_helpers               import ask_yn, ask_weighting, ask_variant

F0_ENERGY = 40.0


N_TOP_MODES = 20


def print_ranking(mac: np.ndarray, freq: np.ndarray, label: str,
                  mode_numbers: np.ndarray = None) -> None:
    n_modes = mac.shape[0]
    print(f"\n=== Best reference for each dynamic mode  [{label}] ===")
    best_ref_idx = mac.argmax(axis=1)
    best_ref_val = mac.max(axis=1)
    for i in range(n_modes):
        mnum = mode_numbers[i] if mode_numbers is not None else i + 1
        print(f"  Mode {mnum:3d} ({freq[i]:7.2f} Hz):  "
              f"{REF_NAMES[best_ref_idx[i]]:<45s}  MAC = {best_ref_val[i]:.4f}")


def main():
    variant = ask_variant()
    dyn  = run_modal_analysis(variant)
    stat = run_static_model(variant)

    modes = dyn["modes"]          # (ADof, nModes)
    freq  = dyn["freq"]
    K     = dyn["K"]
    M     = dyn["M"]
    R     = dyn["R"]
    ref   = stat["ref_moves_raw"] # (ADof, nRefs)
    GDof  = modes.shape[0]

    # --- Interactive choices ------------------------------------------------
    print()
    use_rigid = ask_yn("Remove rigid-body component from reference shapes?")

    Phi = modes
    Psi = ref
    if use_rigid:
        M_dense = densify(M)
        Psi = remove_rigid_body_component(ref, M_dense, R)

        t_idx   = translational_dof_indices(GDof)
        M_t     = M_dense[np.ix_(t_idx, t_idx)]
        e_before = np.einsum("ij,ij->j", ref[t_idx, :],  M_t @ ref[t_idx, :])
        e_after  = np.einsum("ij,ij->j", Psi[t_idx, :],  M_t @ Psi[t_idx, :])
        lost_pct = 100.0 * (e_before - e_after) / np.where(e_before > 0, e_before, 1.0)

        print("\nEnergy lost after rigid-body removal:")
        for j, name in enumerate(REF_NAMES):
            print(f"  {j+1:2d}. {name:<45s}  {lost_pct[j]:6.1f} %")

        if not ask_yn("\nContinue with MAC computation?"):
            print("Aborted.")
            return

    w_idx, w_label = ask_weighting()

    title_parts = []
    if use_rigid:  title_parts.append("rigid removed")
    title_parts.append(w_label)
    title = " | ".join(title_parts)

    # --- Translational DOFs only -------------------------------------------
    t_idx = translational_dof_indices(GDof)
    Phi_t = Phi[t_idx, :]
    Psi_t = Psi[t_idx, :]

    if w_idx == 1:
        mac = compute_mac(Phi_t, Psi_t)
    else:
        M_dense = densify(M)
        K_dense = densify(K)
        M_t = M_dense[np.ix_(t_idx, t_idx)]
        K_t = K_dense[np.ix_(t_idx, t_idx)]
        W_ener = M_t * (2 * np.pi * F0_ENERGY) ** 2 + K_t
        W = {2: M_t, 3: K_t, 4: W_ener}[w_idx]
        mac = compute_mac(Phi_t, Psi_t, W)

    # --- Select top 30 modes by MAC value (sorted by frequency) ------------
    idx30 = select_top_modes({"mac": best_mac_per_mode(mac)}, N_TOP_MODES)
    mac30  = mac[idx30, :]
    freq30 = freq[idx30]

    # --- Print & plot -------------------------------------------------------
    global_mode_nums = N_RIGID_BODY_MODES + idx30 + 1
    print_ranking(mac30, freq30, title, mode_numbers=global_mode_nums)
    print(f"\n(Showing {len(idx30)} most relevant modes out of {len(freq)} total)")

    mode_labels = [f"Mode {N_RIGID_BODY_MODES + idx30[i]+1} ({freq30[i]:.2f} Hz)" for i in range(len(idx30))]
    plot_mac_matrix(mac30, mode_labels, REF_NAMES, title=f"ANSA TB — {title}")
    plt.show()


if __name__ == "__main__":
    main()
