"""
MAC computation for the ANSA Trimmed-Body model.

Run from anywhere:
    py -3 tests/test_mac_ansa.py

Interactive flow
----------------
  1. Remove rigid-body component from reference shapes? (y/n)
  2. Weighting: identity / mass / stiffness / total-energy
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt

from ansa_model.modal_analysis import run_modal_analysis, N_RIGID_BODY_MODES
from ansa_model.static_model    import run_static_model, REF_NAMES
from common.mac_core            import compute_mac
from common.rigid_body          import remove_rigid_body_component
from common.visualization.mac_plot import plot_mac_matrix

F0_ENERGY = 40.0


def translational_dof_indices(gdof: int) -> np.ndarray:
    """Translational DOFs in block layout: [Ux_0..N | Uy_0..N | Uz_0..N]."""
    return np.concatenate([np.arange(d, gdof, 6) for d in range(3)])


def ask_yn(prompt: str) -> bool:
    while True:
        raw = input(prompt + " (y/n): ").strip().lower()
        if raw in ("y", "n"):
            return raw == "y"
        print("  Please enter y or n.")


def ask_weighting() -> tuple:
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


N_TOP_MODES = 30


def top_modes(mac: np.ndarray, n: int = N_TOP_MODES) -> np.ndarray:
    """Return indices of the n modes with highest MAC value (any reference),
    sorted by ascending index (i.e. ascending frequency)."""
    best_per_mode = mac.max(axis=1)
    top_idx = np.argsort(best_per_mode)[-n:]
    return np.sort(top_idx)


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
    dyn  = run_modal_analysis()
    stat = run_static_model()

    modes = dyn["modes"]          # (GDof, nModes)
    freq  = dyn["freq"]
    K     = dyn["K"]
    M     = dyn["M"]
    R     = dyn["R"]
    ref   = stat["ref_moves_raw"] # (GDof, nRefs)
    GDof  = modes.shape[0]

    # --- Interactive choices ------------------------------------------------
    print()
    use_rigid = ask_yn("Remove rigid-body component from reference shapes?")

    Phi = modes
    Psi = ref
    if use_rigid:
        M_dense = M.toarray() if hasattr(M, "toarray") else M
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
        M_dense = M.toarray() if hasattr(M, "toarray") else M
        K_dense = K.toarray() if hasattr(K, "toarray") else K
        M_t = M_dense[np.ix_(t_idx, t_idx)]
        K_t = K_dense[np.ix_(t_idx, t_idx)]
        W_ener = M_t * (2 * np.pi * F0_ENERGY) ** 2 + K_t
        W = {2: M_t, 3: K_t, 4: W_ener}[w_idx]
        mac = compute_mac(Phi_t, Psi_t, W)

    # --- Select top 30 modes by MAC value (sorted by frequency) ------------
    idx30 = top_modes(mac)
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
