"""
MAC comparison across all variants for the ANSA Trimmed-Body model.

Computes in one shot:
  - Plain MAC (identity)
  - Mass-weighted MAC
  - Stiffness-weighted MAC
  - Total-energy-weighted MAC
  Each repeated with and without rigid-body removal from the reference.

Shows:
  1. Bar chart — best MAC value per variant, for the N_TOP_MODES most relevant modes
  2. Summary table printed to console

Run from anywhere:
    py -3 tests/test_mac_comparison_ansa.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from seat_model.modal_analysis import run_modal_analysis, N_RIGID_BODY_MODES
from seat_model.static_model    import run_static_model, REF_NAMES
from common.mac_core            import compute_mac
from common.rigid_body          import remove_rigid_body_component
from common.utils               import translational_dof_indices, densify
from test_helpers               import best_mac_per_mode, ask_variant

F0_ENERGY  = 40.0
N_TOP_MODES = 20   # modes shown in the comparison plot


def compute_all_variants(modes, ref, M_dense, K_dense, R, t_idx) -> dict:
    """
    Compute MAC for all 8 variants. Returns dict: label -> (nModes,) best-MAC array.
    """
    Phi_t = modes[t_idx, :]
    M_t   = M_dense[np.ix_(t_idx, t_idx)]
    K_t   = K_dense[np.ix_(t_idx, t_idx)]
    W_ener = M_t * (2 * np.pi * F0_ENERGY) ** 2 + K_t

    weights = {
        "Identity":     None,
        "Mass":         M_t,
        "Stiffness":    K_t,
        "Energy":       W_ener,
    }

    results = {}
    for rigid_label, psi_full in [("", ref), ("  +rigid removed", None)]:
        if psi_full is None:
            psi_full = remove_rigid_body_component(ref, M_dense, R)
        Psi_t = psi_full[t_idx, :]

        for w_label, W in weights.items():
            label = f"{w_label}{rigid_label}"
            mac   = compute_mac(Phi_t, Psi_t, W)
            results[label] = best_mac_per_mode(mac)

    return results


def select_top_modes(variants: dict, n: int) -> np.ndarray:
    """Indices of the n modes that have the highest MAC in any variant."""
    stacked = np.stack(list(variants.values()), axis=0)  # (nVariants, nModes)
    best    = stacked.max(axis=0)
    top_idx = np.argsort(best)[-n:]
    return np.sort(top_idx)


def print_summary(variants: dict, idx: np.ndarray, freq: np.ndarray) -> None:
    labels  = list(variants.keys())
    col_w   = 10
    header  = f"{'Mode':<14}" + "".join(f"{l:>{col_w}}" for l in labels)
    print("\n=== Best MAC per variant (top modes) ===")
    print(header)
    print("-" * len(header))
    for i in idx:
        global_n = N_RIGID_BODY_MODES + i + 1
        row = f"Mode {global_n:3d} ({freq[i]:5.1f} Hz)"
        for l in labels:
            row += f"{variants[l][i]:>{col_w}.4f}"
        print(row)


def plot_comparison(variants: dict, idx: np.ndarray, freq: np.ndarray) -> None:
    labels   = list(variants.keys())
    n_modes  = len(idx)
    n_vars   = len(labels)

    x        = np.arange(n_modes)
    bar_w    = 0.8 / n_vars
    colors   = cm.tab10(np.linspace(0, 0.9, n_vars))

    fig, ax = plt.subplots(figsize=(max(14, n_modes * 0.8), 6))

    for k, (label, color) in enumerate(zip(labels, colors)):
        vals   = variants[label][idx]
        offset = (k - n_vars / 2 + 0.5) * bar_w
        bars   = ax.bar(x + offset, vals, width=bar_w, label=label,
                        color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            if v > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{v:.2f}", ha="center", va="bottom",
                        fontsize=5.5, rotation=90)

    global_nums = N_RIGID_BODY_MODES + idx + 1
    xlabels = [f"Mode {global_nums[i]}\n({freq[idx[i]]:.1f} Hz)"
               for i in range(n_modes)]

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=7)
    ax.set_ylabel("Best MAC value")
    ax.set_title("MAC comparison — all variants  (ANSA Trimmed Body)")
    ax.set_ylim(0, 1.12)
    ax.axhline(0.8, color="k", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.axhline(0.6, color="k", linewidth=0.6, linestyle=":",  alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()


def main():
    variant = ask_variant()
    print("Loading data...")
    dyn  = run_modal_analysis(variant)
    stat = run_static_model(variant)

    modes    = dyn["modes"]
    freq     = dyn["freq"]
    R        = dyn["R"]
    ref      = stat["ref_moves_raw"]
    GDof     = modes.shape[0]

    print("Densifying M and K (translational DOFs only)...")
    t_idx   = translational_dof_indices(GDof)
    M       = dyn["M"]
    K       = dyn["K"]
    M_dense = densify(M)
    K_dense = densify(K)

    print("Computing all MAC variants...")
    variants = compute_all_variants(modes, ref, M_dense, K_dense, R, t_idx)

    idx = select_top_modes(variants, N_TOP_MODES)

    print_summary(variants, idx, freq)
    plot_comparison(variants, idx, freq)
    plt.show()


if __name__ == "__main__":
    main()
