"""
MAC comparison across all variants for the simple model — reference case 4
(Roll/Torsion global).

Computes in one shot:
  - Plain MAC (identity)
  - Mass-weighted MAC
  - Stiffness-weighted MAC
  - Total-energy-weighted MAC
  Each repeated with and without rigid-body removal from the reference,
  and with and without subdomain averaging.

Shows:
  1. Bar chart — best MAC per variant for the N_TOP_MODES most relevant modes
  2. Summary table printed to console

Run from anywhere:
    py -3 tests/SEAT/mac/test_mac_comparison_simple.py
"""

import sys
from pathlib import Path
# Make the project root importable so this file runs under pytest AND when
# executed directly (IDE Run button).  Walk up to the repo root (the dir that
# contains the `common` package) instead of hard-coding a parents[N] depth.
_p = Path(__file__).resolve()
for _root in _p.parents:
    if (_root / "common").is_dir() and (_root / "main.py").is_file():
        break
sys.path.insert(0, str(_root))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from simple_model.analysis.modal_analysis import run_modal_analysis
from simple_model.analysis.static_model   import run_static_model, REF_NAMES
from simple_model.geometry.chassis        import build_chassis_geometry
from common.mac_core                      import compute_mac
from common.rigid_body                    import remove_rigid_body_component
from common.subdomain                     import average_zones, reduce_mk_by_subdomains
from common.utils                         import translational_dof_indices
from common.mac_core                      import best_mac_per_mode, select_top_modes

REF_CASE   = 3        # 0-based index → case 4 "Roll/Torsion global"
F0_ENERGY  = 40.0
N_TOP_MODES = 10


def compute_all_variants(modes, ref, M, K, R, t_idx,
                         subdomains, n_nodes) -> dict:
    """
    Return dict: label -> (nModes,) best-MAC array.
    Variants: {Identity, Mass, Stiffness, Energy}
              × {no rigid removal, rigid removed}
              × {full DOFs, subdomain averaged}
    """
    import scipy.sparse as _sp
    Phi_t = modes[t_idx, :]
    psi_t = ref[t_idx]
    if _sp.issparse(M):
        M_t = M[t_idx, :][:, t_idx]
        K_t = K[t_idx, :][:, t_idx]
    else:
        M_t = M[np.ix_(t_idx, t_idx)]
        K_t = K[np.ix_(t_idx, t_idx)]
    W_ener = M_t * (2 * np.pi * F0_ENERGY) ** 2 + K_t

    ref_proj = remove_rigid_body_component(ref.reshape(-1,1), M, R)[:, 0]
    psi_t_proj = ref_proj[t_idx]

    Mr, Kr, _ = reduce_mk_by_subdomains(M_t, K_t, subdomains, n_nodes)
    W_ener_r  = Mr * (2 * np.pi * F0_ENERGY) ** 2 + Kr

    Phi_z     = average_zones(Phi_t, subdomains, n_nodes)
    psi_z     = average_zones(psi_t.reshape(-1, 1), subdomains, n_nodes)[:, 0]
    psi_z_proj = average_zones(psi_t_proj.reshape(-1, 1), subdomains, n_nodes)[:, 0]

    results = {}

    # --- Full DOFs ---
    for rigid, psi in [("", psi_t), ("+rigid", psi_t_proj)]:
        psi2 = psi.reshape(-1, 1)
        for w_lbl, W in [("Identity", None), ("Mass", M_t),
                         ("Stiffness", K_t), ("Energy", W_ener)]:
            label = f"{w_lbl} {rigid}".strip()
            results[label] = best_mac_per_mode(compute_mac(Phi_t, psi2, W))

    # --- Subdomain averaged ---
    for rigid, psi in [("", psi_z), ("+rigid", psi_z_proj)]:
        psi2 = psi.reshape(-1, 1)
        for w_lbl, W in [("Identity", None), ("Mass", Mr),
                         ("Stiffness", Kr), ("Energy", W_ener_r)]:
            label = f"{w_lbl} zones {rigid}".strip()
            results[label] = best_mac_per_mode(compute_mac(Phi_z, psi2, W))

    return results


def print_summary(variants: dict, idx: np.ndarray, freq: np.ndarray) -> None:
    labels = list(variants.keys())
    col_w  = 13
    header = f"{'Mode':<16}" + "".join(f"{l:>{col_w}}" for l in labels)
    print(f"\n=== Best MAC per variant — ref case {REF_CASE+1}: {REF_NAMES[REF_CASE]} ===")
    print(header)
    print("-" * len(header))
    for i in idx:
        row = f"Mode {i+1:2d} ({freq[i]:6.2f} Hz)"
        for l in labels:
            row += f"{variants[l][i]:>{col_w}.4f}"
        print(row)


def plot_comparison(variants: dict, idx: np.ndarray, freq: np.ndarray) -> None:
    labels  = list(variants.keys())
    n_modes = len(idx)
    n_vars  = len(labels)

    x      = np.arange(n_modes)
    bar_w  = 0.8 / n_vars
    colors = cm.tab20(np.linspace(0, 1, n_vars))

    fig, ax = plt.subplots(figsize=(max(14, n_modes * 1.2), 6))

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
                        fontsize=5, rotation=90)

    xlabels = [f"Mode {idx[i]+1}\n({freq[idx[i]]:.1f} Hz)" for i in range(n_modes)]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("Best MAC value")
    ax.set_title(f"MAC comparison — all variants\n"
                 f"Ref case {REF_CASE+1}: {REF_NAMES[REF_CASE]}")
    ax.set_ylim(0, 1.18)
    ax.axhline(0.8, color="k", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.axhline(0.6, color="k", linewidth=0.6, linestyle=":",  alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()


def main():
    dyn  = run_modal_analysis()
    stat = run_static_model()

    modes = dyn["modes"]
    freq  = dyn["freq"]
    K     = dyn["K"]
    M     = dyn["M"]
    R     = dyn["R"]
    ref   = stat["ref_moves_raw"][:, REF_CASE]   # single vector
    GDof  = modes.shape[0]
    n_nodes = GDof // 6

    geo        = build_chassis_geometry("torsion")
    subdomains = geo.subdomains

    t_idx = translational_dof_indices(GDof)

    variants = compute_all_variants(modes, ref, M, K, R, t_idx,
                                    subdomains, n_nodes)

    idx = select_top_modes(variants, N_TOP_MODES)

    print_summary(variants, idx, freq)
    plot_comparison(variants, idx, freq)
    plt.show()


if __name__ == "__main__":
    main()
