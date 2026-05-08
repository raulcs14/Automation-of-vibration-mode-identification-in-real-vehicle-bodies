"""
Subdomain MAC analysis for the Trimmed Body (TB) model.

Shows the effect of subdomain averaging on mode identification, with and
without CONM2 (lumped mass) node removal.  Four variants are compared:

  Full DOFs   (all nodes)  — no subdomains, no CONM2 removal
  Zones       (all nodes)  — subdomain-averaged,  no CONM2 removal
  Full DOFs   (no CONM2)   — no subdomains, CONM2 nodes removed
  Zones       (no CONM2)   — subdomain-averaged,  CONM2 nodes removed

Weighting is chosen interactively: Identity / Mass / Stiffness / Energy.

Run from anywhere:
    py -3 tests/SEAT/test_subdomain_mac_tb.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from ansa_model.modal_analysis  import run_modal_analysis, N_RIGID_BODY_MODES
from ansa_model.static_model    import run_static_model, REF_NAMES
from ansa_model.subdomains      import build_biw_subdomains
from common.dof_reduction       import DofSpace
from common.subdomain           import average_zones, reduce_mk_by_subdomains
from common.mac_core            import compute_mac
from common.utils               import translational_dof_indices
from test_helpers               import ask_weighting

F0_ENERGY  = 40.0
N_TOP_MODES = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_variants(space: DofSpace, label: str, w_idx: int) -> dict:
    """
    Compute MAC for full-DOF and subdomain-averaged vectors with the chosen
    weighting.  Returns dict: variant_label -> (nModes, nRefs) MAC matrix.

    w_idx : 1=Identity  2=Mass  3=Stiffness  4=Energy
    """
    import scipy.sparse as sp
    GDof    = space.n_dof
    n_nodes = space.n_nodes
    t_idx   = translational_dof_indices(GDof)

    Phi_t = space.modes[t_idx, :]
    Psi_t = space.refs[t_idx, :]

    # Translational M/K — keep sparse
    if sp.issparse(space.M):
        M_t = space.M[t_idx, :][:, t_idx]
        K_t = space.K[t_idx, :][:, t_idx]
    else:
        M_t = space.M[np.ix_(t_idx, t_idx)]
        K_t = space.K[np.ix_(t_idx, t_idx)]

    W_ener = M_t * (2 * np.pi * F0_ENERGY) ** 2 + K_t
    W_full = {1: None, 2: M_t, 3: K_t, 4: W_ener}[w_idx]

    results = {}
    results[f"Full  {label}"] = compute_mac(Phi_t, Psi_t, W_full)

    subdomains = build_biw_subdomains(space.node_ids, space.node_xyz)
    Phi_z = average_zones(Phi_t, subdomains, n_nodes)
    Psi_z = average_zones(Psi_t, subdomains, n_nodes)

    # Reduced M/K for subdomain weighting
    Mr, Kr, _ = reduce_mk_by_subdomains(M_t, K_t, subdomains, n_nodes)
    W_ener_r  = Mr * (2 * np.pi * F0_ENERGY) ** 2 + Kr
    W_zones   = {1: None, 2: Mr, 3: Kr, 4: W_ener_r}[w_idx]

    results[f"Zones {label}"] = compute_mac(Phi_z, Psi_z, W_zones)

    return results


def _print_table(mac_matrices: dict, top_idx: np.ndarray,
                 freq: np.ndarray, w_label: str) -> None:
    variants = list(mac_matrices.keys())
    col_w    = 18
    header   = f"{'Mode':<22}" + "".join(f"{v:^{col_w}}" for v in variants)
    print(f"\n{'='*len(header)}")
    print(f"  TB subdomain MAC — {w_label} weighting — top {N_TOP_MODES} modes")
    print(f"{'='*len(header)}")
    print(header)
    print("─" * len(header))
    for i in top_idx:
        gn  = N_RIGID_BODY_MODES + i + 1
        row = f"Mode {gn:3d} ({freq[i]:6.2f} Hz)"
        for v in variants:
            row += f"{mac_matrices[v][i].max():^{col_w}.4f}"
        print(row)
    print("─" * len(header))


def _plot_comparison(mac_matrices: dict, top_idx: np.ndarray,
                     freq: np.ndarray, w_label: str) -> None:
    labels  = list(mac_matrices.keys())
    n_modes = len(top_idx)
    n_vars  = len(labels)
    x       = np.arange(n_modes)
    bar_w   = 0.8 / n_vars
    colors  = cm.tab10(np.linspace(0, 0.8, n_vars))

    fig, ax = plt.subplots(figsize=(max(14, n_modes * 0.9), 6))
    for k, (label, color) in enumerate(zip(labels, colors)):
        vals   = mac_matrices[label][top_idx].max(axis=1)
        offset = (k - n_vars / 2 + 0.5) * bar_w
        bars   = ax.bar(x + offset, vals, width=bar_w,
                        label=label, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            if v > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{v:.2f}", ha="center", va="bottom",
                        fontsize=5.5, rotation=90)

    global_nums = N_RIGID_BODY_MODES + top_idx + 1
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"Mode {global_nums[i]}\n({freq[top_idx[i]]:.1f} Hz)"
         for i in range(n_modes)],
        fontsize=7,
    )
    ax.set_ylabel("Best MAC value")
    ax.set_title(
        f"TB — subdomain averaging effect  [{w_label}]  (top {N_TOP_MODES} modes)"
    )
    ax.set_ylim(0, 1.12)
    ax.axhline(0.8, color="k", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.axhline(0.6, color="k", linewidth=0.6, linestyle=":",  alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading TB data...")
    dyn  = run_modal_analysis("TB")
    stat = run_static_model("TB")

    freq           = dyn["freq"]
    conm2_node_ids = dyn["conm2_node_ids"]

    # --- Weighting selection -------------------------------------------------
    w_idx, w_label = ask_weighting()

    # --- Without CONM2 removal -----------------------------------------------
    print("\nBuilding DofSpace (full G-set, no CONM2 removal)...")
    space_full = DofSpace(
        modes    = dyn["modes"],
        refs     = stat["ref_moves_raw"],
        M        = dyn["M"],
        K        = dyn["K"],
        R        = dyn["R"],
        node_ids = dyn["node_ids"],
        node_xyz = dyn["node_coordinates"],
    )
    print(f"  {space_full.n_nodes} nodes, {space_full.n_dof} DOFs")

    print("Computing MAC (all nodes)...")
    mac_full = _compute_variants(space_full, "(all nodes)", w_idx)

    # --- With CONM2 removal --------------------------------------------------
    print("\nBuilding DofSpace (CONM2 nodes removed)...")
    space_conm2 = DofSpace(
        modes    = dyn["modes"],
        refs     = stat["ref_moves_raw"],
        M        = dyn["M"],
        K        = dyn["K"],
        R        = dyn["R"],
        node_ids = dyn["node_ids"],
        node_xyz = dyn["node_coordinates"],
    )
    space_conm2.remove_nodes(conm2_node_ids)
    print(f"  {space_conm2.n_nodes} nodes, {space_conm2.n_dof} DOFs")

    print("Computing MAC (no CONM2)...")
    mac_conm2 = _compute_variants(space_conm2, "(no CONM2)", w_idx)

    # --- Merge and select top modes ------------------------------------------
    all_mac = {**mac_full, **mac_conm2}

    best = np.stack([v.max(axis=1) for v in all_mac.values()]).max(axis=0)
    top  = np.sort(np.argsort(best)[-N_TOP_MODES:])

    _print_table(all_mac, top, freq, w_label)
    _plot_comparison(all_mac, top, freq, w_label)
    plt.show()


if __name__ == "__main__":
    main()
