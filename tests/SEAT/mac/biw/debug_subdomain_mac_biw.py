"""
Diagnostic script: inspect what average_zones does to modes and reference
in the BIW model, and why MAC values may be spurious.

Run from anywhere:
    py -3 tests/SEAT/mac/biw/debug_subdomain_mac_biw.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repo root

import numpy as np
import matplotlib.pyplot as plt

from seat_model.modal_analysis import run_modal_analysis, N_RIGID_BODY_MODES
from seat_model.static_model   import run_static_model
from seat_model.subdomains     import build_biw_subdomains
from common.subdomain          import average_zones
from common.utils              import translational_dof_indices


def main():
    dyn  = run_modal_analysis("BIW")
    stat = run_static_model("BIW")

    node_ids = dyn["node_ids"]
    node_xyz = dyn["node_coordinates"]
    modes    = dyn["modes"]          # (GDof, nModes)
    freq     = dyn["freq"]
    refs     = stat["ref_moves_raw"] # (GDof, 1)
    GDof     = modes.shape[0]
    n_nodes  = len(node_ids)

    t_idx   = translational_dof_indices(GDof)
    modes_t = modes[t_idx, :]       # (3*n_nodes, nModes)
    refs_t  = refs[t_idx, :]        # (3*n_nodes, 1)

    subdomains = build_biw_subdomains(node_ids, node_xyz)
    zone_names = list(subdomains.keys())
    n_zones    = len(zone_names)

    Phi_z = average_zones(modes_t, subdomains, n_nodes)  # (3*nZones, nModes)
    Psi_z = average_zones(refs_t,  subdomains, n_nodes)  # (3*nZones, 1)

    # -----------------------------------------------------------------------
    # 1. Norm of the averaged reference per zone  (should be non-trivial)
    # -----------------------------------------------------------------------
    psi_per_zone = Psi_z.reshape(n_zones, 3)             # (nZones, 3)
    psi_norms    = np.linalg.norm(psi_per_zone, axis=1)  # (nZones,)

    print(f"\n--- Reference norm per zone (sorted) ---")
    order = np.argsort(psi_norms)
    for i in order:
        print(f"  {zone_names[i]:<12}  ||ψ|| = {psi_norms[i]:.4e}")

    n_near_zero = (psi_norms < 1e-6 * psi_norms.max()).sum()
    print(f"\n  Zones with near-zero reference: {n_near_zero} / {n_zones}")

    # -----------------------------------------------------------------------
    # 2. Norm of the averaged mode per zone for the first few elastic modes
    # -----------------------------------------------------------------------
    print(f"\n--- Mode norm per zone (first 5 elastic modes) ---")
    for m in range(min(5, modes_t.shape[1])):
        phi_per_zone = Phi_z[:, m].reshape(n_zones, 3)
        phi_norms    = np.linalg.norm(phi_per_zone, axis=1)
        n_zero = (phi_norms < 1e-6 * phi_norms.max()).sum()
        print(f"  Mode {N_RIGID_BODY_MODES+m+1} ({freq[m]:.1f} Hz): "
              f"max={phi_norms.max():.3e}  near-zero zones={n_zero}/{n_zones}")

    # -----------------------------------------------------------------------
    # 3. Shared nodes between zones (overlap)
    # -----------------------------------------------------------------------
    all_zone_nodes = [idx for v in subdomains.values() for idx in v]
    unique_nodes   = len(set(all_zone_nodes))
    total_entries  = len(all_zone_nodes)
    print(f"\n--- Node overlap ---")
    print(f"  Unique nodes in model : {n_nodes}")
    print(f"  Unique nodes in zones : {unique_nodes}")
    print(f"  Total entries (w/ overlap): {total_entries}")
    print(f"  Overlap (nodes in >1 zone): {total_entries - unique_nodes}")

    # -----------------------------------------------------------------------
    # 4. Plot: reference norm per zone as bar chart
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    ax = axes[0]
    ax.bar(range(n_zones), psi_norms[np.argsort(np.arange(n_zones))],
           color=["red" if v < 1e-6 * psi_norms.max() else "steelblue"
                  for v in psi_norms])
    ax.set_xticks(range(n_zones))
    ax.set_xticklabels(zone_names, rotation=90, fontsize=5)
    ax.set_ylabel("||ψ_zone|| (averaged reference norm)")
    ax.set_title("Averaged static reference norm per PID zone\n"
                 "(red = near-zero → MAC numerically unstable)")
    ax.grid(axis="y", alpha=0.3)

    # -----------------------------------------------------------------------
    # 5. Plot: full-DOF reference norm per node (to see spatial distribution)
    # -----------------------------------------------------------------------
    ref_per_node = refs_t[:, 0].reshape(3, n_nodes)   # (3, n_nodes) block layout
    ref_node_norm = np.linalg.norm(ref_per_node, axis=0)  # (n_nodes,)

    ax2 = axes[1]
    sc = ax2.scatter(node_xyz[:, 0], node_xyz[:, 2],
                     c=ref_node_norm, cmap="viridis", s=2)
    plt.colorbar(sc, ax=ax2, label="||ref|| per node")
    ax2.set_xlabel("X [mm]")
    ax2.set_ylabel("Z [mm]")
    ax2.set_title("Static reference displacement magnitude — side view (X–Z)\n"
                  "(check that torsion pattern makes physical sense)")
    ax2.set_aspect("equal")
    ax2.grid(True, linewidth=0.3)

    fig.tight_layout()

    # -----------------------------------------------------------------------
    # 6. Plot: averaged reference vector per zone — Ux, Uy, Uz components
    #    This is the actual vector Ψ_z that goes into MAC
    # -----------------------------------------------------------------------
    psi_z_flat = Psi_z[:, 0].reshape(n_zones, 3)   # (nZones, 3)
    zone_centers = np.array([
        node_xyz[subdomains[name], :].mean(axis=0)
        for name in zone_names
    ])   # (nZones, 3)

    fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))
    comp_labels = ["Ux (torsion should be ~0)", "Uy (torsion: antisym in Y)", "Uz (torsion: antisym in X)"]
    for ax, ci, lbl in zip(axes3, range(3), comp_labels):
        vals = psi_z_flat[:, ci]
        sc = ax.scatter(zone_centers[:, 0], zone_centers[:, 2],
                        c=vals, cmap="RdBu_r", s=60,
                        vmin=-np.abs(vals).max(), vmax=np.abs(vals).max())
        plt.colorbar(sc, ax=ax)
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Z [mm]")
        ax.set_title(f"Ψ_zone  —  {lbl}", fontsize=8)
        ax.set_aspect("equal")
        ax.grid(True, linewidth=0.3)

    fig3.suptitle("Averaged reference per zone (what MAC sees as Ψ)", fontsize=10)
    fig3.tight_layout()

    # -----------------------------------------------------------------------
    # 7. MAC Identity: full DOFs vs zones — compare distributions
    # -----------------------------------------------------------------------
    from common.mac_core import compute_mac

    mac_full  = compute_mac(modes_t, refs_t,  W=None)   # (nModes, 1)
    mac_zones = compute_mac(Phi_z,   Psi_z,   W=None)   # (nModes, 1)

    fig4, ax4 = plt.subplots(figsize=(14, 4))
    x = np.arange(modes_t.shape[1])
    ax4.plot(x, mac_full[:, 0],  label="Full DOFs", alpha=0.8, linewidth=1.2)
    ax4.plot(x, mac_zones[:, 0], label="Zones (PID)", alpha=0.8, linewidth=1.2)
    ax4.set_xlabel("Mode index (elastic)")
    ax4.set_ylabel("MAC (Identity, torsion ref)")
    ax4.set_title("Full DOFs vs PID-zone MAC — Identity weighting")
    ax4.axhline(0.8, color="k", linewidth=0.8, linestyle="--", alpha=0.4)
    ax4.legend()
    ax4.grid(alpha=0.3)
    fig4.tight_layout()

    print(f"\n--- MAC summary ---")
    print(f"  Full DOFs:  max={mac_full.max():.3f}  mean={mac_full.mean():.3f}  "
          f"modes>0.8: {(mac_full>0.8).sum()}")
    print(f"  PID zones:  max={mac_zones.max():.3f}  mean={mac_zones.mean():.3f}  "
          f"modes>0.8: {(mac_zones>0.8).sum()}")

    plt.show()


if __name__ == "__main__":
    main()
