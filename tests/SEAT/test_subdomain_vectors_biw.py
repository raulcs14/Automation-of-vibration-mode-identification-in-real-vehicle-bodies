"""
Visual test for BIW subdomain averaging + MAC with subdomain-averaged references.

Flow:
  1. Ask which elastic mode to inspect.
  2. Show a 3-D figure with the subdomain-averaged vectors for that mode.
  3. Compute MAC using subdomain-averaged modes vs subdomain-averaged static
     reference, and print + plot the result.

Run from anywhere:
    py -3 tests/SEAT/test_subdomain_vectors_biw.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt

from ansa_model.modal_analysis  import run_modal_analysis, N_RIGID_BODY_MODES
from ansa_model.static_model    import run_static_model, REF_NAMES
from ansa_model.subdomains      import build_biw_subdomains
from common.subdomain           import average_zones
from common.mac_core            import compute_mac
from common.visualization.mac_plot import plot_mac_matrix
from common.utils               import translational_dof_indices
from test_helpers               import ask_mode

SCALE_FACTOR = 500.0   # mm — visual scale for displacement arrows
N_TOP_MODES  = 20      # modes shown in MAC plot and ranking


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _set_equal_axes(ax, xyz: np.ndarray) -> None:
    mins, maxs = xyz.min(axis=0), xyz.max(axis=0)
    center = (mins + maxs) / 2
    half   = (maxs - mins).max() / 2 * 0.6
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def plot_subdomain_vectors_biw(
    ax,
    node_xyz: np.ndarray,
    subdomains: dict,
    vectors: np.ndarray,
    mode_index: int,
    scale: float,
    title: str,
) -> None:
    """
    Draw the subdomain-averaged displacement vectors for one mode over the
    BIW point cloud.

    vectors : (3*nZones, nModes) — output of average_zones
    """
    zone_names = list(subdomains.keys())
    n_zones    = len(zone_names)

    col    = vectors[:, mode_index]          # (3*nZones,)
    v_red  = col.reshape(n_zones, 3)         # (nZones, 3)
    vmax   = np.linalg.norm(v_red, axis=1).max()
    if vmax < 1e-15:
        vmax = 1.0
    sc = scale / vmax

    # Faint point cloud for context
    ax.scatter(node_xyz[:, 0], node_xyz[:, 1], node_xyz[:, 2],
               s=0.3, c="lightgray", alpha=0.25, depthshade=False)

    colors = plt.cm.tab20(np.linspace(0, 1, n_zones))

    for k, name in enumerate(zone_names):
        idx    = subdomains[name]
        center = node_xyz[idx, :].mean(axis=0)
        vec    = v_red[k] * sc

        ax.scatter(*center, s=25, color=colors[k], zorder=5)
        ax.quiver(*center, *vec,
                  color=colors[k], linewidth=1.5, arrow_length_ratio=0.25)

    _set_equal_axes(ax, node_xyz)
    ax.set_title(title, fontsize=7)
    ax.set_xlabel("X [mm]", fontsize=6)
    ax.set_ylabel("Y [mm]", fontsize=6)
    ax.set_zlabel("Z [mm]", fontsize=6)
    ax.tick_params(labelsize=5)
    ax.view_init(elev=20, azim=200)
    ax.grid(True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading BIW modal data...")
    dyn  = run_modal_analysis("BIW")
    stat = run_static_model("BIW")

    modes    = dyn["modes"]           # (ADof, nModes)
    freq     = dyn["freq"]
    node_ids = dyn["node_ids"]
    node_xyz = dyn["node_coordinates"]
    refs     = stat["ref_moves_raw"]  # (ADof, nRefs)
    n_modes  = modes.shape[1]
    n_nodes  = len(node_ids)
    GDof     = modes.shape[0]

    # --- Build subdomains (positional indices into node_ids) ----------------
    print("Building subdomains...")
    subdomains = build_biw_subdomains(node_ids, node_xyz)
    print(f"  {len(subdomains)} zones, {n_nodes} nodes covered")

    # --- Translational DOFs: block layout [Ux|Uy|Uz] ----------------------
    t_idx   = translational_dof_indices(GDof)
    modes_t = modes[t_idx, :]      # (3*n_nodes, nModes)
    refs_t  = refs[t_idx, :]       # (3*n_nodes, nRefs)

    # --- Subdomain averages ------------------------------------------------
    modes_red = average_zones(modes_t, subdomains, n_nodes)  # (3*nZones, nModes)
    refs_red  = average_zones(refs_t,  subdomains, n_nodes)  # (3*nZones, nRefs)

    # --- Ask which mode to visualise ---------------------------------------
    chosen = ask_mode(n_modes, freq)

    if chosen is not None:
        fig = plt.figure(figsize=(10, 7))
        ax  = fig.add_subplot(111, projection="3d")
        global_n = N_RIGID_BODY_MODES + chosen + 1
        plot_subdomain_vectors_biw(
            ax, node_xyz, subdomains, modes_red,
            mode_index=chosen,
            scale=SCALE_FACTOR,
            title=f"BIW Mode {global_n}  ({freq[chosen]:.2f} Hz) — subdomain averages",
        )
        fig.tight_layout()
        plt.show()

    # --- MAC with subdomain-averaged modes and references ------------------
    print("\nComputing MAC (subdomain-averaged)...")
    mac = compute_mac(modes_red, refs_red)   # (nModes, nRefs)

    # Select top N_TOP_MODES by best MAC value across all references
    top_idx  = np.sort(np.argsort(mac.max(axis=1))[-N_TOP_MODES:])
    mac_top  = mac[top_idx, :]

    # Print ranking
    best_ref = mac_top.argmax(axis=1)
    best_val = mac_top.max(axis=1)
    print(f"\n{'Mode':<18}  {'Best reference':<35}  MAC")
    print("-" * 65)
    for k, i in enumerate(top_idx):
        global_n = N_RIGID_BODY_MODES + i + 1
        print(f"Mode {global_n:3d} ({freq[i]:6.2f} Hz)  "
              f"{REF_NAMES[best_ref[k]]:<35s}  {best_val[k]:.4f}")

    # Plot MAC matrix (top modes only)
    mode_labels = [
        f"Mode {N_RIGID_BODY_MODES + i + 1} ({freq[i]:.1f} Hz)"
        for i in top_idx
    ]
    plot_mac_matrix(mac_top, mode_labels, REF_NAMES,
                    title=f"BIW — MAC subdomain-averaged (top {N_TOP_MODES} modes)")
    plt.show()


if __name__ == "__main__":
    main()
