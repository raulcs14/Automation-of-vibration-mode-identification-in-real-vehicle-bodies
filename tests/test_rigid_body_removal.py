"""
Visual test for rigid-body component removal from static reference shapes.

Run from anywhere:
    py -3 tests/test_rigid_body_removal.py

Flow
----
  1. Select reference case to inspect (or all)
  2. Plots deformed mesh before and after rigid-body removal
  3. Ask whether to show averaged subdomain vectors after removal
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt

from simple_model.analysis.static_model   import run_static_model, REF_NAMES
from simple_model.analysis.modal_analysis import run_modal_analysis
from simple_model.geometry.chassis        import (build_chassis_geometry,
                                                   _draw_mesh_lines,
                                                   _set_equal_axes)
from common.rigid_body                    import remove_rigid_body_component
from common.subdomain                     import average_zones
from common.visualization.vectors         import plot_subdomain_vectors


def ask_case(n_cases):
    print("\nAvailable reference cases:")
    for i, name in enumerate(REF_NAMES):
        print(f"  {i+1:2d}. {name}")
    print(f"   0. Show all ({n_cases} cases)")
    while True:
        raw = input("Select case (0 = all): ").strip()
        if raw == "" or raw == "0":
            return None
        if raw.isdigit() and 1 <= int(raw) <= n_cases:
            return int(raw) - 1
        print(f"  Please enter a number between 0 and {n_cases}.")


def ask_yn(prompt: str) -> bool:
    while True:
        raw = input(prompt + " (y/n): ").strip().lower()
        if raw in ("y", "n"):
            return raw == "y"
        print("  Please enter y or n.")


def plot_deformed(ax, nc, en, u_raw, name, target_frac=0.08):
    UX = u_raw[0::6];  UY = u_raw[1::6];  UZ = u_raw[2::6]
    umax = np.sqrt(UX**2 + UY**2 + UZ**2).max()
    bbox_diag = np.linalg.norm(nc.max(axis=0) - nc.min(axis=0))
    scale = np.clip(target_frac * bbox_diag / max(umax, 1e-12), 0.1, 200)

    nc_def = nc + scale * np.column_stack([UX, UY, UZ])
    _draw_mesh_lines(ax, nc,     en, linestyle="k--")
    _draw_mesh_lines(ax, nc_def, en, linestyle="r-")
    _set_equal_axes(ax, np.vstack([nc, nc_def]))
    ax.set_title(f"{name}\nscale={scale:.1f}  umax={umax:.2e}", fontsize=7)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=135)
    ax.grid(True)


def show_single(nc, en, u_before, u_after, name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             subplot_kw={"projection": "3d"})
    plot_deformed(axes[0], nc, en, u_before, f"Original\n{name}", target_frac=0.08)
    plot_deformed(axes[1], nc, en, u_after,  f"Rigid removed\n{name}", target_frac=0.08)
    axes[0].title.set_fontsize(9)
    axes[1].title.set_fontsize(9)
    fig.suptitle(name, fontsize=10)
    fig.tight_layout()


def show_grid(nc, en, raw, raw_proj, n_cases):
    cols, rows = 4, 3
    for label, data in [("Original", raw), ("Rigid removed", raw_proj)]:
        fig = plt.figure(figsize=(5*cols, 4*rows))
        fig.suptitle(f"Static reference cases — {label}", fontsize=12)
        for k in range(n_cases):
            ax = fig.add_subplot(rows, cols, k + 1, projection="3d")
            plot_deformed(ax, nc, en, data[:, k], REF_NAMES[k])
            ax.tick_params(labelsize=5)
        fig.tight_layout()


def main():
    stat = run_static_model()
    dyn  = run_modal_analysis()

    nc      = stat["node_coordinates"]
    en      = stat["element_nodes"]
    raw     = stat["ref_moves_raw"]   # (GDof, nRefs)
    M       = dyn["M"]
    R       = dyn["R"]
    n_cases = raw.shape[1]
    GDof    = raw.shape[0]
    n_nodes = GDof // 6

    geo = build_chassis_geometry("torsion")
    subdomains = geo.subdomains

    # Remove rigid-body component from all cases at once
    raw_proj = remove_rigid_body_component(raw, M, R)

    # --- Ask user -----------------------------------------------------------
    chosen = ask_case(n_cases)

    if chosen is not None:
        show_single(nc, en, raw[:, chosen], raw_proj[:, chosen], REF_NAMES[chosen])
    else:
        show_grid(nc, en, raw, raw_proj, n_cases)

    # --- Average zones option -----------------------------------------------
    use_zones = ask_yn("\nShow averaged subdomain vectors after rigid removal?")
    if use_zones:
        t_idx   = np.concatenate([np.arange(d, GDof, 6) for d in range(3)])
        proj_t  = raw_proj[t_idx, :]
        vec_red = average_zones(proj_t, subdomains, n_nodes)

        if chosen is not None:
            fig = plt.figure(figsize=(9, 7))
            ax  = fig.add_subplot(111, projection="3d")
            plot_subdomain_vectors(ax, nc, en, subdomains, vec_red,
                                   scale_factor=5.0, mode_index=chosen,
                                   title=f"Rigid removed — {REF_NAMES[chosen]}")
            fig.tight_layout()
        else:
            cols, rows = 4, 3
            fig = plt.figure(figsize=(5*cols, 4*rows))
            fig.suptitle("Subdomain vectors — rigid removed", fontsize=12)
            for k in range(n_cases):
                ax = fig.add_subplot(rows, cols, k + 1, projection="3d")
                plot_subdomain_vectors(ax, nc, en, subdomains, vec_red,
                                       scale_factor=5.0, mode_index=k,
                                       title=REF_NAMES[k])
                ax.tick_params(labelsize=5)
                ax.title.set_fontsize(6)
            fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
