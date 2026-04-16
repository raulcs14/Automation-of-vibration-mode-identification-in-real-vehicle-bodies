"""
Visual test for run_static_model.

Run from anywhere:
    py -3 tests/test_static_model.py

Asks interactively which case to inspect individually, then shows all 11.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
from simple_model.analysis.static_model import run_static_model, REF_NAMES
from simple_model.geometry.chassis import _draw_mesh_lines, _set_equal_axes


def plot_deformed(ax, nc, en, u_raw, name, target_frac=0.08):
    """Overlay undeformed (dashed) and deformed (solid) mesh with auto-scale."""
    UX = u_raw[0::6];  UY = u_raw[1::6];  UZ = u_raw[2::6]
    umax = np.sqrt(UX**2 + UY**2 + UZ**2).max()
    bbox_diag = np.linalg.norm(nc.max(axis=0) - nc.min(axis=0))
    scale = np.clip(target_frac * bbox_diag / max(umax, 1e-12), 0.1, 200)

    nc_def = nc + scale * np.column_stack([UX, UY, UZ])
    all_nc = np.vstack([nc, nc_def])

    _draw_mesh_lines(ax, nc, en, linestyle="k--")
    _draw_mesh_lines(ax, nc_def, en, linestyle="r-")
    _set_equal_axes(ax, all_nc)          # also calls invert_zaxis → Z=0 at bottom
    ax.set_title(f"{name}\nscale={scale:.1f}  umax={umax:.2e}", fontsize=8)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=135)
    ax.grid(True)


def ask_case(n_cases):
    """Print menu and return 0-based index chosen by user, or None for all."""
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


def main():
    result  = run_static_model()
    nc      = result["node_coordinates"]
    en      = result["element_nodes"]
    raw     = result["ref_moves_raw"]
    n_cases = raw.shape[1]

    # --- interactive selection ---
    chosen = ask_case(n_cases)

    if chosen is not None:
        fig = plt.figure(figsize=(7, 6))
        ax  = fig.add_subplot(111, projection="3d")
        plot_deformed(ax, nc, en, raw[:, chosen], REF_NAMES[chosen])
        fig.tight_layout()
        plt.show()

    # --- always show the full grid ---
    cols, rows = 4, 3
    fig = plt.figure(figsize=(5*cols, 4*rows))
    fig.suptitle("All reference static shapes", fontsize=12)
    for k in range(n_cases):
        ax = fig.add_subplot(rows, cols, k + 1, projection="3d")
        plot_deformed(ax, nc, en, raw[:, k], REF_NAMES[k])
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
