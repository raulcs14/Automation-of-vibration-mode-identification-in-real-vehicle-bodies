"""
Visual inspection of shell subdomains (PID zones) for BIW or TB.

Node cloud is coloured by subdomain; 2-D projections are also shown.

Run from anywhere:
    py -3 tests/SEAT/test_visualize_subdomains_biw.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from seat_model.modal_analysis import run_modal_analysis
from seat_model.subdomains     import build_subdomains


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ask_variant() -> str:
    print("\nSelect model variant:")
    print("  1 — BIW (Body in White)")
    print("  2 — TB  (Trimmed Body)")
    while True:
        choice = input("Choice [1/2]: ").strip()
        if choice in ("1", "2"):
            return {"1": "BIW", "2": "TB"}[choice]
        print("  Please enter 1 or 2.")


def _set_equal_axes(ax, xyz: np.ndarray) -> None:
    mins, maxs = xyz.min(axis=0), xyz.max(axis=0)
    center = (mins + maxs) / 2
    half   = (maxs - mins).max() / 2 * 0.6
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def _plot(node_xyz: np.ndarray, subdomains: dict, variant: str) -> None:
    zone_names = list(subdomains.keys())
    n_zones    = len(zone_names)
    n_nodes    = len(node_xyz)

    cmap = plt.cm.get_cmap("tab20", n_zones)

    # Assign colour: first zone wins for shared-edge nodes
    node_zone = np.full(n_nodes, -1, dtype=int)
    for zi, name in enumerate(zone_names):
        for idx in subdomains[name]:
            if node_zone[idx] == -1:
                node_zone[idx] = zi

    assigned = node_zone >= 0
    colors = np.tile([0.75, 0.75, 0.75, 0.15], (n_nodes, 1))
    colors[assigned] = [cmap(node_zone[i]) for i in np.where(assigned)[0]]

    n_unassigned = (~assigned).sum()
    if n_unassigned:
        print(f"  WARNING: {n_unassigned} nodes not covered by any zone")

    # --- Figure 1: 3-D view + legend ----------------------------------------
    fig = plt.figure(figsize=(14, 7))
    ax3d = fig.add_subplot(121, projection="3d")
    ax3d.scatter(
        node_xyz[:, 0], node_xyz[:, 1], node_xyz[:, 2],
        c=colors, s=2.5, depthshade=False,
    )
    _set_equal_axes(ax3d, node_xyz)
    ax3d.set_title(f"{variant} subdomains by PID  ({n_zones} zones)", fontsize=9)
    ax3d.set_xlabel("X [mm]", fontsize=7)
    ax3d.set_ylabel("Y [mm]", fontsize=7)
    ax3d.set_zlabel("Z [mm]", fontsize=7)
    ax3d.tick_params(labelsize=6)
    ax3d.view_init(elev=20, azim=210)

    ax_leg = fig.add_subplot(122)
    ax_leg.axis("off")
    patches = [
        mpatches.Patch(
            color=cmap(zi),
            label=f"{name}  ({len(subdomains[name])} nodes)"
        )
        for zi, name in enumerate(zone_names)
    ]
    ax_leg.legend(
        handles=patches, loc="center left",
        fontsize=6, ncol=max(1, n_zones // 35),
        title="Zone (PID)", title_fontsize=7, frameon=False,
    )
    fig.tight_layout()

    # --- Figure 2: 2-D projections ------------------------------------------
    fig2, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (title, xi, yi, xl, yl) in zip(axes, [
        ("Top (X–Y)",   0, 1, "X [mm]", "Y [mm]"),
        ("Side (X–Z)",  0, 2, "X [mm]", "Z [mm]"),
        ("Front (Y–Z)", 1, 2, "Y [mm]", "Z [mm]"),
    ]):
        ax.scatter(node_xyz[:, xi], node_xyz[:, yi], c=colors, s=1.5, linewidths=0)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xl, fontsize=7)
        ax.set_ylabel(yl, fontsize=7)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)
        ax.grid(True, linewidth=0.3)

    fig2.suptitle(
        f"{variant} subdomains — 2-D projections  ({n_zones} PID zones)", fontsize=10
    )
    fig2.tight_layout()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    variant = _ask_variant()

    print(f"\nLoading {variant} modal data...")
    dyn      = run_modal_analysis(variant)
    node_ids = dyn["node_ids"]
    node_xyz = dyn["node_coordinates"]

    print("Building subdomains...")
    try:
        subdomains = build_subdomains(variant, node_ids, node_xyz)
    except FileNotFoundError as e:
        print(f"\n[{variant} subdomains] {e}")
        return
    print(f"  {len(subdomains)} zones")

    # Console summary
    print(f"\n  {'Zone':<12}  {'Nodes':>6}")
    print("  " + "-" * 22)
    for name in subdomains:
        print(f"  {name:<12}  {len(subdomains[name]):>6}")
    print(f"\n  Unique assigned nodes : {sum(1 for v in subdomains.values() for _ in v)}")

    _plot(node_xyz, subdomains, variant)
    plt.show()


if __name__ == "__main__":
    main()
