"""
[VISUAL] Visual inspection of shell subdomains (PID zones) for BIW or TB.

Node cloud is coloured by subdomain; 2-D projections are also shown.

Works for BIW and TB: PID zones are taken from the META subdomains.json if
present, otherwise read straight from the modal H5 element connectivity
(read_hdf5_pid_subdomains), so no separate export is needed.

Run from anywhere:
    py -3 scripts/seat/view_subdomains.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
import _bootstrap  # noqa: F401  -- puts repo root (and scripts/) on sys.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from seat_model.modal_analysis import run_modal_analysis
from seat_model.subdomains     import build_subdomains, _H5_MODAL
from seat_model.reader         import (
    read_hdf5_conm2_node_ids, read_hdf5_rbe_node_ids,
)


def _classify_uncovered(uncovered_ids: np.ndarray, variant: str) -> dict:
    """
    Split the GRID IDs that fall outside every PID zone by what they are.

    A node has no PID only if it belongs to no shell/bar element.  Such nodes
    are either lumped masses (CONM2) or connector grids (RBE2/RBE3); anything
    left is a free/orphan grid not attached to any element.  Categories are
    exclusive in this priority order: mass -> connector -> orphan.

    Returns {"mass": [...], "connector": [...], "orphan": [...]}.
    """
    h5 = _H5_MODAL.get(variant)
    uncovered = set(int(n) for n in uncovered_ids)
    if h5 is None or not h5.exists():
        return {"mass": [], "connector": [], "orphan": sorted(uncovered)}

    conm2 = set(int(x) for x in read_hdf5_conm2_node_ids(h5))
    rbe   = set(int(x) for x in read_hdf5_rbe_node_ids(h5))

    mass      = uncovered & conm2
    connector = (uncovered & rbe) - mass
    orphan    = uncovered - mass - connector
    return {"mass": sorted(mass), "connector": sorted(connector), "orphan": sorted(orphan)}


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


def _plot(node_xyz: np.ndarray, subdomains: dict, variant: str,
          node_ids: np.ndarray) -> None:
    zone_names = list(subdomains.keys())
    n_zones    = len(zone_names)
    n_nodes    = len(node_xyz)

    cmap = plt.colormaps["tab20"].resampled(n_zones)

    # Assign colour: first zone wins for shared-edge nodes
    node_zone = np.full(n_nodes, -1, dtype=int)
    for zi, name in enumerate(zone_names):
        for idx in subdomains[name]:
            if node_zone[idx] == -1:
                node_zone[idx] = zi

    assigned = node_zone >= 0
    colors = np.tile([0.75, 0.75, 0.75, 0.15], (n_nodes, 1))
    colors[assigned] = [cmap(node_zone[i]) for i in np.where(assigned)[0]]

    n_unassigned = int((~assigned).sum())
    if n_unassigned:
        uncovered_ids = np.asarray(node_ids)[~assigned]
        cats = _classify_uncovered(uncovered_ids, variant)
        print(f"  WARNING: {n_unassigned} nodes not covered by any PID zone "
              f"(no shell/bar element):")
        labels = [
            ("mass nodes (CONM2)",    len(cats["mass"])),
            ("connector nodes (RBE)", len(cats["connector"])),
            ("orphan grids (no elem)", len(cats["orphan"])),
        ]
        for name, count in labels:
            if count:
                print(f"    - {name:<22}: {count}")

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
    # ~25 entries per column so many-zone models (e.g. TB, 63 PIDs) wrap into
    # several readable columns instead of one long overflowing list.
    legend_ncol = max(1, int(np.ceil(n_zones / 25)))
    ax_leg.legend(
        handles=patches, loc="center left",
        fontsize=6, ncol=legend_ncol,
        title="Zone (PID)", title_fontsize=7, frameon=False,
        columnspacing=1.0, handletextpad=0.4,
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

    _plot(node_xyz, subdomains, variant, node_ids)
    plt.show()


if __name__ == "__main__":
    main()
