"""
Visual test for subdomain-averaged vectors of the static reference model.

Run from anywhere:
    py -3 tests/test_subdomain_vectors_static.py

Prints the case list, asks which to inspect individually,
then shows the full grid of all 11 cases.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
from simple_model.analysis.static_model import run_static_model, REF_NAMES
from simple_model.geometry.chassis import build_chassis_geometry
from common.subdomain import average_zones
from common.visualization.vectors import plot_subdomain_vectors

SCALE_FACTOR = 5.0


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


def main():
    result  = run_static_model()
    nc      = result["node_coordinates"]
    en      = result["element_nodes"]
    raw     = result["ref_moves_raw"]   # (GDof, n_cases)
    n_cases = raw.shape[1]

    geo = build_chassis_geometry("torsion")
    subdomains = geo.subdomains
    n_nodes = nc.shape[0]

    # Translational DOFs only
    GDof = raw.shape[0]
    t_idx = np.concatenate([np.arange(d, GDof, 6) for d in range(3)])
    raw_t = raw[t_idx, :]

    # Reduce to (3·nZones, n_cases)
    moves_red = average_zones(raw_t, subdomains, n_nodes)

    chosen = ask_case(n_cases)

    if chosen is not None:
        fig = plt.figure(figsize=(9, 7))
        ax  = fig.add_subplot(111, projection="3d")
        plot_subdomain_vectors(ax, nc, en, subdomains, moves_red,
                               scale_factor=SCALE_FACTOR, mode_index=chosen,
                               title=REF_NAMES[chosen])
        fig.tight_layout()
        plt.show()

    # Full grid — 4 cols × 3 rows
    cols, rows = 4, 3
    fig = plt.figure(figsize=(5*cols, 4*rows))
    fig.suptitle("Subdomain-averaged vectors — static reference cases", fontsize=12)
    for k in range(n_cases):
        ax = fig.add_subplot(rows, cols, k + 1, projection="3d")
        plot_subdomain_vectors(ax, nc, en, subdomains, moves_red,
                               scale_factor=SCALE_FACTOR, mode_index=k,
                               title=REF_NAMES[k])
        ax.tick_params(labelsize=6)
        ax.title.set_fontsize(7)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
