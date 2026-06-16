"""
Visual test for build_chassis_geometry.

Run from anywhere:
    py -3 tests/test_chassis_geometry.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
from simple_model.geometry.chassis import build_chassis_geometry
from common.visualization.mesh import plot_chassis, plot_chassis_numbered


def main():
    for movement in ("torsion", "bending"):
        print(f"\n--- movement={movement} ---")
        geo = build_chassis_geometry(movement)
        print(f"  nodes    : {geo.node_coordinates.shape}")
        print(f"  elements : {geo.element_nodes.shape}")
        print(f"  inertia node (0-based): {geo.inertia_node}  "
              f"→ coords {geo.node_coordinates[geo.inertia_node]}")
        print(f"  subdomains: {list(geo.subdomains.keys())}")

    # Use torsion variant for the plots
    geo = build_chassis_geometry("torsion")

    fig1 = plot_chassis(geo, view=(135, 20))
    fig1.suptitle("plot_chassis  (plain mesh, torsion)", fontsize=11)

    fig2 = plot_chassis_numbered(geo, view=(170, -45))
    fig2.suptitle("plot_chassis_numbered  (1-based labels, torsion)", fontsize=11)

    plt.show()


if __name__ == "__main__":
    main()
