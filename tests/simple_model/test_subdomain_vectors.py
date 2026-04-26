"""
Visual test for subdomain averaging and vector plotting.

Run from anywhere:
    py -3 tests/test_subdomain_vectors.py

Prints the frequency table, asks which mode to inspect (or all),
then shows one figure per selected mode with the averaged subdomain vectors.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
from simple_model.analysis.modal_analysis import run_modal_analysis
from common.subdomain import average_zones
from common.visualization.vectors import plot_subdomain_vectors
from simple_model.geometry.chassis import build_chassis_geometry
from common.utils import translational_dof_indices
from test_helpers import ask_mode

SCALE_FACTOR = 5.0


def main():
    result = run_modal_analysis()
    nc     = result["node_coordinates"]
    en     = result["element_nodes"]
    modes  = result["modes"]
    freq   = result["freq"]
    n_modes = modes.shape[1]

    geo = build_chassis_geometry("torsion")
    subdomains = geo.subdomains
    n_nodes = nc.shape[0]

    # Translational DOFs only
    GDof = modes.shape[0]
    t_idx = translational_dof_indices(GDof)
    modes_t = modes[t_idx, :]

    # Compute reduced modes: (3·nZones, nModes)
    modes_red = average_zones(modes_t, subdomains, n_nodes)

    chosen = ask_mode(n_modes, freq)

    indices = [chosen] if chosen is not None else list(range(n_modes))

    if chosen is not None:
        fig = plt.figure(figsize=(9, 7))
        ax  = fig.add_subplot(111, projection="3d")
        plot_subdomain_vectors(ax, nc, en, subdomains, modes_red,
                               scale_factor=SCALE_FACTOR, mode_index=chosen,
                               title=f"Mode {chosen+1}  {freq[chosen]:.2f} Hz — subdomain averages")
        fig.tight_layout()
        plt.show()

    # Full grid — 6 cols × 5 rows
    cols, rows = 6, 5
    fig = plt.figure(figsize=(4*cols, 3.5*rows))
    fig.suptitle("Subdomain-averaged vectors — elastic modes (free-free)", fontsize=13)
    for k in range(n_modes):
        ax = fig.add_subplot(rows, cols, k + 1, projection="3d")
        plot_subdomain_vectors(ax, nc, en, subdomains, modes_red,
                               scale_factor=SCALE_FACTOR, mode_index=k,
                               title=f"Mode {k+1}  {freq[k]:.2f} Hz")
        ax.tick_params(labelsize=6)
        ax.title.set_fontsize(7)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
