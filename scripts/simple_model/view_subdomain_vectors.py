"""
[VISUAL] Subdomain-averaged vectors for one elastic mode.

Run from anywhere:
    py -3 scripts/simple_model/view_subdomain_vectors.py

Prints the frequency table, asks which mode to inspect, and shows that single
mode's averaged subdomain vectors.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
import _bootstrap  # noqa: F401  -- puts repo root (and scripts/) on sys.path
import numpy as np
import matplotlib.pyplot as plt
from simple_model.analysis.modal_analysis import run_modal_analysis
from common.subdomain import average_zones
from common.visualization.vectors import plot_subdomain_vectors
from simple_model.geometry.chassis import build_chassis_geometry
from common.utils import translational_dof_indices
from _helpers import require_mode as ask_mode

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

    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")
    plot_subdomain_vectors(ax, nc, en, subdomains, modes_red,
                           scale_factor=SCALE_FACTOR, mode_index=chosen,
                           title=f"Mode {chosen+1}  {freq[chosen]:.2f} Hz — subdomain averages")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
