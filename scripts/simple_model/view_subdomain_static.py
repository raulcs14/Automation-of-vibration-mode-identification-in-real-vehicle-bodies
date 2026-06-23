"""
[VISUAL] Subdomain-averaged vectors of the static reference model — one case.

Run from anywhere:
    py -3 scripts/simple_model/view_subdomain_static.py

Prints the case list, asks which reference case to inspect, and shows that
single set of subdomain-averaged vectors.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
import _bootstrap  # noqa: F401  -- puts repo root (and scripts/) on sys.path
import numpy as np
import matplotlib.pyplot as plt
from simple_model.analysis.static_model import run_static_model, REF_NAMES
from simple_model.geometry.chassis import build_chassis_geometry
from common.subdomain import average_zones
from common.visualization.vectors import plot_subdomain_vectors
from common.utils import translational_dof_indices
from _helpers import require_case as _require_case

SCALE_FACTOR = 5.0


def ask_case(n_cases):
    return _require_case(n_cases, REF_NAMES)


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
    t_idx = translational_dof_indices(GDof)
    raw_t = raw[t_idx, :]

    # Reduce to (3·nZones, n_cases)
    moves_red = average_zones(raw_t, subdomains, n_nodes)

    chosen = ask_case(n_cases)

    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")
    plot_subdomain_vectors(ax, nc, en, subdomains, moves_red,
                           scale_factor=SCALE_FACTOR, mode_index=chosen,
                           title=REF_NAMES[chosen])
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
