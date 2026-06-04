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
from common.visualization.mesh import plot_deformed as _plot_deformed
from test_helpers import ask_case as _ask_case


def plot_deformed(ax, nc, en, u_raw, name, target_frac=0.08):
    _plot_deformed(ax, nc, en, u_raw, name, _draw_mesh_lines, _set_equal_axes, target_frac)


def ask_case(n_cases):
    return _ask_case(n_cases, REF_NAMES)


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
