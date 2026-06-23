"""
[VISUAL] Visual view of run_static_model — one static reference shape.

Run from anywhere:
    py -3 scripts/simple_model/view_static_shapes.py

Asks interactively which reference case to inspect and shows that single
deformed shape.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
import _bootstrap  # noqa: F401  -- puts repo root (and scripts/) on sys.path
import numpy as np
import matplotlib.pyplot as plt
from simple_model.analysis.static_model import run_static_model, REF_NAMES
from common.visualization.mesh import (
    _draw_mesh_lines, _set_equal_axes, plot_deformed as _plot_deformed,
)
from _helpers import require_case as _require_case


def plot_deformed(ax, nc, en, u_raw, name, target_frac=0.08):
    _plot_deformed(ax, nc, en, u_raw, name, _draw_mesh_lines, _set_equal_axes, target_frac)


def ask_case(n_cases):
    return _require_case(n_cases, REF_NAMES)


def main():
    result  = run_static_model()
    nc      = result["node_coordinates"]
    en      = result["element_nodes"]
    raw     = result["ref_moves_raw"]
    n_cases = raw.shape[1]

    # --- interactive selection ---
    chosen = ask_case(n_cases)

    fig = plt.figure(figsize=(7, 6))
    ax  = fig.add_subplot(111, projection="3d")
    plot_deformed(ax, nc, en, raw[:, chosen], REF_NAMES[chosen])
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
