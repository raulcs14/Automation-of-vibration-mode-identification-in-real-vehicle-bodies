"""
Visual test for run_modal_analysis.

Run from anywhere:
    py -3 tests/test_modal_analysis.py

Prints the frequency table, asks which mode to inspect individually,
then shows the full grid of all 30 elastic modes.
"""

import sys
from pathlib import Path
# Make project root and tests/ importable so this file runs under pytest
# AND when executed directly (IDE Run button).  Walk up to the repo root
# (dir containing the `common` package); _helpers lives in tests/.
_here = Path(__file__).resolve()
for _root in _here.parents:
    if (_root / "common").is_dir() and (_root / "main.py").is_file():
        break
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "tests"))

import numpy as np
import matplotlib.pyplot as plt
from simple_model.analysis.modal_analysis import run_modal_analysis
from common.visualization.mesh import (
    _draw_mesh_lines, _set_equal_axes, draw_interpolated_frame,
)
from _helpers import ask_mode

SCALE = 0.5   # same as MATLAB scaleFact = 0.5


def plot_mode(ax, nc, en, mode, freq_hz, mode_number, fontsize=8):
    UX = mode[0::6];  UY = mode[1::6];  UZ = mode[2::6]
    nc_def = nc + SCALE * np.column_stack([UX, UY, UZ])

    # MATLAB order: deformed mesh (solid dots), undeformed (dashed), Hermite curves
    _draw_mesh_lines(ax, nc_def, en, linestyle="k-", marker=".")
    _draw_mesh_lines(ax, nc,     en, linestyle="k--")
    draw_interpolated_frame(ax, nc, en, mode, scale=SCALE, color="r", linewidth=1.5)

    _set_equal_axes(ax, np.vstack([nc, nc_def]))
    ax.set_title(f"Mode {mode_number}  {freq_hz:.2f} Hz", fontsize=fontsize)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.tick_params(labelsize=fontsize)
    ax.view_init(elev=20, azim=135)
    ax.grid(True)


def main():
    result = run_modal_analysis()
    nc     = result["node_coordinates"]
    en     = result["element_nodes"]
    modes  = result["modes"]
    freq   = result["freq"]
    n_modes = modes.shape[1]

    chosen = ask_mode(n_modes, freq)

    if chosen is not None:
        fig = plt.figure(figsize=(8, 7))
        ax  = fig.add_subplot(111, projection="3d")
        plot_mode(ax, nc, en, modes[:, chosen], freq[chosen], chosen + 1, fontsize=14)
        fig.tight_layout()
        plt.show()

    # Full grid — 6 cols × 5 rows = 30 modes
    cols, rows = 6, 5
    fig = plt.figure(figsize=(4*cols, 3.5*rows))
    fig.suptitle("Elastic modes (free-free)", fontsize=13)
    for k in range(n_modes):
        ax = fig.add_subplot(rows, cols, k + 1, projection="3d")
        plot_mode(ax, nc, en, modes[:, k], freq[k], k + 1)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
