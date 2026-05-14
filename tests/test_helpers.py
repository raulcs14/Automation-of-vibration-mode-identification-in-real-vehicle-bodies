"""
Shared helpers for interactive visual tests.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from common.interaction import (
    ask_yn, ask_variant, ask_case, ask_mode, ask_weighting,
)

F0_ENERGY = 40.0

# Re-export so existing test scripts that do `from test_helpers import X` keep working
__all__ = ["ask_yn", "ask_variant", "ask_case", "ask_mode", "ask_weighting",
           "plot_deformed", "best_mac_per_mode", "F0_ENERGY"]


def plot_deformed(ax, nc, en, u_raw, name,
                  draw_mesh_fn, set_axes_fn, target_frac=0.08):
    from common.visualization.mesh import plot_deformed as _plot_deformed
    _plot_deformed(ax, nc, en, u_raw, name, draw_mesh_fn, set_axes_fn, target_frac)


def best_mac_per_mode(mac: np.ndarray) -> np.ndarray:
    from common.mac_core import best_mac_per_mode as _best
    return _best(mac)
