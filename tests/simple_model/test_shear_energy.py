"""
MAC based on shear internal forces — simple model.

Run from anywhere:
    py -3 tests/simple_model/test_shear_energy.py

Computes the MAC matrix between dynamic modes and static reference shapes
projected onto the shear-force space (Fy, Fz at both element nodes).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt

from simple_model.analysis.modal_analysis import run_modal_analysis
from simple_model.analysis.static_model   import run_static_model, SHORT_NAMES, REF_NAMES
from simple_model.geometry.chassis        import build_chassis_geometry
from simple_model.fem.internal_forces     import build_shear_projection_matrix
from common.mac_core                      import compute_mac
from common.visualization.mac_plot        import plot_mac_matrix
import config


def main():
    dyn  = run_modal_analysis()
    stat = run_static_model()
    geo  = build_chassis_geometry("torsion")

    modes = dyn["modes"]
    freq  = dyn["freq"]
    refs  = stat["ref_moves_raw"]

    B       = build_shear_projection_matrix(
        geo, config.E, config.G, config.A, config.IY, config.IZ, config.J
    )
    Phi_tau = B @ modes
    Psi_tau = B @ refs

    mac = compute_mac(Phi_tau, Psi_tau)

    # --- Print table ---------------------------------------------------------
    best_j   = mac.argmax(axis=1)
    best_val = mac.max(axis=1)

    print("\n=== MAC shear — simple model ===")
    print(f"  {'Mode':<20}  {'Best reference':<20}  {'MAC':>5}")
    print("  " + "-" * 50)
    for i, f in enumerate(freq):
        ref_name = SHORT_NAMES[best_j[i]]
        print(f"  Mode {i+1:2d} ({f:6.1f} Hz)   {ref_name:<20}  {best_val[i]:.3f}")

    # --- Plot MAC heatmap ----------------------------------------------------
    mode_labels = [f"{i+1} ({f:.0f}Hz)" for i, f in enumerate(freq)]
    plot_mac_matrix(mac, mode_labels, SHORT_NAMES, title="MAC shear — simple model")
    plt.show()


if __name__ == "__main__":
    main()
