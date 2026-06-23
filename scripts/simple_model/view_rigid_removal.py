"""
[VISUAL] Rigid-body component removal from a static reference shape.

Run from anywhere:
    py -3 scripts/simple_model/view_rigid_removal.py

Flow
----
  1. Select one reference case to inspect
  2. Plots its deformed mesh before and after rigid-body removal
  3. Ask whether to show its averaged subdomain vectors after removal
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
import _bootstrap  # noqa: F401  -- puts repo root (and scripts/) on sys.path
import numpy as np
import matplotlib.pyplot as plt

from simple_model.analysis.static_model   import run_static_model, REF_NAMES
from simple_model.analysis.modal_analysis import run_modal_analysis
from simple_model.geometry.chassis        import build_chassis_geometry
from common.visualization.mesh            import _draw_mesh_lines, _set_equal_axes
from common.rigid_body                    import remove_rigid_body_component
from common.subdomain                     import average_zones
from common.visualization.vectors         import plot_subdomain_vectors
from common.utils                         import translational_dof_indices
from common.visualization.mesh import plot_deformed as _plot_deformed
from _helpers               import require_case as _require_case, ask_yn


def ask_case(n_cases):
    return _require_case(n_cases, REF_NAMES)


def plot_deformed(ax, nc, en, u_raw, name, target_frac=0.08):
    _plot_deformed(ax, nc, en, u_raw, name, _draw_mesh_lines, _set_equal_axes, target_frac)


def show_single(nc, en, u_before, u_after, name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             subplot_kw={"projection": "3d"})
    plot_deformed(axes[0], nc, en, u_before, f"Original\n{name}", target_frac=0.08)
    plot_deformed(axes[1], nc, en, u_after,  f"Rigid removed\n{name}", target_frac=0.08)
    axes[0].title.set_fontsize(9)
    axes[1].title.set_fontsize(9)
    fig.suptitle(name, fontsize=10)
    fig.tight_layout()


def main():
    stat = run_static_model()
    dyn  = run_modal_analysis()

    nc      = stat["node_coordinates"]
    en      = stat["element_nodes"]
    raw     = stat["ref_moves_raw"]   # (GDof, nRefs)
    M       = dyn["M"]
    R       = dyn["R"]
    n_cases = raw.shape[1]
    GDof    = raw.shape[0]
    n_nodes = GDof // 6

    geo = build_chassis_geometry("torsion")
    subdomains = geo.subdomains

    # Remove rigid-body component from all cases at once
    raw_proj = remove_rigid_body_component(raw, M, R)

    # --- Ask user -----------------------------------------------------------
    chosen = ask_case(n_cases)
    show_single(nc, en, raw[:, chosen], raw_proj[:, chosen], REF_NAMES[chosen])

    # --- Average zones option -----------------------------------------------
    use_zones = ask_yn("\nShow averaged subdomain vectors after rigid removal?")
    if use_zones:
        t_idx   = translational_dof_indices(GDof)
        proj_t  = raw_proj[t_idx, :]
        vec_red = average_zones(proj_t, subdomains, n_nodes)

        fig = plt.figure(figsize=(9, 7))
        ax  = fig.add_subplot(111, projection="3d")
        plot_subdomain_vectors(ax, nc, en, subdomains, vec_red,
                               scale_factor=5.0, mode_index=chosen,
                               title=f"Rigid removed — {REF_NAMES[chosen]}")
        fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
