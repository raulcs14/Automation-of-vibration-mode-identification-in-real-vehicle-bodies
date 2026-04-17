"""
Visual test for energy lost after rigid-body component removal.

Run from anywhere:
    py -3 tests/test_rigid_body_energy.py

For each vector (dynamic modes + static reference shapes) computes:

    energy_lost% = 100 * (uᵀMu - u_projᵀM u_proj) / (uᵀMu)

where the norm uses translational DOFs only (consistent with the MAC pipeline).

Dynamic modes should show ~0% loss (free-free eigh already orthogonal to rigid body).
Static reference shapes will show non-zero loss (inertia relief introduces rigid components).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt

from simple_model.analysis.static_model   import run_static_model, REF_NAMES
from simple_model.analysis.modal_analysis import run_modal_analysis
from common.rigid_body                    import remove_rigid_body_component


def translational_dof_indices(gdof: int) -> np.ndarray:
    return np.concatenate([np.arange(d, gdof, 6) for d in range(3)])


def mass_norm(V: np.ndarray, M: np.ndarray) -> np.ndarray:
    """(nVectors,) array of uᵀMu for each column of V."""
    return np.einsum("ij,ij->j", V, M @ V)


def energy_lost_pct(V: np.ndarray, V_proj: np.ndarray, M: np.ndarray) -> np.ndarray:
    e_before = mass_norm(V, M)
    e_after  = mass_norm(V_proj, M)
    lost = np.where(e_before > 0, 100.0 * (e_before - e_after) / e_before, 0.0)
    return lost


def main():
    dyn  = run_modal_analysis()
    stat = run_static_model()

    modes = dyn["modes"]
    freq  = dyn["freq"]
    M     = dyn["M"]
    R     = dyn["R"]
    ref   = stat["ref_moves_raw"]
    GDof  = modes.shape[0]

    t_idx = translational_dof_indices(GDof)
    M_t   = M[np.ix_(t_idx, t_idx)]

    # Project out rigid body
    modes_proj = remove_rigid_body_component(modes, M, R)
    ref_proj   = remove_rigid_body_component(ref,   M, R)

    # Energy loss on translational DOFs only
    lost_modes = energy_lost_pct(modes[t_idx, :], modes_proj[t_idx, :], M_t)
    lost_ref   = energy_lost_pct(ref[t_idx, :],   ref_proj[t_idx, :],   M_t)

    # --- Print summary ------------------------------------------------------
    print("\n=== Energy lost after rigid-body removal ===")
    print("\nDynamic modes:")
    for i, (f, pct) in enumerate(zip(freq, lost_modes)):
        print(f"  Mode {i+1:2d} ({f:7.2f} Hz):  {pct:.4f} %")

    print("\nStatic reference shapes:")
    for j, (name, pct) in enumerate(zip(REF_NAMES, lost_ref)):
        print(f"  {name:<45s}  {pct:.2f} %")

    # --- Plot ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Dynamic modes
    ax = axes[0]
    mode_labels = [f"{i+1}\n({f:.0f} Hz)" for i, f in enumerate(freq)]
    bars = ax.bar(range(len(lost_modes)), lost_modes, color="steelblue")
    ax.set_xticks(range(len(lost_modes)))
    ax.set_xticklabels(mode_labels, fontsize=6)
    ax.set_xlabel("Dynamic mode")
    ax.set_ylabel("Energy lost [%]")
    ax.set_title("Dynamic modes — energy lost after rigid-body removal")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y")
    for bar, pct in zip(bars, lost_modes):
        if pct > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{pct:.2f}%", ha="center", va="bottom", fontsize=6)

    # Static reference shapes
    ax = axes[1]
    ref_labels = [f"{j+1}" for j in range(len(lost_ref))]
    bars = ax.bar(range(len(lost_ref)), lost_ref, color="tomato")
    ax.set_xticks(range(len(lost_ref)))
    ax.set_xticklabels(ref_labels, fontsize=8)
    ax.set_xlabel("Reference case")
    ax.set_ylabel("Energy lost [%]")
    ax.set_title("Static reference shapes — energy lost after rigid-body removal")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y")
    for bar, pct in zip(bars, lost_ref):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=7)

    # Reference case legend below plot
    legend_text = "\n".join(f"{j+1}: {name}" for j, name in enumerate(REF_NAMES))
    fig.text(0.55, -0.02, legend_text, fontsize=6, va="top", family="monospace")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
