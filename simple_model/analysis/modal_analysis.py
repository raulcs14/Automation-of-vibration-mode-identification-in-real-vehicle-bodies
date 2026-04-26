"""
Modal analysis: computes free-free dynamic modes of the chassis.
Equivalent to Modes_Calculator.m
"""

import numpy as np
from pathlib import Path
from scipy.linalg import eigh
from simple_model.geometry.chassis import build_chassis_geometry
from simple_model.fem.stiffness import form_stiffness
from simple_model.fem.mass import form_mass
from common.rigid_body import build_rigid_body_basis
import config

DATA_DIR = Path("data/simple_model")

N_RIGID_BODY_MODES = 6
N_ELASTIC_MODES    = 30


def run_modal_analysis() -> dict:
    """
    Solve K φ = λ M φ (free-free, no prescribed DOFs), discard the first 6
    rigid-body modes, keep N_ELASTIC_MODES elastic modes, and save to
    data/simple_model/dynamic_modes.npz.

    Returns a dict with keys: modes, omega, freq, R, node_coordinates,
    element_nodes, K, M.
    """
    geo = build_chassis_geometry("torsion")
    nc  = geo.node_coordinates
    en  = geo.element_nodes
    GDof = 6 * nc.shape[0]

    K = form_stiffness(geo, config.E, config.G, config.A, config.IY, config.IZ, config.J)
    M = form_mass(geo, config.RHO, config.A, config.IY, config.IZ)

    # Free-free: no prescribed DOFs → solve full system
    eigenvalues, V = eigh(K, M)          # ascending order, M-normalized (φᵀMφ=1)

    # Sort ascending (eigh already does this, but be explicit)
    idx         = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    V           = V[:, idx]

    # Normalize so max translational component = 1 (matches MATLAB visual scale)
    # Translational DOFs: indices 0,1,2 of every 6-DOF block
    trans_rows = np.concatenate([np.arange(d, V.shape[0], 6) for d in range(3)])
    maxabs     = np.abs(V[trans_rows, :]).max(axis=0, keepdims=True)
    maxabs     = np.where(maxabs < 1e-14, 1.0, maxabs)   # avoid division by zero
    modes_all  = V / maxabs

    # Frequencies
    # Clamp small negatives (numerical noise in rigid-body modes) to 0
    omega_all = np.sqrt(np.maximum(eigenvalues, 0.0))
    freq_all  = omega_all / (2 * np.pi)

    print("First 10 frequencies [Hz]:")
    for i in range(min(10, len(freq_all))):
        print(f"  mode {i+1:3d}: {freq_all[i]:.4f} Hz")

    # Discard rigid-body modes (first 6), keep next N_ELASTIC_MODES
    start = N_RIGID_BODY_MODES
    end   = N_RIGID_BODY_MODES + N_ELASTIC_MODES
    modes = modes_all[:, start:end]
    omega = omega_all[start:end]
    freq  = freq_all[start:end]

    print(f"\nElastic modes kept: {modes.shape[1]}")
    print(f"Frequency range: {freq[0]:.2f} – {freq[-1]:.2f} Hz")

    R = build_rigid_body_basis(nc)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(DATA_DIR / "dynamic_modes.npz",
             modes=modes, omega=omega, freq=freq,
             node_coordinates=nc, element_nodes=en,
             GDof=GDof, R=R)
    print(f"Saved to {DATA_DIR}/dynamic_modes.npz")

    return dict(modes=modes, omega=omega, freq=freq,
                node_coordinates=nc, element_nodes=en,
                K=K, M=M, R=R)
