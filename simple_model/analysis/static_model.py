"""
Static reference model: generates 11 normalized displacement patterns.
Equivalent to Static_Model.m
"""

import numpy as np
from pathlib import Path
from simple_model.geometry.chassis import build_chassis_geometry
from simple_model.fem.stiffness import form_stiffness
from simple_model.fem.mass import form_mass
from simple_model.fem.solver import inertia_relief
import config

DATA_DIR = Path("data/simple_model")

REF_NAMES = [
    "01 Heave / Global vertical bending (floor down)",
    "02 Pitch (front down, rear up)",
    "03 Lateral bending (+Y)",
    "04 Roll/Torsion global (4 forces)",
    "05 Roll/Torsion front only",
    "06 Roll/Torsion rear only",
    "07 Roof heave / global bending (roof down)",
    "08 Pitch (front floor down, rear roof up)",
    "09 Torsion (front clockwise, back anticlockwise)",
    "10 Combo: roll + heave",
    "11 Forced Torsion (F = KU)",
]

SHORT_NAMES = [
    "Heave",
    "Pitch",
    "Lat. Bending",
    "Torsion",
    "Torsion",
    "Torsion",
    "Heave",
    "Pitch",
    "Torsion",
    "Torsion",
    "Torsion",
]


def run_static_model() -> dict:
    """
    Build chassis, apply 11 load cases, solve via inertia relief,
    normalize each solution, and save to data/simple_model/static_reference_moves.npz.

    Returns a dict with keys: ref_moves, ref_moves_raw, ref_names,
    node_coordinates, element_nodes, K, M.
    """
    geo = build_chassis_geometry("torsion")
    nc  = geo.node_coordinates
    en  = geo.element_nodes
    GDof = 6 * nc.shape[0]

    K = form_stiffness(geo, config.E, config.G, config.A, config.IY, config.IZ, config.J)
    M = form_mass(geo, config.RHO, config.A, config.IY, config.IZ)

    print(f"rcond(K) = {1.0 / np.linalg.cond(K):.3e}")

    # Inertia-relief reference DOFs: all 6 DOFs of the inertia node (0-based → 1-based in comment)
    ir_node = geo.inertia_node          # 0-based
    r_set = np.array([6*ir_node + d for d in range(6)])

    # Helper: 0-based global DOF index
    def dof(n1based, d):               # keeps MATLAB 1-based convention internally
        return 6*(n1based - 1) + (d - 1)

    F0 = 2e6   # base load amplitude [N]

    # Named node aliases (MATLAB 1-based numbering)
    LBF, RBF   =  1,  2   # bumper floor
    LFF, RFF   =  5,  6   # front floor
    LRF, RRF   = 17, 18   # rear floor
    LTF, RTF   = 23, 24   # tail floor
    LBU, RBU   =  3,  4   # bumper upper
    LFU, RFU   =  7,  8   # front belt
    LFUU, RFUU =  9, 10   # front cowl
    LRU, RRU   = 21, 22   # rear roof
    LTU, RTU   = 27, 28   # tail upper
    TLR, TRR   = 21, 22   # rear roof (same as LRU/RRU)
    BLR, BRR   = 19, 20   # rear belt

    cases = []

    # --- Case 1: Heave ---
    f = np.zeros(GDof)
    for n in (LFF, RFF, LBF, RBF, LRF, RRF, LTF, RTF):
        f[dof(n, 3)] = -F0
    cases.append(f)

    # --- Case 2: Pitch ---
    f = np.zeros(GDof)
    for n in (LFF, RFF): f[dof(n, 3)] = -F0
    for n in (LRF, RRF): f[dof(n, 3)] = +F0
    cases.append(f)

    # --- Case 3: Lateral ---
    f = np.zeros(GDof)
    for n in (LFF, RFF, LRF, RRF, LFU, RFU, BLR, BRR,
              LBF, RBF, LTF, RTF, LBU, RBU, LFU, RFU):
        f[dof(n, 2)] = +F0
    cases.append(f)

    # --- Case 4: Roll/Torsion global ---
    f = np.zeros(GDof)
    f[dof(LFF, 3)] = +F0;  f[dof(RFF, 3)] = -F0
    f[dof(LRF, 3)] = -F0;  f[dof(RRF, 3)] = +F0
    cases.append(f)

    # --- Case 5: Front roll only ---
    f = np.zeros(GDof)
    f[dof(LBF,  3)] = -F0;  f[dof(RBF,  2)] = -F0
    f[dof(LBU,  2)] = +F0;  f[dof(RBU,  3)] = +F0
    f[dof(LFF,  3)] = -F0;  f[dof(RFF,  2)] = -F0
    f[dof(LFU,  2)] = +F0;  f[dof(RFU,  3)] = +F0
    f[dof(LFUU, 2)] = +F0;  f[dof(RFUU, 3)] = +F0
    cases.append(f)

    # --- Case 6: Rear roll only ---
    f = np.zeros(GDof)
    f[dof(LRF, 2)] = +F0;  f[dof(RRF, 3)] = -F0
    f[dof(LRU, 3)] = +F0;  f[dof(RRU, 2)] = -F0
    f[dof(LTF, 2)] = +F0;  f[dof(RTF, 3)] = -F0
    f[dof(LTU, 3)] = +F0;  f[dof(RTU, 2)] = -F0
    cases.append(f)

    # --- Case 7: Roof heave ---
    f = np.zeros(GDof)
    for n in (LFUU, RFUU, TLR, TRR, LTU, RTU, LBU, RBU):
        f[dof(n, 3)] = -F0
    cases.append(f)

    # --- Case 8: Pitch roof+floor ---
    f = np.zeros(GDof)
    f[dof(LFF, 3)] = -F0;  f[dof(RFF, 3)] = -F0
    f[dof(TLR, 3)] = +F0;  f[dof(TRR, 3)] = +F0
    cases.append(f)

    # --- Case 9: Torsion roof+floor ---
    f = np.zeros(GDof)
    f[dof(LBF,  3)] = -F0;  f[dof(RBF,  2)] = -F0
    f[dof(LBU,  2)] = +F0;  f[dof(RBU,  3)] = +F0
    f[dof(LFF,  3)] = -F0;  f[dof(RFF,  2)] = -F0
    f[dof(LFU,  2)] = +F0;  f[dof(RFU,  3)] = +F0
    f[dof(LFUU, 2)] = +F0;  f[dof(RFUU, 3)] = +F0
    f[dof(LRF,  2)] = +F0;  f[dof(RRF,  3)] = -F0
    f[dof(LRU,  3)] = +F0;  f[dof(RRU,  2)] = -F0
    f[dof(LTF,  2)] = +F0;  f[dof(RTF,  3)] = -F0
    f[dof(LTU,  3)] = +F0;  f[dof(RTU,  2)] = -F0
    cases.append(f)

    # --- Case 10: Combo roll + heave ---
    alpha, beta = 0.5, 0.75
    f = np.zeros(GDof)
    for n in (LFF, RFF, LRF, RRF):
        f[dof(n, 3)] += beta * (-F0)
    f[dof(LFF, 3)] += alpha * (+F0);  f[dof(RFF, 3)] += alpha * (-F0)
    f[dof(LRF, 3)] += alpha * (+F0);  f[dof(RRF, 3)] += alpha * (-F0)
    cases.append(f)

    # --- Case 11: Forced torsion F = K·φ₁ (requires dynamic modes pre-computed) ---
    # Use a pure torsion load as proxy (same as case 9) when modes are unavailable.
    # If dynamic_modes.npz exists, load φ₁ and compute K·φ₁.
    dyn_file = DATA_DIR / "dynamic_modes.npz"
    if dyn_file.exists():
        D = np.load(dyn_file)
        phi1 = D["modes"][:, 0]
        f = K @ phi1
    else:
        f = cases[8].copy()   # fall back to case 9 torsion
    cases.append(f)

    # Solve all cases
    ref_moves_raw = np.zeros((GDof, len(cases)))
    ref_moves     = np.zeros((GDof, len(cases)))
    for k, f in enumerate(cases):
        u = inertia_relief(K, M, f, r_set)
        ref_moves_raw[:, k] = u
        ref_moves[:, k]     = u / np.linalg.norm(u)
        print(f"  case {k+1:2d}: umax = {np.abs(u).max():.3e}")

    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(DATA_DIR / "static_reference_moves.npz",
             ref_moves=ref_moves,
             ref_moves_raw=ref_moves_raw,
             ref_names=np.array(REF_NAMES),
             node_coordinates=nc,
             element_nodes=en)
    np.savez(DATA_DIR / "matrices.npz", K=K, M=M)
    print(f"Saved to {DATA_DIR}/")

    return dict(ref_moves=ref_moves, ref_moves_raw=ref_moves_raw,
                ref_names=REF_NAMES, node_coordinates=nc,
                element_nodes=en, K=K, M=M)
