# Automation of Vibration Mode Identification in Real Vehicle Bodies

TFM project implementing an automated pipeline for identifying and classifying vibration modes of vehicle body structures using Finite Element Analysis (FEA) and Modal Assurance Criterion (MAC).

## Overview

The pipeline correlates dynamic vibration modes (obtained from FEA or experimental data) against a library of known static reference deformation patterns. This allows automatic labelling of modes as heave, pitch, torsion, roll, etc.

Two models are supported:

- **Simple model** — analytical 3D beam-frame chassis (35 nodes, ~80 elements) assembled from scratch in Python/NumPy.
- **ANSA model** — full vehicle FE model solved in ANSA/Meta, results imported for MAC analysis.

## Project Structure

```
├── main.py                    # Pipeline entry point
├── config.py                  # Global material and section properties
│
├── simple_model/              # Analytical beam-frame chassis
│   ├── geometry/
│   │   └── chassis.py         # Node coordinates, element connectivity, subdomains
│   ├── fem/
│   │   ├── stiffness.py       # Global stiffness matrix K
│   │   ├── mass.py            # Global consistent mass matrix M
│   │   └── solver.py          # Eigenvalue solver + inertia relief static solver
│   └── analysis/
│       ├── static_model.py    # 11 reference load cases → normalized displacements
│       ├── modal_analysis.py  # Free-free modal analysis → elastic modes
│       └── mac.py             # MAC correlation (mass- and stiffness-weighted)
│
├── ansa_model/                # ANSA/Meta full vehicle model
│   ├── reader.py              # Parse ANSA/Meta output files
│   ├── modal_analysis.py      # Post-process imported modes
│   └── mac.py                 # MAC correlation for the full model
│
├── common/                    # Shared utilities (model-agnostic)
│   ├── mac_core.py            # MAC formula with arbitrary weighting matrix
│   ├── subdomain.py           # Subdomain averaging and Galerkin reduction
│   ├── rigid_body.py          # Rigid-body component removal
│   └── visualization/
│       ├── mesh.py            # Mesh + Hermite-interpolated deformed shape
│       ├── mac_plot.py        # MAC heatmap
│       ├── vectors.py         # Node and subdomain vector arrows
│       └── animation.py       # Mode shape animation
│
└── data/                      # Generated output files (git-ignored)
    ├── simple_model/
    └── ansa_model/
```

## Pipeline

```
1. Static_Model      →  11 normalized reference displacement patterns
2. Modal_Analysis    →  30 elastic free-free vibration modes + frequencies
3. MAC_Analysis      →  MAC matrices (full + subdomain), correlation ranking, heatmaps
4. Animation         →  (optional) animated mode shape visualization
```

Steps 1–2 are independent and can run in either order. Step 3 requires both outputs.

## FEM Formulation

| Property | Value |
|---|---|
| Element type | 3D Euler–Bernoulli beam (12 DOF) |
| DOFs per node | 6 (Ux, Uy, Uz, Rx, Ry, Rz) |
| Nodes / Elements (simple model) | 35 / ~80 |
| Total DOFs | 210 |
| Boundary conditions | Free-free (inertia relief) |
| Mass formulation | Consistent |

Material properties (steel): E = 210 GPa, G = 84 GPa, ρ = 7850 kg/m³.

## MAC Definition

$$\text{MAC}(i,j) = \frac{|\varphi_i^T W \psi_j|^2}{(\varphi_i^T W \varphi_i)(\psi_j^T W \psi_j)}$$

where **φ** are dynamic modes, **ψ** are static references, and **W** is a weighting matrix (identity, mass M, stiffness K, or energy M·ω²+K).

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

Results are saved to `data/simple_model/` as `.npz` files.

## References

- Ferreira, A.J.M. & Fantuzzi, N. (2019). *MATLAB Codes for Finite Element Analysis*. Springer.
- Allemang, R.J. (2003). The Modal Assurance Criterion — Twenty Years of Use and Abuse. *Sound and Vibration*.
