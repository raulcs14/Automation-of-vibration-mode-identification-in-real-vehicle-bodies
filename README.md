# Automation of Vibration Mode Identification in Real Vehicle Bodies

TFM project implementing an automated pipeline for identifying and classifying vibration modes of vehicle body structures using FEA, Modal Assurance Criterion (MAC), and geometric torsion scoring.

Two models are supported:

- **Simple model** — analytical 3D beam-frame chassis built in Python/NumPy. No external files required, runs out of the box.
- **SEAT model** — full vehicle FE model (Body in White and Trimmed Body) solved in ANSA/Epilysis. Requires HDF5 result files and a local paths configuration.

---

## Installation

```bash
pip install numpy scipy matplotlib h5py
```

---

## Quick start — Simple model

No configuration needed. Just run:

```bash
python main.py
```

Select **Simple model** in the menu and choose a flow (MAC or Torsion ID).

---

## Setup — SEAT model (BIW / TB)

The SEAT model reads results produced by ANSA/Epilysis. Follow these steps before running.

### 1. Configure local paths

```bash
cp seat_model/epilysis_runner/config/paths.py.example \
   seat_model/epilysis_runner/config/paths.py
```

Open `paths.py` and set `EPILYSIS_EXE` to the path of your local Epilysis executable:

```python
EPILYSIS_EXE = Path(r"C:\BETA_CAE_Systems\ansa_vXX.X.X\epilysis.bat")
```

`paths.py` is git-ignored — each user keeps their own copy.

### 2. Place the ANSA input files

Copy your `.dat` deck files into the corresponding input folders:

```
data/seat_model/
├── BIW/ansa/
│   ├── modal/input/       ← 000_Header_BIW_modal.dat
│   ├── static/input/      ← 000_Header_BIW_static_reference.dat
│   └── matrices/input/    ← 000_Header_BIW_getKM.dat
└── TB/ansa/
    ├── modal/input/       ← 000_Header_TB_modal.dat
    ├── static/input/      ← 000_Header_TB_static_reference.dat
    └── matrices/input/    ← 000_Header_TB_getKM.dat
```

### 3. Run the Epilysis analyses

```bash
python seat_model/epilysis_runner/run_analyses.py
```

This launches the modal, static, and matrix extraction analyses and writes the HDF5 result files to the corresponding `output/` folders.

### 4. Run the pipeline

```bash
python main.py
```

Select **ANSA BIW** or **ANSA TB** and choose a flow.

---

## Flows

### MAC — correlation with static reference shapes

Correlates each dynamic mode against a set of static reference deformation patterns (heave, pitch, torsion, roll, etc.) using the MAC formula:

$$\text{MAC}(i,j) = \frac{|\varphi_i^T W \psi_j|^2}{(\varphi_i^T W \varphi_i)(\psi_j^T W \psi_j)}$$

Available weightings: **Identity**, **Mass (M)**, **Stiffness (K)**, **Energy (Mω² + K)**.

Optional: remove the rigid-body component from the reference shapes before computing MAC.

For the SEAT model, CONM2 mass nodes can be excluded from the DOF space before correlation.

### Torsion ID — geometric torsion identification (SEAT only)

Identifies torsional modes without reference shapes. The ranking score combines
one physical fingerprint with two soft-gated quality conditions and a local-mode veto:

| Sub-score | What it measures |
|---|---|
| **antisym** | Lever-arm-aware rigid-rotation fit `rigid_uz` (R² of Uz vs θ·Y); drives the ranking |
| **linearity** | The θx(X) rotation profile is linear with significant amplitude |
| **centering** | Rotation centre x₀ is near the geometric centre of the vehicle |
| **local_veto** | 0 if the mode parks >60 % of its energy in the hottest 1 % of nodes (local mode), else 1 |

**Combined score** = antisym × gate(linearity) × gate(centering) × local_veto ∈ [0, 1]

`antisym` is left ungated (full [0, 1] range) so the best torsion mode stands out;
`linearity` and `centering` enter as smooth sigmoid gates (see `_soft_gate`) rather
than raw factors, keeping borderline torsion modes while still penalising poor quality.
`uniformity` (Shannon-entropy spread) is still computed for diagnostics but no longer
enters the score — localisation is handled by the `peak`/`local_veto` term instead.

A rigid rotation about X leaves two independent antisymmetric fingerprints
(U_z = +θx·Y, U_y = −θx·Z), used for classification:

| Score | Definition | +1 means |
|---|---|---|
| **score_lr** | −corr(Uz_left, Uz_right) | torsion (lateral zones) / −1 = vertical bending |
| **score_tb** | −corr(Uy_top, Uy_bottom) | torsion (upper/lower zones) / −1 = rigid roll |
| **score_ly** | −corr(Uy_left, Uy_right) | lateral antisymmetry |
| **score_xvar** | variation of per-slice mean U_y along X | distinguishes roll (≈0) from lateral bending (large) |

Mode classification:

| Condition | Type |
|---|---|
| score_lr > 0.5 | TORSION (lateral U_z antisymmetry — the reliable rotation signature) |
| score_lr < −0.5 | BENDING-V (in-phase vertical motion) |
| score_tb / score_ly < −0.5, score_xvar small | ROLLING (rigid lateral roll) |
| score_tb / score_ly < −0.5, score_xvar large | BENDING-L (lateral bending) |
| otherwise | LOCAL / MIXED |

TORSION is decided by `score_lr` alone: on real trimmed bodies the lateral
left/right U_z fingerprint is the clean signature of a rotation about X, whereas
`score_tb` (U_y based) is contaminated by local/lumped-mass motion, so it is
reported as a confidence/coupling axis rather than used as a gate.

---

## Tests and scripts

The repository separates **automated verification** from **manual observation**:

| Folder | Purpose | How to run |
|---|---|---|
| `tests/` | **Verification** — pytest with assertions, no figures, no input. Runs in CI. | `pytest` |
| `scripts/` | **Observation** — open matplotlib figures and/or ask for input. Run by hand. | `py -3 scripts/<sub>/<name>.py` |

Every file declares its nature in the first docstring line:

- `[VERIFICATION]` — assert-based test (in `tests/`)
- `[VISUAL]` — opens a figure to look at a result (in `scripts/`)
- `[EXPLORATION]` — experiments with / compares metrics (in `scripts/`)

Naming inside `scripts/`: `view_*` (look at one result), `compare_*` (compare
variants), `explore_*` (iterate on a metric).  `pytest` only collects `tests/`,
so the visual scripts never pollute the suite.  Observation scripts import
`_bootstrap` (path setup) and, when interactive, `_helpers` (prompt helpers).

---

## References

- Ferreira, A.J.M. & Fantuzzi, N. (2019). *MATLAB Codes for Finite Element Analysis*. Springer.
- Allemang, R.J. (2003). The Modal Assurance Criterion — Twenty Years of Use and Abuse. *Sound and Vibration*.
