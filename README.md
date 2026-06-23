# Automation of Vibration Mode Identification in Real Vehicle Bodies

Master's thesis (TFM) project: an automated pipeline that identifies and
classifies the vibration modes of a vehicle body from its finite-element modal
results — telling apart torsion, bending and rigid-body-like motions without a
human inspecting every mode shape by hand.

It works on two models:

- **Simple model** — a 3D beam-frame chassis built analytically in Python/NumPy.
  Needs no external data and runs out of the box. Useful for developing and
  validating the methods on a small, well-understood structure.
- **SEAT model** — a full vehicle FE model (Body in White and Trimmed Body)
  solved in ANSA/Epilysis. Reads the HDF5 result files that solver produces.

---

## What it does

The pipeline offers two independent analyses (chosen from a menu in `main.py`):

### 1. MAC — correlation with reference shapes
Correlates each dynamic mode against a set of static reference deformations
(heave, pitch, torsion, roll, …) using the **Modal Assurance Criterion**:

$$\text{MAC}(i,j) = \frac{|\varphi_i^{T} W \psi_j|^2}{(\varphi_i^{T} W \varphi_i)(\psi_j^{T} W \psi_j)}$$

A high MAC means the mode "looks like" that reference shape, so each mode gets
named after its best match. Options:

- **Weighting** `W`: Identity, Mass (M), Stiffness (K), or Energy (Mω² + K).
- **Rigid-body removal** from the reference shapes before correlating.
- **Subdomain averaging**: group nodes into zones and correlate the averaged
  motion, reducing local-mesh noise. (For the SEAT model the zones come from the
  element properties in the H5 file.)
- **CONM2 exclusion** (SEAT): drop lumped-mass nodes from the DOF space.

### 2. Torsion ID — geometric torsion identification
Finds the torsion modes **without** any reference shape, purely from how the
structure rotates about its longitudinal axis. Each mode gets a `combined` score
and a class label (TORSION / BENDING-V / BENDING-L / ROLLING / LOCAL).

The score asks one physical question — *"does this mode rotate like a rigid body
about the car's long axis?"* — and gates it by two quality checks:

```
combined = antisym × gate(linearity) × gate(centering) × local_veto
```

| Term | Meaning |
|---|---|
| **antisym** | How well the mode matches an ideal rigid rotation. Geometric mean of two least-squares fits: the lateral law `Uz = θ·Y` and the full field that also requires `Uy = −θ·Z`. ~1 only for a real rotation. |
| **linearity** | The per-slice rotation angle θ(X) grows linearly along the car (first torsion mode). |
| **centering** | The rotation axis sits near the geometric centre, not at one end. |
| **local_veto** | 0 if the mode's energy is concentrated in a tiny region (a local, non-global mode), else 1. |

`linearity` and `centering` enter as smooth sigmoid gates, so a genuine torsion
mode (good on every term) is barely touched while a poor one is suppressed —
`antisym` is what makes the true torsion mode stand out in the ranking.

The class label comes from antisymmetry fingerprints of the mode shape
(`score_lr`, `score_tb`, …); a mode is called **TORSION** when its left/right
vertical displacements are in opposite phase along the car.

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ (`numpy`, `scipy`, `matplotlib`, `h5py`).

---

## Usage

### Simple model — runs immediately
```bash
python main.py
```
Pick **Simple model** and a flow (MAC or Torsion ID). No setup, no data files.

### SEAT model (BIW / TB)
The SEAT model reads HDF5 results from ANSA/Epilysis. The empty folder tree is
already in the repo (kept by `.gitkeep`); you only add the data files. Two cases:

- **You already have the `.h5` result files** → drop each one into its folder
  (exact name and path below), then run `python main.py`. Torsion ID needs only
  the *modal* file; MAC needs all three.

  | Needed by | File (TB shown; BIW is the same with `BIW`) | Put it in |
  |---|---|---|
  | Torsion ID + MAC | `000_Header_TB_modal_run.h5` | `data/seat_model/TB/ansa/modal/output/` |
  | MAC only | `000_Header_TB_static_reference_run.h5` | `data/seat_model/TB/ansa/static/output/` |
  | MAC only | `000_Header_TB_getKM.h5` | `data/seat_model/TB/ansa/matrices/output/` |

  The file name must match exactly — the loader looks for these specific names.
  (Folders ending in `input/` are for the ANSA `.dat` decks; results go in
  `output/`.)

- **You need to solve the model first** → put your `.dat` decks in the matching
  `input/` folders, then configure your local Epilysis path and run the analyses:
  ```bash
  cp seat_model/epilysis_runner/config/paths.py.example \
     seat_model/epilysis_runner/config/paths.py   # then edit EPILYSIS_EXE
  python seat_model/epilysis_runner/run_analyses.py
  ```
  This writes the HDF5 result files into the `output/` folders. `paths.py` is
  git-ignored, so each user keeps their own.

> `data/` is git-ignored — the FE result files are not committed.

---

## Project layout

| Folder | Contents |
|---|---|
| `common/` | Model-agnostic core: MAC, rigid-body removal, subdomain reduction, torsion scoring, plotting. |
| `simple_model/` | The analytical beam-frame chassis (geometry, FE assembly, solvers). |
| `seat_model/` | SEAT model: HDF5 readers, modal/static loaders, subdomains, Epilysis runner. |
| `scripts/` | Observation scripts — open figures or print tables. Run by hand, not collected by pytest. |
| `tests/` | Automated verification (pytest, assertions only). |
| `main.py` | Interactive entry point for both flows and both models. |

Inside `scripts/`, the first docstring line tags each file: `[VISUAL]` (look at
one result), `[EXPLORATION]` (compare/iterate on a metric). Files in `tests/`
are tagged `[VERIFICATION]`.

---

## Tests

```bash
pytest
```
Runs the verification suite (geometry, MAC properties, DOF reduction, torsion
metrics, H5 pipeline). The visual `scripts/` are never collected.

---

## AI assistance

This project was developed with the help of **Claude Code** (Anthropic's
AI coding assistant), used for code review, refactoring, test writing, and
documentation. All AI-assisted changes were reviewed and validated by the author
before being committed. The engineering decisions, the physical formulation of
the torsion criteria, and the validation against the FE models are the author's
own.

---

## References

- Ferreira, A.J.M. & Fantuzzi, N. (2019). *MATLAB Codes for Finite Element Analysis*. Springer.
- Allemang, R.J. (2003). The Modal Assurance Criterion — Twenty Years of Use and Abuse. *Sound and Vibration*.
