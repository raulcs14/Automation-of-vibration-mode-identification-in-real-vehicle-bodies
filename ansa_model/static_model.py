"""
Load pre-computed static reference displacements for the ANSA vehicle models.

Variants:
    "BIW" — Body in White (no lumped masses)
    "TB"  — Trimmed Body (with lumped masses)

Primary data source (inside the repo, gitignored):
    data/ansa_model/<variant>/ref_total_results.csv   full [trans+rot] interleaved (nDOF, nRefs)

To regenerate, run ansa_model/meta_scripts/export_static_reference.py from META.
"""

from pathlib import Path
import numpy as np
from ansa_model.reader import (read_csv, read_f06_biw)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ANSA_ROOT = _REPO_ROOT / "data" / "ansa_model"
VARIANTS   = ["BIW", "TB"]

_META_ROOT = Path(r"C:\Users\raulc\Documents\ProyectosGit\TFM\META\Test_Epilysis")

_GETKM_F06_BIW = _META_ROOT / "BodyInWhite" / "dummycar_BIW_matrices" / "output" / "000_Header_BIW_getKM.f06"
_BDF_DIR_BIW   = _META_ROOT / "BodyInWhite" / "dummycar_BIW_matrices"

REF_NAMES   = ["Torsion reference"]
SHORT_NAMES = ["Torsion"]
# Add entries here when more reference load cases are exported from META.


def run_static_model(variant: str = "BIW") -> dict:
    """
    Load ANSA static reference displacement(s).

    BIW: filtered to A-set DOFs (same space as run_modal_analysis).
    TB:  returned in full G-set. Apply dyn["conm2_keep_mask"] via
         apply_dof_mask() if you want to remove CONM2 DOFs.

    Returns a dict with:
      ref_moves_raw : (GDof, nRefs)  reference displacement vectors
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got '{variant}'")

    DATA_DIR = _ANSA_ROOT / variant
    print(f"Loading ANSA static reference data [{variant}]...")

    ref_csv = DATA_DIR / "ref_total_results.csv"
    if not ref_csv.exists():
        raise FileNotFoundError(
            f"{ref_csv} not found.\n"
            f"Run ansa_model/meta_scripts/export_static_reference.py with VARIANT='{variant}' from META."
        )

    ref = read_csv(ref_csv, ensure_2d=True)[:, :len(REF_NAMES)]

    if variant == "BIW":
        _NAS_EXTS = ("*.bdf", "*.nas", "*.dat", "*.inc")
        bdf_files = sorted({f for ext in _NAS_EXTS for f in _BDF_DIR_BIW.glob(ext)})
        f06data       = read_f06_biw(_GETKM_F06_BIW, bdf_files)
        aset_dof_mask = f06data["aset_dof_mask"]
        ref = ref[aset_dof_mask, :]

    print(f"  Reference shape: {ref.shape}  ({len(REF_NAMES)} reference(s): {REF_NAMES})")
    return dict(ref_moves_raw=ref, ref_names=list(REF_NAMES), short_names=list(SHORT_NAMES))
