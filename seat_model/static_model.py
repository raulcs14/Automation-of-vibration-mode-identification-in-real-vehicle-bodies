"""
Load pre-computed static reference displacements for the ANSA vehicle models.

Variants:
    "BIW" — Body in White (no lumped masses)
    "TB"  — Trimmed Body (with lumped masses)

Primary data source (inside the repo, gitignored):
    data/seat_model/<variant>/meta/static/ref_total_results.csv   full [trans+rot] interleaved (nDOF, nRefs)

To regenerate, run seat_model/meta_runner/scripts/export_static_reference.py from META.
"""

from pathlib import Path
import numpy as np
from seat_model.reader import read_csv

_REPO_ROOT  = Path(__file__).resolve().parents[1]
_SEAT_ROOT  = _REPO_ROOT / "data" / "seat_model"
VARIANTS    = ["BIW", "TB"]

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

    DATA_DIR = _SEAT_ROOT / variant / "meta"
    print(f"Loading ANSA static reference data [{variant}]...")

    ref_csv = DATA_DIR / "static" / "ref_total_results.csv"
    if not ref_csv.exists():
        raise FileNotFoundError(
            f"{ref_csv} not found.\n"
            f"Run seat_model/meta_runner/scripts/export_static_reference.py with VARIANT='{variant}' from META."
        )

    ref = read_csv(ref_csv, ensure_2d=True)[:, :len(REF_NAMES)]

    print(f"  Reference shape: {ref.shape}  ({len(REF_NAMES)} reference(s): {REF_NAMES})")
    return dict(ref_moves_raw=ref, ref_names=list(REF_NAMES), short_names=list(SHORT_NAMES))
