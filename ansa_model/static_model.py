"""
Load pre-computed static reference displacements for the ANSA Trimmed-Body model.

Primary data source (inside the repo, gitignored):
    data/ansa_model/ref_total_results.csv   full [trans+rot] interleaved (nDOF, nRefs)

To regenerate, run ansa_model/meta_scripts/export_static_reference.py from META.
"""

from pathlib import Path
import numpy as np
from ansa_model.reader import read_reference_csv

_REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR   = _REPO_ROOT / "data" / "ansa_model"

REF_NAMES = ["Torsion reference"]


def run_static_model() -> dict:
    """
    Load ANSA static reference displacement(s).

    Returns a dict with:
      ref_moves_raw : (GDof, nRefs)  reference displacement vectors
    """
    print("Loading ANSA static reference data...")

    ref_csv = DATA_DIR / "ref_total_results.csv"
    if not ref_csv.exists():
        raise FileNotFoundError(
            f"{ref_csv} not found.\n"
            "Run ansa_model/meta_scripts/export_static_reference.py from META to generate it."
        )

    ref = read_reference_csv(ref_csv)   # (GDof, nRefs)
    print(f"  Reference shape: {ref.shape}")
    return dict(ref_moves_raw=ref)
