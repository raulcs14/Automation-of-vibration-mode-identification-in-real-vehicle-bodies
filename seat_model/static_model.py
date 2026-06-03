"""
Load static reference displacements for the ANSA vehicle models from Epilysis H5 output.

Variants:
    "BIW" — Body in White (no lumped masses)
    "TB"  — Trimmed Body (with lumped masses)

Data source:
    data/seat_model/<variant>/ansa/static/output/000_Header_<variant>_static_reference_run.h5
"""

from pathlib import Path
from seat_model.reader import read_hdf5_static

_REPO_ROOT  = Path(__file__).resolve().parents[1]
_SEAT_ROOT  = _REPO_ROOT / "data" / "seat_model"
VARIANTS    = ["BIW", "TB"]

REF_NAMES   = ["Torsion reference"]
SHORT_NAMES = ["Torsion"]

_H5_STATIC = {
    "BIW": _SEAT_ROOT / "BIW" / "ansa" / "static" / "output" / "000_Header_BIW_static_reference_run.h5",
    "TB":  _SEAT_ROOT / "TB"  / "ansa" / "static" / "output" / "000_Header_TB_static_reference_run.h5",
}


def run_static_model(variant: str = "BIW") -> dict:
    """
    Load ANSA static reference displacement(s) from the Epilysis H5 output.

    Returns:
      ref_moves_raw : (GDof, nRefs)  reference displacement vectors
      ref_names     : list[str]
      short_names   : list[str]
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got '{variant}'")

    h5_file = _H5_STATIC[variant]
    if not h5_file.exists():
        raise FileNotFoundError(f"{h5_file} not found.")

    print(f"Loading ANSA static reference data [{variant}] from {h5_file.name}...")
    data = read_hdf5_static(h5_file)
    ref  = data["refs"][:, :len(REF_NAMES)]

    print(f"  Reference shape: {ref.shape}  ({len(REF_NAMES)} reference(s): {REF_NAMES})")
    return dict(ref_moves_raw=ref, ref_names=list(REF_NAMES), short_names=list(SHORT_NAMES))
