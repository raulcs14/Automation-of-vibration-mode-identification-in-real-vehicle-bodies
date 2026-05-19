"""
Paths configuration for META export scripts.

VARIANT selects which model to export:
    "BIW" — dummycar Body in White (no lumped masses)
    "TB"  — dummycar Trimmed Body (with lumped masses)

Edit INPUT_* paths to point to your Nastran result files for each variant.
OUTPUT_DIR is set automatically to data/ansa_model/<variant>/.
"""

from pathlib import Path

VARIANT = "BIW"   # <-- change to "TB" when exporting the Trimmed Body

_META_ROOT   = Path(r"C:\Users\raulc\Documents\ProyectosGit\TFM\META\Test_Epilysis")
_REPO_ROOT   = Path(__file__).resolve().parents[3]
_SEAT_ROOT   = _REPO_ROOT / "data" / "seat_model"

_BIW_META = _META_ROOT / "BodyInWhite"

# ---------------------------------------------------------------------------
# Input files per variant  — edit paths here when folder names change
# ---------------------------------------------------------------------------

_INPUTS = {
    "BIW": {
        "modal_dat":    _BIW_META / "dummycar_BIW_modal" / "000_Header_BIW_modal.dat",
        "modal_op2":    _BIW_META / "dummycar_BIW_modal" / "output" / "000_Header_BIW_modal.op2",
        "static_dat":   _BIW_META / "dummycar_BIW_staticReference" / "000_Header_BIW.dat",
        "static_op2":   _BIW_META / "dummycar_BIW_staticReference" / "output" / "000_Header_BIW.op2",
        "matrices_K":   _BIW_META / "dummycar_BIW_matrices" / "output" / "K.npz",
        "matrices_M":   _BIW_META / "dummycar_BIW_matrices" / "output" / "M.npz",
    },
    "TB": {
        "modal_dat":    _SEAT_ROOT / "TB" / "ansa" / "modal"    / "000_Header_TB_modal.dat",
        "modal_op2":    _SEAT_ROOT / "TB" / "ansa" / "modal"    / "000_Header_TB_modal_run.op2",
        "static_dat":   _SEAT_ROOT / "TB" / "ansa" / "static"   / "000_Header_TB_static_reference.dat",
        "static_op2":   _SEAT_ROOT / "TB" / "ansa" / "static"   / "000_Header_TB_static_reference_run.op2",
        "matrices_h5":  _SEAT_ROOT / "TB" / "ansa" / "matrices" / "000_Header_TB_getKM.h5",
    },
}

_OUTPUTS = {
    "BIW": _SEAT_ROOT / "BIW" / "meta",
    "TB":  _SEAT_ROOT / "TB"  / "meta",
}

if VARIANT not in _INPUTS:
    raise ValueError(f"Unknown VARIANT '{VARIANT}'. Choose 'BIW' or 'TB'.")

INPUT_MODAL_DAT  = _INPUTS[VARIANT]["modal_dat"]
INPUT_MODAL_OP2  = _INPUTS[VARIANT]["modal_op2"]
INPUT_STATIC_DAT = _INPUTS[VARIANT]["static_dat"]
INPUT_STATIC_OP2 = _INPUTS[VARIANT]["static_op2"]

if VARIANT == "BIW":
    INPUT_MATRICES_K = _INPUTS[VARIANT]["matrices_K"]
    INPUT_MATRICES_M = _INPUTS[VARIANT]["matrices_M"]
else:
    INPUT_MATRICES_H5 = _INPUTS[VARIANT]["matrices_h5"]

OUTPUT_DIR          = _OUTPUTS[VARIANT]
OUTPUT_DIR_MODAL    = OUTPUT_DIR / "modal"
OUTPUT_DIR_STATIC   = OUTPUT_DIR / "static"
OUTPUT_DIR_MATRICES = OUTPUT_DIR / "matrices"
