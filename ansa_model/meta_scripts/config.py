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
_OUTPUT_ROOT = Path(r"C:\Users\raulc\Documents\ProyectosGit\TFM\Automation-of-vibration-mode-identification-in-real-vehicle-bodies\data")

_TB  = _META_ROOT / "TrimmedBody"
_BIW = _META_ROOT / "BodyInWhite"

# ---------------------------------------------------------------------------
# Input files per variant  — edit paths here when folder names change
# ---------------------------------------------------------------------------

_INPUTS = {
    "BIW": {
        "modal_dat":    _BIW / "dummycar_BIW_modal" / "000_Header_BIW_modal.dat",
        "modal_op2":    _BIW / "dummycar_BIW_modal" / "output" / "000_Header_BIW_modal.op2",
        "static_dat":   _BIW / "dummycar_BIW_staticReference" / "000_Header_BIW.dat",
        "static_op2":   _BIW / "dummycar_BIW_staticReference" / "output" / "000_Header_BIW.op2",
        "matrices_K":   _BIW / "dummycar_BIW_matrices" / "output" / "K.npz",
        "matrices_M":   _BIW / "dummycar_BIW_matrices" / "output" / "M.npz",
    },
    "TB": {
        "modal_dat":    _TB / "dummycar_TB" / "000_Header_TB_modal.dat",
        "modal_op2":    _TB / "dummycar_TB" / "000_Header_TB_modal.op2",
        "static_dat":   _TB / "dummycar_TB_static_reference" / "000_Header_TB_static_reference.dat",
        "static_op2":   _TB / "dummycar_TB_static_reference" / "000_Header_TB_static_reference.op2",
        "matrices_h5":  _TB / "dummycar_TB_matrices" / "000_Header_TB_getKM.h5",
    },
}

_OUTPUTS = {
    "BIW": _OUTPUT_ROOT / "ansa_model" / "BIW",
    "TB":  _OUTPUT_ROOT / "ansa_model" / "TB",
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

OUTPUT_DIR = _OUTPUTS[VARIANT]
