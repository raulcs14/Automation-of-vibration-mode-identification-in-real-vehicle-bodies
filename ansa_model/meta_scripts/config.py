"""
Paths configuration for META export scripts.
Edit INPUT_* to point to your Nastran result files.
OUTPUT_DIR is where CSVs will be saved — should match data/ansa_model/ in the repo.
"""

from pathlib import Path

# --- Input files (Nastran outputs, outside the repo) -------------------------
INPUT_MODAL_DAT = Path(r"C:\Users\raulc\Documents\ProyectosGit\TFM\META\Test_Epilysis\dummycar_TB\000_Header_TB_modal.dat")
INPUT_MODAL_OP2 = Path(r"C:\Users\raulc\Documents\ProyectosGit\TFM\META\Test_Epilysis\dummycar_TB\000_Header_TB_modal.op2")

INPUT_STATIC_DAT = Path(r"C:\Users\raulc\Documents\ProyectosGit\TFM\META\Test_Epilysis\dummycar_TB_static_reference\000_Header_TB_static_reference.dat")
INPUT_STATIC_OP2 = Path(r"C:\Users\raulc\Documents\ProyectosGit\TFM\META\Test_Epilysis\dummycar_TB_static_reference\000_Header_TB_static_reference.op2")

# --- Output directory (inside the repo, gitignored) --------------------------
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "ansa_model"
