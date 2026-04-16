"""
Modal analysis pipeline for the ANSA model.
Reads pre-computed modes from Meta and prepares them for MAC correlation.
"""

import numpy as np
from pathlib import Path
from ansa_model.reader import read_ansa_results

DATA_DIR = Path("data/ansa_model")


def run_modal_analysis(file_path: Path) -> None:
    """
    Load ANSA/Meta modal results, apply any post-processing
    (DOF reordering, normalization), and save to data/ansa_model/dynamic_modes.npz
    """
    raise NotImplementedError
