"""
MAC correlation for the ANSA model.
Reuses common MAC core and subdomain utilities.
"""

import numpy as np
from pathlib import Path
from common.mac_core import compute_mac
from common.subdomain import average_zones

DATA_DIR = Path("data/ansa_model")


def run_mac_analysis() -> None:
    """
    Load ANSA dynamic modes and reference patterns, compute MAC matrices,
    and save results to data/ansa_model/mac_results.npz
    """
    raise NotImplementedError
