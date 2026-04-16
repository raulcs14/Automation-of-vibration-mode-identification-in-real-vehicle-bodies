"""
MAC correlation for the simple model: correlates dynamic modes to static references.
Equivalent to MAC_Matrix.m
"""

import numpy as np
from pathlib import Path
from common.mac_core import compute_mac
from common.subdomain import average_zones, reduce_mk_by_subdomains

DATA_DIR = Path("data/simple_model")


def run_mac_analysis() -> None:
    """
    Load dynamic modes and static references, compute mass- and
    stiffness-weighted MAC matrices (full and subdomain-reduced),
    print correlation table, plot heatmaps, and save results to
    data/simple_model/mac_results.npz
    """
    raise NotImplementedError
