"""
Shared interaction helpers for the observation scripts in ``scripts/``.

Lives at ``scripts/_helpers.py``.  Scripts import it as ``from _helpers import ...``
after importing ``_bootstrap`` (which puts ``scripts/`` on sys.path).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root

import numpy as np

from common.interaction import (
    ask_yn, ask_variant, ask_case, ask_mode, ask_weighting,
)

F0_ENERGY = 40.0


def require_case(n_cases: int, names: list) -> int:
    """Like ask_case but forces a single concrete choice (no 'show all').

    Re-prompts until a valid 0-based index is selected, so callers always get
    exactly one case to display — used by the single-figure view_* scripts.
    """
    while True:
        chosen = ask_case(n_cases, names)
        if chosen is not None:
            return chosen
        print("  Please choose a specific case (0 / show-all is not available here).")


def require_mode(n_modes: int, freq: np.ndarray) -> int:
    """Like ask_mode but forces a single concrete choice (no 'show all').

    Re-prompts until a valid 0-based index is selected, so callers always get
    exactly one mode to display — used by the single-figure view_* scripts.
    """
    while True:
        chosen = ask_mode(n_modes, freq)
        if chosen is not None:
            return chosen
        print("  Please choose a specific mode (0 / show-all is not available here).")


__all__ = [
    "ask_yn", "ask_variant", "ask_case", "ask_mode", "ask_weighting",
    "require_case", "require_mode", "F0_ENERGY",
]
