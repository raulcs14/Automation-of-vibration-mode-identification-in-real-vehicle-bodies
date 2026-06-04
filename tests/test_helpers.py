"""
Shared helpers for interactive visual tests.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.interaction import (
    ask_yn, ask_variant, ask_case, ask_mode, ask_weighting,
)

F0_ENERGY = 40.0

__all__ = ["ask_yn", "ask_variant", "ask_case", "ask_mode", "ask_weighting", "F0_ENERGY"]
