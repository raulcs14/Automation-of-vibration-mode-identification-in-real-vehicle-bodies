"""
Shared helpers for interactive visual tests.

Lives at tests/_helpers.py (leading underscore so pytest does not collect it as
a test).  Tests import it as `from _helpers import ...` after putting the tests/
directory on sys.path.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root

from common.interaction import (
    ask_yn, ask_variant, ask_case, ask_mode, ask_weighting,
)

F0_ENERGY = 40.0

__all__ = ["ask_yn", "ask_variant", "ask_case", "ask_mode", "ask_weighting", "F0_ENERGY"]
