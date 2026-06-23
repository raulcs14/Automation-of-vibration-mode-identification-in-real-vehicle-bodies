"""
Pytest bootstrap for the project.

Placing a conftest.py at the repository root makes pytest add this directory to
sys.path (rootdir insertion), so test modules can import the top-level packages
(`common`, `simple_model`, `seat_model`) when the suite is run with
`pytest` from anywhere — without each test having to patch sys.path manually.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
