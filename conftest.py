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

# Several MAC/simple-model scripts import the shared interactive helpers as a
# bare module: `from _helpers import ...`.  That file lives at tests/_helpers.py,
# so expose the tests/ directory on sys.path too (it historically relied on the
# now removed package __init__.py files).
HELPERS_DIR = ROOT / "tests"
if str(HELPERS_DIR) not in sys.path:
    sys.path.insert(0, str(HELPERS_DIR))
