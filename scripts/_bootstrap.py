"""
Path bootstrap for the observation scripts in ``scripts/``.

These files are NOT pytest tests — they open matplotlib windows and/or ask for
interactive input.  They are meant to be run directly (IDE "Run" button or
``py -3 scripts/<sub>/<name>.py``) from any working directory.

Importing this module makes the project importable regardless of where the
script lives or how deep it is nested:

    import _bootstrap  # noqa: F401  (adds repo root + scripts/ to sys.path)

It walks up from this file until it finds the repo root (the directory that
contains the ``common`` package and ``main.py``), then puts both the repo root
(for ``common``/``simple_model``/``seat_model`` imports) and the ``scripts``
directory (for ``from _helpers import ...``) on sys.path.  No hard-coded
``parents[N]`` depth, so moving or renesting a script never breaks it.
"""

import sys
from pathlib import Path

_here = Path(__file__).resolve()
_SCRIPTS_DIR = _here.parent

for _root in _here.parents:
    if (_root / "common").is_dir() and (_root / "main.py").is_file():
        REPO_ROOT = _root
        break
else:  # pragma: no cover - only if run outside the repo
    REPO_ROOT = _SCRIPTS_DIR.parent

for _p in (str(REPO_ROOT), str(_SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
