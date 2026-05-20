"""
Copy the K/M matrix H5 file to data/seat_model/<MODEL>/meta/matrices/.

Both TB and BIW produce a .h5 via PARAM,MAT2HDF,ALL.
No META needed — pure Python file I/O.
"""

import shutil
from pathlib import Path


def run(variant: str, inputs: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    src = inputs["matrices_h5"]
    if not src.exists():
        raise FileNotFoundError(f"H5 file not found: {src}")
    dest = output_dir / "matrices.h5"
    shutil.copy2(src, dest)
    print(f"Copied {src.name} -> {dest}  ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
