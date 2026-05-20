"""
Export shell subdomains (PID -> list of GRID IDs) for BIW or TB.

Invoked by meta_runner/run_postprocess.py via:
    meta_post64.bat -b -s export_biw_subdomains.py

Reads paths from environment variables set by the launcher:
    META_MODAL_DAT, META_OUTPUT_DIR
"""

import json
import os
from pathlib import Path
from meta import models

SHELL_DECK_TYPES = {"CQUAD4", "CQUAD8", "CTRIA3", "CTRIA6"}

INPUT_MODAL_DAT = Path(os.environ["META_MODAL_DAT"])
OUTPUT_DIR      = Path(os.environ["META_OUTPUT_DIR"])
OUTPUT_FILE     = OUTPUT_DIR / "subdomains.json"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading model ...")
model = models.LoadModel("MetaPost", str(INPUT_MODAL_DAT), "NASTRAN")
all_elems = model.get_elements("all")

shell_elems = [e for e in all_elems if e.get_deck_type() in SHELL_DECK_TYPES]
print(f"  Shell elements: {len(shell_elems)}")

pid_to_grids: dict[int, set] = {}
for e in shell_elems:
    pid = e.part_id
    grids = {n.id for n in e.get_nodes()}
    pid_to_grids.setdefault(pid, set()).update(grids)

print(f"  Distinct PIDs : {len(pid_to_grids)}")

subdomains = {
    f"pid_{pid}": sorted(pid_to_grids[pid])
    for pid in sorted(pid_to_grids)
}

total_nodes = sum(len(v) for v in subdomains.values())
print(f"  Total node entries (with overlap): {total_nodes}")

for key, grids in subdomains.items():
    if len(grids) < 4:
        print(f"  WARNING: {key} has only {len(grids)} nodes")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(subdomains, f, indent=2)

print(f"\nExported {len(subdomains)} subdomains -> {OUTPUT_FILE}")
print("Done.")
