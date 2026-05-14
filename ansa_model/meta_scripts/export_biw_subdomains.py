"""
Export shell subdomains (PID -> list of GRID IDs) for BIW or TB.

Select the model by setting VARIANT in config.py, then run from META:
    File > Execute Script > export_biw_subdomains.py

Output:
    data/ansa_model/<VARIANT>/subdomains.json
    {
      "pid_2":  [101, 102, 103, ...],
      "pid_17": [201, 345, ...],
      ...
    }

Keys are "pid_<N>" strings so the JSON is self-documenting.
Only shell elements (CQUAD4, CQUAD8, CTRIA3, CTRIA6) are considered.
"""

import json
from config import INPUT_MODAL_DAT, OUTPUT_DIR

from meta import models

SHELL_DECK_TYPES = {"CQUAD4", "CQUAD8", "CTRIA3", "CTRIA6"}
OUTPUT_FILE = OUTPUT_DIR / "subdomains.json"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
print("Loading model …")
model = models.LoadModel("MetaPost", str(INPUT_MODAL_DAT), "NASTRAN")
all_elems = model.get_elements("all")

# Filter to shell elements only
shell_elems = [e for e in all_elems if e.get_deck_type() in SHELL_DECK_TYPES]
print(f"  Shell elements: {len(shell_elems)}")

# ---------------------------------------------------------------------------
# Build PID -> set of GRID IDs
pid_to_grids: dict[int, set] = {}
for e in shell_elems:
    pid = e.part_id
    grids = {n.id for n in e.get_nodes()}
    pid_to_grids.setdefault(pid, set()).update(grids)

print(f"  Distinct PIDs : {len(pid_to_grids)}")

# ---------------------------------------------------------------------------
# Serialize: sort PIDs and node lists for reproducibility
subdomains = {
    f"pid_{pid}": sorted(pid_to_grids[pid])
    for pid in sorted(pid_to_grids)
}

total_nodes = sum(len(v) for v in subdomains.values())
print(f"  Total node entries (with overlap): {total_nodes}")

# Report any PIDs with very few nodes (might be degenerate)
for key, grids in subdomains.items():
    if len(grids) < 4:
        print(f"  WARNING: {key} has only {len(grids)} nodes")

# ---------------------------------------------------------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(subdomains, f, indent=2)

print(f"\nExported {len(subdomains)} subdomains -> {OUTPUT_FILE}")
print("Done.")
