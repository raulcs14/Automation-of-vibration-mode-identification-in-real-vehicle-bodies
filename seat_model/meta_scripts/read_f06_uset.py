"""
Parse a Nastran .f06 file from a getKM run and extract the node IDs
that are present in the stiffness/mass matrices (A-set).

The .f06 contains two USET tables:
  - "A  DISPLACEMENT SET"  — the DOFs included in K and M  (nDOF_A rows)
  - "G  DISPLACEMENT SET"  — all model DOFs                (nDOF_G rows)

Each table row has the format:
    <pos_from>= <nodeID>-<dof> ... <nodeID>-<dof>  = <pos_to>
e.g.:
      1=        1-1        1-2        1-3  ...  = 10
    361=       62-1       62-2  ...

Node IDs in these tables are internal (sequential) IDs. In the BIW/TB models
they happen to coincide with the external Nastran GRID IDs (1..N), so no
remapping is needed.

Usage (standalone):
    py -3 read_f06_uset.py          # uses VARIANT from config.py
    py -3 read_f06_uset.py BIW
    py -3 read_f06_uset.py TB

Output saved to data/ansa_model/<VARIANT>/:
    node_ids_aset.csv      — node IDs in A-set (sorted); use to filter modes/ref
    node_ids_gset.csv      — all node IDs in G-set (sorted)
    excluded_node_ids.csv  — node IDs in G-set but NOT in A-set
"""

import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — resolved from config.py, same as all other meta scripts
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]

_F06_PATHS = {
    "BIW": Path(r"C:\Users\raulc\Documents\ProyectosGit\TFM\META\Test_Epilysis"
                r"\BodyInWhite\dummycar_BIW_matrices\output\000_Header_BIW_getKM.f06"),
    "TB":  Path(r"C:\Users\raulc\Documents\ProyectosGit\TFM\META\Test_Epilysis"
                r"\TrimmedBody\dummycar_TB_matrices\output\000_Header_TB_getKM.f06"),
}

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
_USET_HEADER = re.compile(
    r"U\s+S\s+E\s+T\s+D\s+E\s+F\s+I\s+N\s+I\s+T\s+I\s+O\s+N\s+T\s+A\s+B\s+L\s+E"
)
_SET_LABEL = re.compile(r"\b([A-Z])\s+DISPLACEMENT\s+SET")
_DATA_ROW  = re.compile(r"^\s*\d+=(.+?)(?:=\s*\d+)?\s*$")
_DOF_TOKEN = re.compile(r"(\d+)-\d")


def _parse_uset_block(lines: list[str], start: int, set_name: str) -> tuple[set[int], int]:
    """
    Parse one USET data block starting just after the 'X DISPLACEMENT SET'
    header line. Returns (set_of_node_ids, index_of_last_consumed_line).
    """
    node_ids: set[int] = set()
    i = start
    while i < len(lines):
        line = lines[i]
        # A page break (Nastran inserts '1 title ...' lines) means the table
        # continues on the next page — skip header lines until data resumes.
        if re.match(r"^1\s+\S", line):
            i += 1
            continue
        # Another USET header or a new "X DISPLACEMENT SET" label signals a
        # different set table — stop consuming lines for the current block.
        if _USET_HEADER.search(line):
            break
        lm = _SET_LABEL.search(line)
        if lm and lm.group(1) != set_name:
            break
        m = _DATA_ROW.match(line)
        if m:
            for token_m in _DOF_TOKEN.finditer(m.group(1)):
                node_ids.add(int(token_m.group(1)))
        i += 1
    return node_ids, i


def parse_uset_tables(f06_path: Path) -> dict[str, set[int]]:
    """
    Read f06 and return a dict with keys 'A' and 'G', each mapping to the
    set of node IDs present in that displacement set.

    Two formats are handled:
      1. Full header:  "U S E T  D E F I N I T I O N  T A B L E ..."
                       followed by "X  DISPLACEMENT SET" a few lines later.
      2. Page-break continuation: no USET header, just "X  DISPLACEMENT SET"
         appearing directly (Nastran reprints the set label on each new page).
    """
    lines = f06_path.read_text(encoding="latin-1").splitlines()
    sets: dict[str, set[int]] = {}

    i = 0
    while i < len(lines):
        # Case 1: full USET header → look for set label in the next few lines
        if _USET_HEADER.search(lines[i]):
            for j in range(i + 1, min(i + 6, len(lines))):
                lm = _SET_LABEL.search(lines[j])
                if lm:
                    set_name = lm.group(1)
                    node_ids, i = _parse_uset_block(lines, j + 1, set_name)
                    sets[set_name] = node_ids
                    break
            else:
                i += 1
        # Case 2: bare "X  DISPLACEMENT SET" label (continuation page or
        # second table without its own USET header)
        else:
            lm = _SET_LABEL.search(lines[i])
            if lm:
                set_name = lm.group(1)
                if set_name not in sets:
                    node_ids, i = _parse_uset_block(lines, i + 1, set_name)
                    sets[set_name] = node_ids
                else:
                    i += 1
            else:
                i += 1

    return sets


def parse_singular_nodes(f06_path: Path) -> set[int]:
    """
    Extract node IDs listed in the GRID POINT SINGULARITY TABLE
    (nodes with stiffness ratio 0.00E+00 moved to SB set).
    """
    singular: set[int] = set()
    in_table = False
    pattern = re.compile(r"^\s+(\d+)\s+G\s+\d+\s+0\.00E\+00")
    for line in f06_path.read_text(encoding="latin-1").splitlines():
        if "G R I D   P O I N T   S I N G U L A R I T Y" in line:
            in_table = True
            continue
        if in_table:
            m = pattern.match(line)
            if m:
                singular.add(int(m.group(1)))
    return singular


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(variant: str | None = None):
    if variant is None:
        variant = sys.argv[1] if len(sys.argv) > 1 else None
    if variant is None:
        from config import VARIANT
        variant = VARIANT
    if variant not in _F06_PATHS:
        raise ValueError(f"variant must be one of {list(_F06_PATHS)}, got '{variant}'")

    f06_path   = _F06_PATHS[variant]
    output_dir = _REPO_ROOT / "data" / "ansa_model" / variant

    print(f"Variant : {variant}")
    print(f"Parsing : {f06_path}")

    sets = parse_uset_tables(f06_path)

    if "A" not in sets or "G" not in sets:
        found = list(sets.keys())
        raise ValueError(f"Expected sets 'A' and 'G' in f06, found: {found}")

    a_nodes  = sets["A"]
    g_nodes  = sets["G"]
    excluded = g_nodes - a_nodes

    print(f"\nG-set (all nodes)   : {len(g_nodes):5d} nodes  ({len(g_nodes)*6:6d} DOFs)")
    print(f"A-set (matrix nodes): {len(a_nodes):5d} nodes  ({len(a_nodes)*6:6d} DOFs)")
    print(f"Excluded from A-set : {len(excluded):5d} nodes  ({len(excluded)*6:6d} DOFs)")

    singular = parse_singular_nodes(f06_path)
    print(f"\nSingular nodes (zero stiffness in SINGULARITY TABLE): {len(singular)}")
    spc_excluded = excluded - singular
    print(f"Other excluded nodes (SPC or constraints)           : {len(spc_excluded)}")
    if spc_excluded:
        print(f"  IDs: {sorted(spc_excluded)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    a_arr  = np.array(sorted(a_nodes),  dtype=int)
    g_arr  = np.array(sorted(g_nodes),  dtype=int)
    ex_arr = np.array(sorted(excluded), dtype=int)

    np.savetxt(output_dir / "node_ids_aset.csv",     a_arr,  fmt="%d", delimiter=",")
    np.savetxt(output_dir / "node_ids_gset.csv",     g_arr,  fmt="%d", delimiter=",")
    np.savetxt(output_dir / "excluded_node_ids.csv", ex_arr, fmt="%d", delimiter=",")

    print(f"\nSaved to {output_dir}/")
    print(f"  node_ids_aset.csv      ({len(a_arr)} nodes)")
    print(f"  node_ids_gset.csv      ({len(g_arr)} nodes)")
    print(f"  excluded_node_ids.csv  ({len(ex_arr)} nodes)")


if __name__ == "__main__":
    main()
