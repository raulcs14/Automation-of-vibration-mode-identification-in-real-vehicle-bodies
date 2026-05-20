"""
Extract A-set and G-set node IDs from the Epilysis H5 file (MAT2HDF output).

  EPILYSIS/ASSEMBLY/USET/DATA  — (nDOF_A,) structured array with fields ID, C
                                  contains every DOF in the A-set (stiffness/mass matrices)
  EPILYSIS/INPUT/NODE/GRID     — (nNodes,) structured array with field ID
                                  contains all G-set nodes

Can be run directly (no META needed):
    python meta_runner/scripts/read_f06_uset.py TB
    python meta_runner/scripts/read_f06_uset.py BIW

Or called programmatically by run_postprocess.py.

Outputs saved to data/seat_model/<MODEL>/meta/:
    node_ids_aset.csv
    node_ids_gset.csv
    excluded_node_ids.csv
"""

import sys
import numpy as np
import h5py
from pathlib import Path


def run(h5_path: Path, output_dir: Path) -> None:
    print(f"Reading : {h5_path}")

    with h5py.File(h5_path, "r") as f:
        uset = f["EPILYSIS/ASSEMBLY/USET/DATA"][:]
        grids = f["EPILYSIS/INPUT/NODE/GRID"][:]

    a_nodes = set(int(x) for x in np.unique(uset["ID"]))
    g_nodes = set(int(x) for x in grids["ID"])
    excluded = g_nodes - a_nodes

    print(f"\nG-set (all nodes)   : {len(g_nodes):5d} nodes  ({len(g_nodes)*6:6d} DOFs)")
    print(f"A-set (matrix nodes): {len(a_nodes):5d} nodes  ({len(a_nodes)*6:6d} DOFs)")
    print(f"Excluded from A-set : {len(excluded):5d} nodes  ({len(excluded)*6:6d} DOFs)")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_dir / "node_ids_aset.csv",     np.array(sorted(a_nodes),  dtype=int), fmt="%d", delimiter=",")
    np.savetxt(output_dir / "node_ids_gset.csv",     np.array(sorted(g_nodes),  dtype=int), fmt="%d", delimiter=",")
    np.savetxt(output_dir / "excluded_node_ids.csv", np.array(sorted(excluded), dtype=int), fmt="%d", delimiter=",")

    print(f"\nSaved to {output_dir}/")
    print(f"  node_ids_aset.csv      ({len(a_nodes)} nodes)")
    print(f"  node_ids_gset.csv      ({len(g_nodes)} nodes)")
    print(f"  excluded_node_ids.csv  ({len(excluded)} nodes)")


def main():
    _HERE = Path(__file__).resolve()
    sys.path.insert(0, str(_HERE.parents[1]))  # meta_runner/

    from config.paths import ALL_INPUTS, ALL_OUTPUTS

    variant = sys.argv[1] if len(sys.argv) > 1 else None
    if variant is None or variant not in ALL_INPUTS:
        print("Usage: python read_f06_uset.py TB|BIW")
        sys.exit(1)

    h5_path    = ALL_INPUTS[variant]["matrices_h5"]
    output_dir = ALL_OUTPUTS[variant]
    run(h5_path, output_dir)


if __name__ == "__main__":
    main()
