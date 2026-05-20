"""
Interactive launcher for Epilysis analyses.

Usage:
    python epilysis_runner/run_analyses.py

Requires epilysis_runner/config/paths.py (copy from paths.py.example and fill
in your local paths — the file is gitignored).
"""

import subprocess
import sys
import re
import time
import threading
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve config
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

try:
    from config.paths import EPILYSIS_EXE
    from config.paths import (
        TB_MODAL_DIR, TB_STATIC_DIR, TB_MATRICES_DIR,
        TB_MODAL_DAT, TB_STATIC_DAT, TB_MATRICES_DAT, TB_OUTPUT_ROOT,
        BIW_MODAL_DIR, BIW_STATIC_DIR, BIW_MATRICES_DIR,
        BIW_MODAL_DAT, BIW_STATIC_DAT, BIW_MATRICES_DAT, BIW_OUTPUT_ROOT,
    )
except ImportError:
    sys.exit(
        "ERROR: epilysis_runner/config/paths.py not found.\n"
        "Copy epilysis_runner/config/paths.py.example to paths.py and fill in your paths."
    )

# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------

# Known analysis phases in order, with approximate cumulative % of total time
_PHASES = [
    ("Analysis begin",              0),
    ("Model load begin",            2),
    ("Model load end",             10),
    ("Geometry processor begin",   12),
    ("Geometry processor end",     15),
    ("Structural processor begin", 17),
    ("Structural processor end",   30),
    ("Solution begin",             32),
    ("Solution end",               65),
    ("Data recovery begin",        67),
    ("Data recovery end",          80),
    ("Output processor begin",     82),
    ("Output processor end",       98),
    ("Analysis end",              100),
]

_BAR_WIDTH = 40


def _render_bar(pct: int, phase: str) -> str:
    filled = int(_BAR_WIDTH * pct / 100)
    bar = "#" * filled + "-" * (_BAR_WIDTH - filled)
    label = phase[:35].ljust(35)
    return f"\r  [{bar}] {pct:3d}%  {label}"


class ProgressMonitor:
    """Watches the .f04 file written by Epilysis and renders a progress bar."""

    def __init__(self, f04_path: Path):
        self._f04 = f04_path
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._pct = 0
        self._phase = "Starting..."

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)
        # Print final 100% line
        sys.stdout.write(_render_bar(100, "Analysis end") + "\n")
        sys.stdout.flush()

    def _watch(self):
        seen_lines = 0
        while not self._stop.is_set():
            if self._f04.exists():
                try:
                    lines = self._f04.read_text(encoding="utf-8", errors="replace").splitlines()
                except OSError:
                    pass
                else:
                    for line in lines[seen_lines:]:
                        for phase, pct in _PHASES:
                            if phase in line:
                                self._pct = pct
                                self._phase = phase
                                break
                    seen_lines = len(lines)

            sys.stdout.write(_render_bar(self._pct, self._phase))
            sys.stdout.flush()
            time.sleep(0.5)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val if val else default


def _ask_bool(prompt: str, default: bool = True) -> bool:
    suffix = "Y/n" if default else "y/N"
    val = input(f"{prompt} [{suffix}]: ").strip().lower()
    if not val:
        return default
    return val in ("y", "yes", "s", "si")


def _choose(prompt: str, options: list[tuple[str, str]], multi: bool = False) -> list[str]:
    print(f"\n{prompt}")
    for i, (key, label) in enumerate(options, 1):
        print(f"  {i}) {label}")
    if multi:
        print("  (enter numbers separated by spaces, or 'all')")
    raw = input("  > ").strip().lower()
    if not raw:
        return []
    if multi and raw == "all":
        return [k for k, _ in options]
    chosen = []
    for token in raw.split():
        try:
            idx = int(token) - 1
            if 0 <= idx < len(options):
                chosen.append(options[idx][0])
        except ValueError:
            pass
    return chosen


def _patch_dat(src: Path, patches: dict[str, str], dest: Path) -> None:
    """Apply line-level patches from src and write the result to dest."""
    text = src.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    patched = []
    for line in lines:
        replaced = False
        for fragment, new_line in patches.items():
            if fragment in line:
                patched.append(new_line + "\n")
                replaced = True
                break
        if not replaced:
            patched.append(line)
    dest.write_text("".join(patched), encoding="utf-8")


_EPILYSIS_EXTS = {".f04", ".f06", ".msg", ".op2", ".pch", ".edb", ".out", ".h5"}


def _clean_outdir(outdir: Path) -> None:
    """Remove all Epilysis output files from a previous run."""
    if not outdir.exists():
        return
    removed = 0
    for f in outdir.iterdir():
        if not f.is_file():
            continue
        # Match clean extensions (.f06, .h5, ...) and Epilysis backup suffixes (.f06.1, .h5.1, ...)
        ext = f.suffix
        stem_ext = Path(f.stem).suffix  # e.g. ".f06" from "foo.f06.1"
        if ext in _EPILYSIS_EXTS or stem_ext in _EPILYSIS_EXTS:
            f.unlink()
            removed += 1
    if removed:
        print(f"  Cleaned {removed} file(s) from previous run.")


def _extract_errors(f06_path: Path) -> list[str]:
    """Return lines containing FATAL ERROR or ERROR from the f06."""
    if not f06_path.exists():
        return []
    errors = []
    for line in f06_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if re.search(r"\*\*\* (FATAL )?ERROR", line):
            errors.append(line.strip())
    return errors

# ---------------------------------------------------------------------------
# Analysis configurators
# ---------------------------------------------------------------------------

def _configure_modal() -> dict:
    print("\n--- Modal analysis (SOL 103) ---")

    freq_min = _ask("  Minimum frequency [Hz]", "-1.")
    freq_max = _ask("  Maximum frequency [Hz]", "150.")

    outputs = _choose(
        "  Outputs to request (modal punch):",
        [("disp",    "DISP    — nodal displacements (eigenvectors)"),
         ("stress",  "STRESS  — element stresses (incl. shear)"),
         ("force",   "FORCE   — element forces (membrane/bending)"),
         ("gpforce", "GPFORCE — grid point force balance per element"),],
        multi=True,
    )
    if not outputs:
        outputs = ["disp", "stress", "force", "gpforce"]
        print("  (no selection — using default: DISP + STRESS + FORCE + GPFORCE)")

    patches = {
        "EIGRL,": f"EIGRL,3,{freq_min},{freq_max}.",
        "DISP(REAL,PUNCH":   "DISP(REAL,PUNCH,SORT2)   = ALL" if "disp"    in outputs else "$DISP omitted",
        "STRESS(REAL,PUNCH": "STRESS(REAL,PUNCH,SORT2) = ALL" if "stress"  in outputs else "$STRESS omitted",
        "FORCE(REAL,PUNCH":  "FORCE(REAL,PUNCH,SORT2)  = ALL" if "force"   in outputs else "$FORCE omitted",
        "GPFORCE(PUNCH":     "GPFORCE(PUNCH,SORT2)     = ALL" if "gpforce" in outputs else "$GPFORCE omitted",
    }

    return {"patches": patches}


def _configure_static() -> dict:
    print("\n--- Static reference analysis (SOL 101) ---")

    change_forces = _ask_bool("  Change applied forces?", default=False)
    patches = {}
    if change_forces:
        print("  Current forces: nodes 31,32,33,34 — unit Z loads (torsion reference)")
        print("  Enter new FORCE cards (one per line, blank line to finish):")
        force_lines = []
        while True:
            line = input("    FORCE> ").strip()
            if not line:
                break
            force_lines.append(line)
        for i, node in enumerate([31, 32, 33, 34]):
            if i < len(force_lines):
                patches[f"FORCE        900      {node:2d}"] = force_lines[i]

    outputs = _choose(
        "  Outputs to request:",
        [("disp",    "DISP    — nodal displacements"),
         ("stress",  "STRESS  — element stresses (incl. shear)"),
         ("force",   "FORCE   — element forces (membrane/bending)"),
         ("gpforce", "GPFORCE — grid point force balance per element"),
         ("ese",     "ESE     — element strain energy"),],
        multi=True,
    )
    if not outputs:
        outputs = ["disp", "stress", "force", "gpforce", "ese"]
        print("  (no selection — using default: all outputs)")

    if "disp"    in outputs: patches["DISP(REAL,PUNCH"]   = "DISP(REAL,PUNCH,SORT2)   = ALL"
    if "stress"  in outputs: patches["STRESS(REAL,PUNCH"] = "STRESS(REAL,PUNCH,SORT2) = ALL"
    if "force"   in outputs: patches["FORCE(REAL,PUNCH"]  = "FORCE(REAL,PUNCH,SORT2)  = ALL"
    if "gpforce" in outputs: patches["GPFORCE(PUNCH"]     = "GPFORCE(PUNCH,SORT2)     = ALL"
    if "ese"     in outputs: patches["ESE(PUNCH"]         = "ESE(PUNCH)               = ALL"

    return {"patches": patches}


def _configure_matrices() -> dict:
    print("\n--- K/M matrices extraction (SOL 101 getKM) ---")
    print("  No configurable parameters — PARAM,MAT2HDF,ALL is fixed.")
    return {"patches": {}}


# ---------------------------------------------------------------------------
# Analysis definitions — one dict per model
# ---------------------------------------------------------------------------

_ANALYSIS_DEFS_BY_MODEL = {
    "TB": {
        "modal": {
            "label":        "Modal (SOL 103) — eigenvectors + frequencies",
            "dat":          TB_MODAL_DIR    / TB_MODAL_DAT,
            "outdir":       TB_OUTPUT_ROOT  / "modal"    / "output",
            "configurator": _configure_modal,
        },
        "static": {
            "label":        "Static reference (SOL 101) — displacements + stresses",
            "dat":          TB_STATIC_DIR   / TB_STATIC_DAT,
            "outdir":       TB_OUTPUT_ROOT  / "static"   / "output",
            "configurator": _configure_static,
        },
        "matrices": {
            "label":        "K/M matrices getKM (SOL 101)",
            "dat":          TB_MATRICES_DIR / TB_MATRICES_DAT,
            "outdir":       TB_OUTPUT_ROOT  / "matrices" / "output",
            "configurator": _configure_matrices,
        },
    },
    "BIW": {
        "modal": {
            "label":        "Modal (SOL 103) — eigenvectors + frequencies",
            "dat":          BIW_MODAL_DIR    / BIW_MODAL_DAT,
            "outdir":       BIW_OUTPUT_ROOT  / "modal"    / "output",
            "configurator": _configure_modal,
        },
        "static": {
            "label":        "Static reference (SOL 101) — displacements + stresses",
            "dat":          BIW_STATIC_DIR   / BIW_STATIC_DAT,
            "outdir":       BIW_OUTPUT_ROOT  / "static"   / "output",
            "configurator": _configure_static,
        },
        "matrices": {
            "label":        "K/M matrices getKM (SOL 101)",
            "dat":          BIW_MATRICES_DIR / BIW_MATRICES_DAT,
            "outdir":       BIW_OUTPUT_ROOT  / "matrices" / "output",
            "configurator": _configure_matrices,
        },
    },
}

# Default (kept for backward compatibility if imported externally)
ANALYSIS_DEFS = _ANALYSIS_DEFS_BY_MODEL["TB"]

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _validate() -> None:
    if not EPILYSIS_EXE.exists():
        sys.exit(f"ERROR: Epilysis not found: {EPILYSIS_EXE}")


def _run(key: str, cfg: dict, patches: dict) -> int:
    dat_src: Path = cfg["dat"]
    outdir:  Path = cfg["outdir"]
    outdir.mkdir(parents=True, exist_ok=True)
    _clean_outdir(outdir)

    if not dat_src.exists():
        print(f"  ERROR: DAT file not found: {dat_src}")
        return 1

    # The .dat must stay in its original folder so that relative INCLUDE paths
    # (for .bdf/.nas files) resolve correctly. If patches are needed, write a
    # patched copy next to the original using a fixed name; Epilysis output
    # files will then get a clean, predictable stem.
    if patches:
        dat_to_run = dat_src.parent / f"{dat_src.stem}_run.dat"
        _patch_dat(dat_src, patches, dat_to_run)
    else:
        dat_to_run = dat_src

    stem = dat_to_run.stem
    f04_path = outdir / f"{stem}.f04"
    f06_path = outdir / f"{stem}.f06"

    cmd = [
        str(EPILYSIS_EXE),
        "-i", str(dat_to_run),
        "--out-dir", str(outdir),
        "--scratch-dir", str(outdir),
        "-nopost",
    ]

    print(f"\n{'='*60}")
    print(f"  {cfg['label']}")
    print(f"  DAT    : {dat_src.name}" + (" (patched)" if patches else ""))
    print(f"  Output : {outdir}")
    print(f"{'='*60}")

    monitor = ProgressMonitor(f04_path)
    monitor.start()

    result = subprocess.run(cmd, text=True)

    monitor.stop()

    if patches and dat_to_run.exists():
        dat_to_run.unlink()

    if result.returncode == 0:
        print(f"  -> Completed. Output in: {outdir}")
    else:
        print(f"  -> FAILED (exit code {result.returncode})")
        errors = _extract_errors(f06_path)
        if errors:
            print("  Errors found in f06:")
            for e in errors:
                print(f"    {e}")
        print(f"  Full log: {f06_path}")

    return result.returncode


def main() -> None:
    print("=" * 60)
    print("  Epilysis Analysis Launcher")
    print("=" * 60)

    _validate()

    model_choice = _choose(
        "Which model do you want to analyse?",
        [("TB",  "TB  — Trimmed Body (with masses)"),
         ("BIW", "BIW — Body in White (without masses)")],
    )
    if not model_choice:
        print("No model selected. Exiting.")
        return
    model_key = model_choice[0]
    analysis_defs = _ANALYSIS_DEFS_BY_MODEL[model_key]
    print(f"\n  Model: {model_key}")

    options = [(k, v["label"]) for k, v in analysis_defs.items()]
    selected = _choose("Which analyses do you want to run?", options, multi=True)
    if not selected:
        print("No analyses selected. Exiting.")
        return

    analysis_configs = {}
    for key in selected:
        extra = analysis_defs[key]["configurator"]()
        analysis_configs[key] = extra

    print(f"\n{'='*60}")
    print(f"  Model : {model_key}")
    print("  Ready to run:")
    for key in selected:
        print(f"    - {analysis_defs[key]['label']}")
    if not _ask_bool("\nProceed?", default=True):
        print("Aborted.")
        return

    failed = []
    for key in selected:
        rc = _run(key, analysis_defs[key], analysis_configs[key]["patches"])
        if rc != 0:
            failed.append(key)

    print(f"\n{'='*60}")
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("  All selected analyses completed successfully.")


if __name__ == "__main__":
    main()
