"""
Interactive launcher for META post-processing tasks.

Usage:
    python meta_runner/run_postprocess.py

Requires meta_runner/config/paths.py (copy from paths.py.example and fill
in your local paths — the file is gitignored).

Tasks that need META (SOL103/SOL101 result extraction) are run as:
    meta_post64.bat -b -s <script.py>
with paths passed via environment variables.

Tasks that are pure Python (f06 parsing, matrix copy) run in-process.
"""

import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

try:
    from config.paths import META_EXE, ALL_INPUTS, ALL_OUTPUTS
except ImportError:
    sys.exit(
        "ERROR: meta_runner/config/paths.py not found.\n"
        "Copy meta_runner/config/paths.py.example to paths.py and fill in your paths."
    )

_SCRIPTS_DIR = _HERE / "scripts"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _run_meta_script(script_name: str, env_extra: dict) -> int:
    """Run a Python script inside META batch mode.

    META requires session files that start with //#!python to execute Python.
    We generate a temporary .ses wrapper so the original .py stays clean.
    """
    script_path = _SCRIPTS_DIR / script_name
    if not script_path.exists():
        print(f"  ERROR: script not found: {script_path}")
        return 1

    env = os.environ.copy()
    env.update({k: str(v) for k, v in env_extra.items()})
    # Make utils.py importable from within the META Python environment
    env["PYTHONPATH"] = str(_SCRIPTS_DIR) + os.pathsep + env.get("PYTHONPATH", "")

    # Build a temporary .ses wrapper: META recognises //#!python as a Python session
    ses_path = _SCRIPTS_DIR / f"_run_{script_path.stem}.ses"
    py_source = script_path.read_text(encoding="utf-8")
    ses_path.write_text("//#!python\n" + py_source, encoding="utf-8")

    # META writes META_post.log/.ses to its working directory regardless of -nolog/-noses.
    # Use -d to redirect those files to a dedicated folder inside the repo.
    meta_workdir = _HERE / "logs"
    meta_workdir.mkdir(parents=True, exist_ok=True)

    try:
        cmd = [str(META_EXE), "-b", "-d", str(meta_workdir), "-s", str(ses_path)]
        print(f"  Running META: {script_name}")
        result = subprocess.run(cmd, env=env, cwd=str(meta_workdir))
    finally:
        ses_path.unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"  -> FAILED (exit code {result.returncode})")
    else:
        print(f"  -> Done.")
    return result.returncode


# ---------------------------------------------------------------------------
# Task runners
# ---------------------------------------------------------------------------

def _run_modal(model: str, inputs: dict, output_dir: Path) -> int:
    return _run_meta_script("export_modes.py", {
        "META_MODAL_DAT":  inputs["modal_dat"],
        "META_MODAL_OP2":  inputs["modal_op2"],
        "META_OUTPUT_DIR": output_dir / "modal",
    })


def _run_static(model: str, inputs: dict, output_dir: Path) -> int:
    return _run_meta_script("export_static_reference.py", {
        "META_STATIC_DAT": inputs["static_dat"],
        "META_STATIC_OP2": inputs["static_op2"],
        "META_OUTPUT_DIR": output_dir / "static",
    })


def _run_subdomains(model: str, inputs: dict, output_dir: Path) -> int:
    return _run_meta_script("export_biw_subdomains.py", {
        "META_MODAL_DAT":  inputs["modal_dat"],
        "META_OUTPUT_DIR": output_dir,
    })


def _run_matrices(model: str, inputs: dict, output_dir: Path) -> int:
    sys.path.insert(0, str(_SCRIPTS_DIR))
    from export_matrices import run as _export
    try:
        _export(model, inputs, output_dir / "matrices")
        return 0
    except Exception as e:
        print(f"  -> FAILED: {e}")
        return 1


def _run_h5_uset(_model: str, inputs: dict, output_dir: Path) -> int:
    sys.path.insert(0, str(_SCRIPTS_DIR))
    from read_f06_uset import run as _parse
    try:
        _parse(inputs["matrices_h5"], output_dir)
        return 0
    except Exception as e:
        print(f"  -> FAILED: {e}")
        return 1


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

_TASKS = {
    "modal":      ("Modal eigenvectors (SOL103 → CSV)",                    _run_modal),
    "static":     ("Static reference displacements (SOL101 → CSV)",        _run_static),
    "subdomains": ("Shell subdomains (PID → GRID IDs → JSON)",             _run_subdomains),
    "matrices":   ("Copy K/M matrices to meta/ output folder",             _run_matrices),
    "h5_uset":    ("Extract A-set / G-set node IDs from H5",               _run_h5_uset),
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _validate() -> None:
    if not META_EXE.exists():
        sys.exit(f"ERROR: META not found: {META_EXE}")


def main() -> None:
    print("=" * 60)
    print("  META Post-processing Launcher")
    print("=" * 60)

    _validate()

    model_choice = _choose(
        "Which model do you want to post-process?",
        [("TB",  "TB  — Trimmed Body (with masses)"),
         ("BIW", "BIW — Body in White (without masses)")],
    )
    if not model_choice:
        print("No model selected. Exiting.")
        return
    model = model_choice[0]
    inputs     = ALL_INPUTS[model]
    output_dir = ALL_OUTPUTS[model]
    print(f"\n  Model: {model}")

    task_opts = [(k, v[0]) for k, v in _TASKS.items()]
    selected = _choose("Which tasks do you want to run?", task_opts, multi=True)
    if not selected:
        print("No tasks selected. Exiting.")
        return

    print(f"\n{'='*60}")
    print(f"  Model : {model}")
    print("  Tasks :")
    for key in selected:
        print(f"    - {_TASKS[key][0]}")
    print(f"  Output: {output_dir}")
    if not _ask_bool("\nProceed?", default=True):
        print("Aborted.")
        return

    failed = []
    for key in selected:
        print(f"\n{'='*60}")
        print(f"  [{key}] {_TASKS[key][0]}")
        print(f"{'='*60}")
        rc = _TASKS[key][1](model, inputs, output_dir)
        if rc != 0:
            failed.append(key)

    print(f"\n{'='*60}")
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("  All selected tasks completed successfully.")


if __name__ == "__main__":
    main()
