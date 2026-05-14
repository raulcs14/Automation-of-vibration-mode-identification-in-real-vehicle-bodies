"""
Interactive menu helpers for terminal-based scripts.
"""

import numpy as np


def ask(prompt: str, options: list[str], default: int = 0) -> int:
    """Print numbered options, return 0-based index of chosen option."""
    print(f"\n{prompt}")
    for i, opt in enumerate(options):
        marker = " (default)" if i == default else ""
        print(f"  [{i+1}] {opt}{marker}")
    while True:
        raw = input("  > ").strip()
        if raw == "":
            return default
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print(f"  Por favor introduce un número entre 1 y {len(options)}.")


def ask_multi(prompt: str, options: list[str]) -> list[int]:
    """Allow selecting multiple options by comma-separated numbers, or 0 for all."""
    print(f"\n{prompt}")
    for i, opt in enumerate(options):
        print(f"  [{i+1}] {opt}")
    print(f"  [0] Todas")
    while True:
        raw = input("  > ").strip()
        if raw == "" or raw == "0":
            return list(range(len(options)))
        parts = [p.strip() for p in raw.split(",")]
        try:
            indices = [int(p) - 1 for p in parts]
            if all(0 <= idx < len(options) for idx in indices):
                return sorted(set(indices))
        except ValueError:
            pass
        print(f"  Introduce números separados por coma (1-{len(options)}) o 0 para todas.")


def ask_int(prompt: str, default: int) -> int:
    print(f"\n{prompt} (default: {default})")
    raw = input("  > ").strip()
    if raw == "":
        return default
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return default


def ask_yes(prompt: str, default: bool = False) -> bool:
    yn = "y/n" if default else "y/N"
    raw = input(f"\n{prompt} [{yn}]: ").strip().lower()
    if raw == "":
        return default
    return raw in ("s", "si", "sí", "y", "yes")


def ask_yn(prompt: str) -> bool:
    """Strict y/n prompt with no default (used in English-language test scripts)."""
    while True:
        raw = input(prompt + " (y/n): ").strip().lower()
        if raw in ("y", "n"):
            return raw == "y"
        print("  Please enter y or n.")


def ask_variant() -> str:
    """Ask which ANSA model variant to use; return 'BIW' or 'TB'."""
    print("\nModel variant:")
    print("  1. BIW — Body in White (no lumped masses)")
    print("  2. TB  — Trimmed Body  (with lumped masses)")
    while True:
        raw = input("Select (1/2): ").strip()
        if raw == "1":
            return "BIW"
        if raw == "2":
            return "TB"
        print("  Please enter 1 or 2.")


def ask_case(n_cases: int, names: list) -> int | None:
    """Print case menu; return 0-based index or None for all."""
    print("\nAvailable reference cases:")
    for i, name in enumerate(names):
        print(f"  {i+1:2d}. {name}")
    print(f"   0. Show all ({n_cases} cases)")
    while True:
        raw = input("Select case (0 = all): ").strip()
        if raw == "" or raw == "0":
            return None
        if raw.isdigit() and 1 <= int(raw) <= n_cases:
            return int(raw) - 1
        print(f"  Please enter a number between 0 and {n_cases}.")


def ask_mode(n_modes: int, freq: np.ndarray) -> int | None:
    """Print mode menu; return 0-based index or None for all."""
    print("\nElastic mode frequencies:")
    for i in range(n_modes):
        print(f"  {i+1:3d}.  {freq[i]:.2f} Hz")
    print(f"    0.  Show all ({n_modes} modes)")
    while True:
        raw = input("Select mode to inspect (0 = all): ").strip()
        if raw == "" or raw == "0":
            return None
        if raw.isdigit() and 1 <= int(raw) <= n_modes:
            return int(raw) - 1
        print(f"  Please enter a number between 0 and {n_modes}.")


_WEIGHTING_OPTIONS = {
    1: "Identity (plain MAC)",
    2: "Mass-weighted",
    3: "Stiffness-weighted",
    4: "Total-energy-weighted  (M·(2π·40)² + K)",
}


def ask_weighting() -> tuple[int, str]:
    """Print weighting menu; return (1-based index, label)."""
    print("\nWeighting:")
    for k, name in _WEIGHTING_OPTIONS.items():
        print(f"  {k}. {name}")
    while True:
        raw = input("Select (1-4): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= 4:
            idx = int(raw)
            return idx, _WEIGHTING_OPTIONS[idx]
        print("  Please enter a number between 1 and 4.")
