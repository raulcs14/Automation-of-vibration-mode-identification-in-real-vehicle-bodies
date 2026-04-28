"""
Shared helpers for interactive visual tests.
"""

import numpy as np

F0_ENERGY = 40.0

_WEIGHTING_OPTIONS = {
    1: "Identity (plain MAC)",
    2: "Mass-weighted",
    3: "Stiffness-weighted",
    4: f"Total-energy-weighted  (M·(2π·{F0_ENERGY})² + K)",
}


def ask_variant() -> str:
    """Ask the user which model variant to use; return 'BIW' or 'TB'."""
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


def ask_yn(prompt: str) -> bool:
    while True:
        raw = input(prompt + " (y/n): ").strip().lower()
        if raw in ("y", "n"):
            return raw == "y"
        print("  Please enter y or n.")


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


def plot_deformed(ax, nc, en, u_raw, name,
                  draw_mesh_fn, set_axes_fn, target_frac=0.08):
    """Overlay undeformed (dashed) and deformed (solid) mesh with auto-scale."""
    UX = u_raw[0::6];  UY = u_raw[1::6];  UZ = u_raw[2::6]
    umax = np.sqrt(UX**2 + UY**2 + UZ**2).max()
    bbox_diag = np.linalg.norm(nc.max(axis=0) - nc.min(axis=0))
    scale = np.clip(target_frac * bbox_diag / max(umax, 1e-12), 0.1, 200)

    nc_def = nc + scale * np.column_stack([UX, UY, UZ])
    draw_mesh_fn(ax, nc,     en, linestyle="k--")
    draw_mesh_fn(ax, nc_def, en, linestyle="r-")
    set_axes_fn(ax, np.vstack([nc, nc_def]))
    ax.set_title(f"{name}\nscale={scale:.1f}  umax={umax:.2e}", fontsize=8)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=135)
    ax.grid(True)


def best_mac_per_mode(mac: np.ndarray) -> np.ndarray:
    """Return (nModes,) array of max MAC value over all references."""
    return mac.max(axis=1)
