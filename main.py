"""
Unified MAC analysis pipeline — interactive mode.

Run:
    py -3 main.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ---------------------------------------------------------------------------
# Interactive menu helpers
# ---------------------------------------------------------------------------

def _ask(prompt: str, options: list[str], default: int = 0) -> int:
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


def _ask_multi(prompt: str, options: list[str]) -> list[int]:
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


def _ask_int(prompt: str, default: int) -> int:
    print(f"\n{prompt} (default: {default})")
    raw = input("  > ").strip()
    if raw == "":
        return default
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return default


def _ask_yes(prompt: str, default: bool = False) -> bool:
    yn = "S/n" if default else "s/N"
    raw = input(f"\n{prompt} [{yn}]: ").strip().lower()
    if raw == "":
        return default
    return raw in ("s", "si", "sí", "y", "yes")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _translational_indices(gdof: int) -> np.ndarray:
    return np.concatenate([np.arange(d, gdof, 6) for d in range(3)])


def _densify(mat):
    return mat.toarray() if hasattr(mat, "toarray") else np.asarray(mat)


def _select_top_from_matrices(mac_matrices: dict, n: int) -> np.ndarray:
    """Select top-n modes by best MAC value across all variants and references."""
    best_per_mode = np.stack(
        [m.max(axis=1) for m in mac_matrices.values()], axis=0
    ).max(axis=0)
    return np.sort(np.argsort(best_per_mode)[-n:])


# ---------------------------------------------------------------------------
# MAC computation — returns full (nModes, nRefs) matrices
# ---------------------------------------------------------------------------

WEIGHT_LABELS = ["Identity", "Mass", "Stiffness", "Energy"]


def _compute_mac_matrices(
    modes, refs, M_dense, K_dense, R,
    t_idx, weightings: list[str],
    use_rigid: bool, f0_energy: float,
    subdomains=None, n_nodes: int = 0,
) -> dict:
    """
    Returns dict: label -> MAC matrix (nModes, nRefs).
    Keys include weighting variants, rigid-removal variants,
    and subdomain-averaged variants (simple model only).
    """
    from common.mac_core   import compute_mac
    from common.rigid_body import remove_rigid_body_component

    Phi_t  = modes[t_idx, :]
    M_t    = M_dense[np.ix_(t_idx, t_idx)]
    K_t    = K_dense[np.ix_(t_idx, t_idx)]
    W_ener = M_t * (2 * np.pi * f0_energy) ** 2 + K_t

    all_weights = {"Identity": None, "Mass": M_t, "Stiffness": K_t, "Energy": W_ener}
    weights = {k: v for k, v in all_weights.items() if k in weightings}

    results = {}

    # Full DOFs — raw reference
    Psi_t = refs[t_idx, :]
    for w_lbl, W in weights.items():
        results[w_lbl] = compute_mac(Phi_t, Psi_t, W)

    # Full DOFs — rigid-body removed reference
    refs_proj = None
    if use_rigid and R is not None:
        refs_proj  = remove_rigid_body_component(refs, M_dense, R)
        Psi_t_proj = refs_proj[t_idx, :]
        for w_lbl, W in weights.items():
            results[f"{w_lbl} +rigid"] = compute_mac(Phi_t, Psi_t_proj, W)

    # Subdomain averaged (simple model only)
    if subdomains is not None:
        from common.subdomain import average_zones, reduce_mk_by_subdomains

        Mr, Kr, _ = reduce_mk_by_subdomains(M_t, K_t, subdomains, n_nodes)
        W_ener_r  = Mr * (2 * np.pi * f0_energy) ** 2 + Kr
        sub_w_all = {"Identity": None, "Mass": Mr, "Stiffness": Kr, "Energy": W_ener_r}
        sub_w = {k: v for k, v in sub_w_all.items() if k in weightings}

        Phi_z = average_zones(Phi_t, subdomains, n_nodes)
        Psi_z = average_zones(refs[t_idx, :], subdomains, n_nodes)
        for w_lbl, W in sub_w.items():
            results[f"{w_lbl} zones"] = compute_mac(Phi_z, Psi_z, W)

        if use_rigid and refs_proj is not None:
            Psi_z_proj = average_zones(refs_proj[t_idx, :], subdomains, n_nodes)
            for w_lbl, W in sub_w.items():
                results[f"{w_lbl} zones +rigid"] = compute_mac(Phi_z, Psi_z_proj, W)

    return results


# ---------------------------------------------------------------------------
# Table — Identification: each variant column shows "ShortName (MAC)"
# ---------------------------------------------------------------------------

def _print_identification_table(
    mac_matrices: dict,
    idx: np.ndarray,
    freq: np.ndarray,
    short_names: list[str],
    title: str,
    mode_offset: int = 0,
) -> None:
    """
    One row per mode. Each variant column shows the best-matching reference
    name and its MAC value, e.g. "Torsion (0.91)". A final CONSENSUS column
    marks whether all variants agree on the same reference.
    """
    variants  = list(mac_matrices.keys())
    col_w     = 18   # width per variant column
    mode_col  = 18   # "Mode N (freq Hz)"
    sep       = "─"

    header = f"{'Mode':<{mode_col}}" + "".join(f"{v:^{col_w}}" for v in variants) + f"{'CONSENSUS':^12}"
    print(f"\n{'='*len(header)}")
    print(f"  {title}")
    print(f"{'='*len(header)}")
    print(header)
    print(sep * len(header))

    for i in idx:
        n = mode_offset + i + 1
        mode_label = f"Mode {n:3d} ({freq[i]:.1f} Hz)"

        assigned = []
        cells    = []
        for v in variants:
            mac_row  = mac_matrices[v][i]           # (nRefs,)
            best_j   = int(np.argmax(mac_row))
            best_val = mac_row[best_j]
            sname    = short_names[best_j] if best_j < len(short_names) else f"Ref{best_j+1}"
            cell     = f"{sname} ({best_val:.2f})"
            cells.append(f"{cell:^{col_w}}")
            assigned.append(best_j)

        consensus = "✓" if len(set(assigned)) == 1 else "~"
        print(f"{mode_label:<{mode_col}}" + "".join(cells) + f"{consensus:^12}")

    print(sep * len(header))


# ---------------------------------------------------------------------------
# Table 2 — Per-reference detail: full MAC row for one mode
# ---------------------------------------------------------------------------

def _print_detail_table(
    mac_matrices: dict,
    mode_i: int,
    freq: float,
    ref_names: list[str],
    mode_offset: int = 0,
) -> None:
    """Show the full MAC value against every reference for a single mode."""
    n = mode_offset + mode_i + 1
    variants = list(mac_matrices.keys())
    col_w = 10

    print(f"\n  Detalle modo {n} ({freq:.2f} Hz)")
    print(f"  {'Referencia':<35}" + "".join(f"{v:>{col_w}}" for v in variants))
    print("  " + "─" * (35 + col_w * len(variants)))
    for j, rname in enumerate(ref_names):
        row = f"  {rname[:34]:<35}"
        for v in variants:
            row += f"{mac_matrices[v][mode_i, j]:>{col_w}.4f}"
        print(row)


# ---------------------------------------------------------------------------
# Plot — bar chart of best MAC per mode
# ---------------------------------------------------------------------------

def _plot(mac_matrices: dict, idx: np.ndarray, freq: np.ndarray,
          short_names: list[str], title: str, mode_offset: int = 0) -> None:
    labels  = list(mac_matrices.keys())
    n_modes = len(idx)
    n_vars  = len(labels)
    x       = np.arange(n_modes)
    bar_w   = 0.8 / n_vars
    colors  = cm.tab20(np.linspace(0, 1, n_vars))

    fig, ax = plt.subplots(figsize=(max(14, n_modes * 1.0), 6))
    for k, (label, color) in enumerate(zip(labels, colors)):
        mac_block = mac_matrices[label][idx]            # (n_modes, nRefs)
        best_j    = mac_block.argmax(axis=1)            # winning ref per mode
        vals      = mac_block.max(axis=1)
        offset    = (k - n_vars / 2 + 0.5) * bar_w
        bars      = ax.bar(x + offset, vals, width=bar_w, label=label,
                           color=color, alpha=0.85)
        for bar, v, j in zip(bars, vals, best_j):
            if v > 0.15:
                sname = short_names[j] if j < len(short_names) else f"Ref{j+1}"
                cx = bar.get_x() + bar.get_width() / 2
                # label inside bar, vertically centered
                ax.text(cx, v / 2,
                        sname, ha="center", va="center",
                        fontsize=6, rotation=90,
                        color="white", fontweight="bold")
                # MAC value just above bar
                ax.text(cx, v + 0.01,
                        f"{v:.2f}", ha="center", va="bottom",
                        fontsize=6)

    global_nums = mode_offset + idx + 1
    xlabels = [f"Mode {global_nums[i]}\n({freq[idx[i]]:.1f} Hz)"
               for i in range(n_modes)]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=7)
    ax.set_ylabel("Best MAC value")
    ax.set_title(title)
    ax.set_ylim(0, 1.12)
    ax.axhline(0.8, color="k", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.axhline(0.6, color="k", linewidth=0.6, linestyle=":",  alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()


# ---------------------------------------------------------------------------
# Model runners
# ---------------------------------------------------------------------------

def _run_simple(cfg: dict) -> None:
    from simple_model.analysis.modal_analysis import run_modal_analysis
    from simple_model.analysis.static_model   import run_static_model, SHORT_NAMES
    from simple_model.geometry.chassis        import build_chassis_geometry

    print("\nCargando datos del modelo simple...")
    dyn  = run_modal_analysis()
    stat = run_static_model()

    modes    = dyn["modes"]
    freq     = dyn["freq"]
    M        = _densify(dyn["M"])
    K        = _densify(dyn["K"])
    R        = dyn["R"]
    refs_all = stat["ref_moves_raw"]   # (GDof, nRefs)
    GDof     = modes.shape[0]
    n_nodes  = GDof // 6

    subdomains = None
    if not cfg["no_subdomain"]:
        geo        = build_chassis_geometry("torsion")
        subdomains = geo.subdomains

    t_idx = _translational_indices(GDof)

    refs        = refs_all[:, cfg["ref_cases"]]
    short_names = [SHORT_NAMES[rc] for rc in cfg["ref_cases"]]
    title       = "Simple model — Identificación de modos"

    print("\nCalculando matrices MAC...")
    mac_matrices = _compute_mac_matrices(
        modes, refs, M, K, R, t_idx,
        weightings   = cfg["weightings"],
        use_rigid    = cfg["use_rigid"],
        f0_energy    = cfg["f0_energy"],
        subdomains   = subdomains,
        n_nodes      = n_nodes,
    )

    idx = _select_top_from_matrices(mac_matrices, cfg["top_modes"])
    _print_identification_table(mac_matrices, idx, freq, short_names, title)

    if cfg["show_plot"]:
        _plot(mac_matrices, idx, freq, short_names, title)
        plt.show()


def _run_ansa(cfg: dict) -> None:
    from ansa_model.modal_analysis import run_modal_analysis, N_RIGID_BODY_MODES
    from ansa_model.static_model   import run_static_model, SHORT_NAMES

    print("\nCargando datos del modelo ANSA...")
    dyn  = run_modal_analysis()
    stat = run_static_model()

    modes = dyn["modes"]
    freq  = dyn["freq"]
    R     = dyn["R"]
    refs  = stat["ref_moves_raw"]   # (GDof, nRefs)
    GDof  = modes.shape[0]

    print("Densificando M y K (solo DOFs traslacionales)...")
    M_dense = _densify(dyn["M"])
    K_dense = _densify(dyn["K"])

    t_idx = _translational_indices(GDof)
    title = "ANSA Trimmed Body — Identificación de modos"

    print("Calculando matrices MAC...")
    mac_matrices = _compute_mac_matrices(
        modes, refs, M_dense, K_dense, R, t_idx,
        weightings = cfg["weightings"],
        use_rigid  = cfg["use_rigid"],
        f0_energy  = cfg["f0_energy"],
    )

    idx = _select_top_from_matrices(mac_matrices, cfg["top_modes"])
    _print_identification_table(
        mac_matrices, idx, freq, SHORT_NAMES, title,
        mode_offset=N_RIGID_BODY_MODES,
    )

    if cfg["show_plot"]:
        _plot(mac_matrices, idx, freq, SHORT_NAMES, title, mode_offset=N_RIGID_BODY_MODES)
        plt.show()


# ---------------------------------------------------------------------------
# Interactive configuration
# ---------------------------------------------------------------------------

def _interactive_config(model: str) -> dict:
    from simple_model.analysis.static_model import REF_NAMES as SIMPLE_REF_NAMES

    print("\n" + "─"*52)
    print("  Configuración del análisis MAC")
    print("─"*52)

    # --- Weightings ---
    w_idx = _ask_multi(
        "Selecciona los tipos de ponderación (0 = todas):",
        WEIGHT_LABELS,
    )
    weightings = [WEIGHT_LABELS[i] for i in w_idx]

    # --- Rigid body removal ---
    use_rigid = _ask_yes("¿Aplicar rigid-body removal a la referencia?", default=True)

    # --- Subdomain (solo simple) ---
    no_subdomain = False
    if model == "simple":
        no_subdomain = not _ask_yes("¿Usar subdomain averaging?", default=True)

    # --- Reference cases (solo simple) ---
    ref_cases = list(range(len(SIMPLE_REF_NAMES)))
    if model == "simple":
        ref_cases = _ask_multi(
            "Selecciona los casos de referencia estática (0 = todos):",
            SIMPLE_REF_NAMES,
        )

    # --- Top modes ---
    default_top = 10 if model == "simple" else 20
    top_modes = _ask_int("¿Cuántos modos mostrar?", default=default_top)

    # --- Energy frequency ---
    f0_energy = 40.0
    if "Energy" in weightings:
        f0_val = input(
            "\n  Frecuencia de referencia para energy weighting [Hz] (default: 40.0): "
        ).strip()
        if f0_val:
            try:
                f0_energy = float(f0_val)
            except ValueError:
                pass

    # --- Plot ---
    show_plot = _ask_yes("¿Mostrar gráfica de barras?", default=True)

    print("\n" + "─"*52)
    print("  Resumen de configuración")
    print("─"*52)
    print(f"  Modelo       : {model}")
    print(f"  Ponderaciones: {weightings}")
    print(f"  Rigid removal: {use_rigid}")
    if model == "simple":
        print(f"  Subdomains   : {not no_subdomain}")
        print(f"  Ref cases    : {[rc+1 for rc in ref_cases]}")
    print(f"  Top modos    : {top_modes}")
    if "Energy" in weightings:
        print(f"  f0 energy    : {f0_energy} Hz")
    print(f"  Gráfica      : {show_plot}")
    print("─"*52)

    return dict(
        weightings   = weightings,
        use_rigid    = use_rigid,
        no_subdomain = no_subdomain,
        ref_cases    = ref_cases,
        top_modes    = top_modes,
        f0_energy    = f0_energy,
        show_plot    = show_plot,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("╔════════════════════════════════════════════════════════╗")
    print("║   Vibration Mode Identification — Pipeline MAC         ║")
    print("╚════════════════════════════════════════════════════════╝")

    model_idx = _ask(
        "Selecciona el modelo:",
        ["Simple model (beam FEM chassis)", "ANSA Trimmed Body (dummycar)"],
    )
    model = ["simple", "ansa"][model_idx]

    cfg = _interactive_config(model)

    ok = _ask_yes("\n¿Iniciar el análisis con esta configuración?", default=True)
    if not ok:
        print("Análisis cancelado.")
        sys.exit(0)

    if model == "simple":
        _run_simple(cfg)
    else:
        _run_ansa(cfg)

    print("\nAnálisis completado.")


if __name__ == "__main__":
    main()
