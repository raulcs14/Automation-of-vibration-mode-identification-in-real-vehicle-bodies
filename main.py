"""
Vibration Mode Identification — interactive pipeline.

Available flows:
  MAC        : correlation with static reference shapes (all models)
  Torsion ID : automatic torsional mode identification via geometric criteria
               (ANSA BIW and TB only; simple model not supported)

Run:
    py -3 main.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from common.visualization.mac_plot import plot_mac_bar as _plot_mac_bar
from common.utils        import translational_dof_indices as _translational_indices
from common.interaction  import ask as _ask, ask_multi as _ask_multi, ask_int as _ask_int, ask_yes as _ask_yes
from common.mac_core     import best_mac_per_mode as _best_mac_per_mode, select_top_modes as _select_top_modes


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _select_top_from_matrices(mac_matrices: dict, n: int) -> np.ndarray:
    """Select top-n modes by best MAC value across all variants and references."""
    best_per_variant = {k: _best_mac_per_mode(v) for k, v in mac_matrices.items()}
    return _select_top_modes(best_per_variant, n)


# ---------------------------------------------------------------------------
# MAC computation — returns full (nModes, nRefs) matrices
# ---------------------------------------------------------------------------

WEIGHT_LABELS = ["Identity", "Mass", "Stiffness", "Energy"]


def _compute_mac_matrices(
    modes, refs, M, K, R,
    t_idx, weightings: list[str],
    use_rigid: bool, f0_energy: float,
    subdomains=None, n_nodes: int = 0,
) -> dict:
    """
    Returns dict: label -> MAC matrix (nModes, nRefs).
    M and K may be sparse or dense — M_t/K_t are sliced keeping sparsity.
    Keys include weighting variants, rigid-removal variants,
    and subdomain-averaged variants (simple model only).
    """
    import scipy.sparse as sp
    from common.mac_core   import compute_mac
    from common.rigid_body import remove_rigid_body_component

    Phi_t = modes[t_idx, :]

    # Slice translational submatrix — keeps sparse if M/K are sparse
    if sp.issparse(M):
        M_t = M[t_idx, :][:, t_idx]
        K_t = K[t_idx, :][:, t_idx]
    else:
        M_t = M[np.ix_(t_idx, t_idx)]
        K_t = K[np.ix_(t_idx, t_idx)]

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
        refs_proj  = remove_rigid_body_component(refs, M, R)
        Psi_t_proj = refs_proj[t_idx, :]
        for w_lbl, W in weights.items():
            results[f"{w_lbl} +rigid"] = compute_mac(Phi_t, Psi_t_proj, W)

    # Subdomain averaged
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

    print(f"\n  Detail — Mode {n} ({freq:.2f} Hz)")
    print(f"  {'Reference':<35}" + "".join(f"{v:>{col_w}}" for v in variants))
    print("  " + "─" * (35 + col_w * len(variants)))
    for j, rname in enumerate(ref_names):
        row = f"  {rname[:34]:<35}"
        for v in variants:
            row += f"{mac_matrices[v][mode_i, j]:>{col_w}.4f}"
        print(row)


# ---------------------------------------------------------------------------
# Model runners
# ---------------------------------------------------------------------------

def _run_simple(cfg: dict) -> None:
    from simple_model.analysis.modal_analysis import run_modal_analysis
    from simple_model.analysis.static_model   import run_static_model, SHORT_NAMES
    from simple_model.geometry.chassis        import build_chassis_geometry

    print("\nLoading simple model data...")
    dyn  = run_modal_analysis()
    stat = run_static_model()

    modes    = dyn["modes"]
    freq     = dyn["freq"]
    M        = dyn["M"]
    K        = dyn["K"]
    R        = dyn["R"]
    refs_all = stat["ref_moves_raw"]   # (GDof, nRefs)
    GDof     = modes.shape[0]
    n_nodes  = GDof // 6

    subdomains = None
    if cfg["subdomain"]:
        geo        = build_chassis_geometry("torsion")
        subdomains = geo.subdomains

    t_idx = _translational_indices(GDof)

    refs        = refs_all[:, cfg["ref_cases"]]
    short_names = [SHORT_NAMES[rc] for rc in cfg["ref_cases"]]
    title       = "Simple model — Mode identification"

    print("\nComputing MAC matrices...")
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
        _plot_mac_bar(mac_matrices, idx, freq, short_names, title)
        plt.show()


def _run_ansa(cfg: dict) -> None:
    from seat_model.modal_analysis import run_modal_analysis, N_RIGID_BODY_MODES
    from seat_model.static_model   import run_static_model, SHORT_NAMES

    variant = cfg["ansa_variant"]
    print(f"\nLoading ANSA model data [{variant}]...")
    dyn  = run_modal_analysis(variant=variant)
    stat = run_static_model(variant=variant)

    from common.dof_reduction import DofSpace

    space = DofSpace(
        modes    = dyn["modes"],
        refs     = stat["ref_moves_raw"],
        M        = dyn["M"],
        K        = dyn["K"],
        R        = dyn["R"],
        node_ids = dyn["node_ids"],
        node_xyz = dyn["node_coordinates"],
    )

    freq           = dyn["freq"]
    conm2_node_ids = dyn.get("conm2_node_ids")
    if conm2_node_ids is not None and cfg.get("remove_conm2", False):
        print("  Removing CONM2 node DOFs...")
        space.remove_nodes(conm2_node_ids)

    # Translational slice — always computed after remove_nodes
    t_idx, *_ = space.translational_slice()

    # Subdomains (BIW and TB) — always built after remove_nodes
    subdomains = None
    n_nodes    = 0
    if cfg.get("subdomain", False):
        from seat_model.subdomains import build_subdomains
        try:
            subdomains_grid = build_subdomains(variant, space.node_ids, space.node_xyz)
            subdomains      = space.build_subdomains(subdomains_grid)
            n_nodes         = space.n_nodes
            print(f"  Subdomains {variant}: {len(subdomains)} zones")
        except FileNotFoundError as e:
            print(f"  [subdomains] {e}\n  Continuing without subdomains.")

    conm2_tag = " (CONM2 removed)" if cfg.get("remove_conm2") and conm2_node_ids is not None else ""
    title = f"ANSA {variant}{conm2_tag} -- Mode identification"

    print("Computing MAC matrices...")
    mac_matrices = _compute_mac_matrices(
        space.modes, space.refs, space.M, space.K, space.R, t_idx,
        weightings = cfg["weightings"],
        use_rigid  = cfg["use_rigid"],
        f0_energy  = cfg["f0_energy"],
        subdomains = subdomains,
        n_nodes    = n_nodes,
    )

    idx = _select_top_from_matrices(mac_matrices, cfg["top_modes"])
    _print_identification_table(
        mac_matrices, idx, freq, SHORT_NAMES, title,
        mode_offset=N_RIGID_BODY_MODES,
    )

    if cfg["show_plot"]:
        _plot_mac_bar(mac_matrices, idx, freq, SHORT_NAMES, title, mode_offset=N_RIGID_BODY_MODES)
        plt.show()


# ---------------------------------------------------------------------------
# Interactive configuration
# ---------------------------------------------------------------------------

def _interactive_config_mac(model: str, variant: str = "BIW") -> dict:
    from simple_model.analysis.static_model import REF_NAMES as SIMPLE_REF_NAMES

    print("\n" + "─"*52)
    print("  MAC analysis configuration")
    print("─"*52)

    ansa_variant = variant

    # --- CONM2 removal (TB only) ---
    remove_conm2 = False
    if model == "ansa" and ansa_variant == "TB":
        remove_conm2 = _ask_yes(
            "Remove DOFs of added-mass nodes (CONM2)?", default=True
        )

    # --- Weightings ---
    w_idx = _ask_multi(
        "Select weighting types (0 = all):",
        WEIGHT_LABELS,
    )
    weightings = [WEIGHT_LABELS[i] for i in w_idx]

    # --- Rigid body removal ---
    use_rigid = _ask_yes("Apply rigid-body removal to reference shapes?", default=True)

    # --- Subdomain averaging ---
    subdomain = False
    if model == "simple" or model == "ansa":
        subdomain = _ask_yes("Use subdomain averaging?", default=True)

    # --- Reference cases (simple model only) ---
    ref_cases = list(range(len(SIMPLE_REF_NAMES)))
    if model == "simple":
        ref_cases = _ask_multi(
            "Select static reference cases (0 = all):",
            SIMPLE_REF_NAMES,
        )

    # --- Top modes ---
    default_top = 10 if model == "simple" else 20
    top_modes = _ask_int("How many top modes to display?", default=default_top)

    # --- Energy frequency ---
    f0_energy = 40.0
    if "Energy" in weightings:
        f0_val = input(
            "\n  Reference frequency for energy weighting [Hz] (default: 40.0): "
        ).strip()
        if f0_val:
            try:
                f0_energy = float(f0_val)
            except ValueError:
                pass

    # --- Plot ---
    show_plot = _ask_yes("Show bar chart?", default=True)

    print("\n" + "─"*52)
    print("  Configuration summary")
    print("─"*52)
    print(f"  Model        : {model}")
    if model == "ansa":
        labels = {"BIW": "Body in White (BIW)", "TB": "Trimmed Body (TB)"}
        print(f"  Variant      : {labels.get(ansa_variant, ansa_variant)}")
        if ansa_variant == "TB":
            print(f"  Remove CONM2 : {remove_conm2}")
    print(f"  Weightings   : {weightings}")
    print(f"  Rigid removal: {use_rigid}")
    print(f"  Subdomains   : {subdomain}")
    if model == "simple":
        print(f"  Ref cases    : {[rc+1 for rc in ref_cases]}")
    print(f"  Top modes    : {top_modes}")
    if "Energy" in weightings:
        print(f"  f0 energy    : {f0_energy} Hz")
    print(f"  Show plot    : {show_plot}")
    print("─"*52)

    return dict(
        weightings   = weightings,
        use_rigid    = use_rigid,
        subdomain    = subdomain,
        ref_cases    = ref_cases,
        top_modes    = top_modes,
        f0_energy    = f0_energy,
        show_plot    = show_plot,
        ansa_variant = ansa_variant,
        remove_conm2 = remove_conm2,
    )


# ---------------------------------------------------------------------------
# Torsion identification runner
# ---------------------------------------------------------------------------

def _run_torsion_id(model: str, variant: str) -> None:
    """
    Torsion mode identification flow based on geometric criteria:
        combined = antisym x gate(linearity) x gate(centering) x local_veto
    where antisym is the lever-arm-aware rigid-rotation fit (rigid_uz).

    Supports:
      - ANSA BIW / TB  : loaded from H5 modal file (coords in mm)
      - Simple model   : loaded from run_modal_analysis(). The beam model is
                         dimensionless (x in [-1, 5.5]); coords are scaled x1000
                         only so the fixed mm thresholds (min_radius_sq=100,
                         y_threshold=50) stay in range — not a physical m->mm conversion.
    """
    from pathlib import Path
    from common.torsion_analysis import scan_torsion_scores_v2
    from common.visualization.torsion_plots import (
        plot_cross_sections, plot_mode_map, plot_theta_profiles,
    )

    if model == "simple":
        from simple_model.analysis.modal_analysis import run_modal_analysis
        print("\nLoading simple model modes...")
        dyn      = run_modal_analysis()
        # dimensionless beam coords scaled x1000 so fixed mm thresholds stay in range
        node_xyz  = dyn["node_coordinates"] * 1000.0
        modes     = dyn["modes"]
        freq      = dyn["freq"]
        model_label = "Simple model"
    else:
        from seat_model.reader import read_hdf5_modal
        H5_PATHS = {
            "TB":  Path("data/seat_model/TB/ansa/modal/output/000_Header_TB_modal_run.h5"),
            "BIW": Path("data/seat_model/BIW/ansa/modal/output/000_Header_BIW_modal_run.h5"),
        }
        h5_path = H5_PATHS[variant]
        if not h5_path.exists():
            print(f"\n  [ERROR] Modal H5 file not found: {h5_path}")
            print("  Check that the file exists and try again.")
            return
        print(f"\nLoading modes [{variant}] from {h5_path.name}...")
        data     = read_hdf5_modal(h5_path)
        node_xyz    = data["node_xyz"]
        modes       = data["modes"]
        freq        = data["freq"]
        model_label = f"ANSA {variant}"

    # --- Configuration ---
    print("\n" + "─"*52)
    n_slices  = _ask_int("Number of longitudinal slices", default=30)
    top_modes = _ask_int("Top torsional modes to show in the plot", default=6)
    show_plot = _ask_yes("Show figures?", default=True)
    print("─"*52)

    # --- Compute scores ---
    print("\nComputing torsion scores for all modes...")
    results = scan_torsion_scores_v2(
        node_xyz, modes, freq, n_slices=n_slices, skip_rigid=True
    )

    # --- Classification ---
    THR = 0.5
    from common.visualization.torsion_plots import _classify_row

    # --- Console table ---
    print()
    # 'ant' is the rigid-rotation fit rigid_uz (drives the ranking); 'rig_uy'
    # is the stricter full (Uz,Uy) rigid fit, shown for diagnostics.  uniformity
    # is no longer displayed: it does not enter the ranking, and the local-mode
    # protection it used to provide is now handled by the 'peak' veto (which is
    # stricter for a dominant peak on a low background — see peak_concentration).
    print(f"  {'rank':>4}  {'mode':>4}  {'freq':>9}  {'combined':>8}  {'rel':>5}  "
          f"{'lin':>5}  {'cen':>5}  {'ant':>5}  {'rig_uy':>6}  {'peak':>5}  {'x0(mm)':>7}  type")
    print("  " + "-" * 104)
    torsion_only = [r for r in results if _classify_row(r, THR) == "TORSION"]
    # relative term: 1.0 for the strongest torsion mode (results sorted by combined)
    comb_max = max((float(r["combined"]) for r in torsion_only), default=1.0) or 1.0
    for rank, row in enumerate(torsion_only, 1):
        x0  = float(row["x0"])
        x0s = f"{x0:7.0f}" if not np.isnan(x0) else "    nan"
        mtype = _classify_row(row, THR)
        rel = float(row["combined"]) / comb_max
        print(f"  {rank:4d}  {int(row['mode_idx']):4d}  {float(row['freq_hz']):9.3f}"
              f"  {float(row['combined']):8.4f}  {rel*100:4.0f}%"
              f"  {float(row['linearity']):5.3f}  {float(row['centering']):5.3f}"
              f"  {float(row['antisym']):5.3f}  {float(row['rigid_uzuy']):6.3f}"
              f"  {float(row['peak']):5.3f}  {x0s}  {mtype}")

    if not show_plot:
        return

    # --- Figure 0: 3-D cross-section geometry ---
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig0 = plt.figure(figsize=(14, 8))
    ax0  = fig0.add_subplot(111, projection="3d")
    plot_cross_sections(node_xyz, n_slices=n_slices, ax=ax0)
    ax0.set_box_aspect(None)
    fig0.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

    # --- Figure 1: score_lr vs score_tb classification map ---
    plot_mode_map(results, model_label=model_label, thr=THR)

    # --- Figure 2: theta_x profiles for top torsional modes ---
    plot_theta_profiles(results, node_xyz, modes, n_slices,
                        model_label=model_label, top_modes=top_modes, thr=THR)

    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("╔════════════════════════════════════════════════════════╗")
    print("║   Vibration Mode Identification                        ║")
    print("╚════════════════════════════════════════════════════════╝")

    # --- Model selection ---
    model_idx = _ask(
        "Select model:",
        ["Simple model (beam FEM chassis)",
         "ANSA Body in White / Trimmed Body (dummycar)"],
    )
    model = ["simple", "ansa"][model_idx]

    # --- ANSA variant ---
    variant = ""
    if model == "ansa":
        v_idx   = _ask("Select variant:",
                       ["Body in White — BIW (no lumped masses)",
                        "Trimmed Body — TB (with lumped masses)"])
        variant = ["BIW", "TB"][v_idx]

    # --- Analysis flow ---
    flow_idx = _ask(
        "Select analysis type:",
        ["MAC — correlation with static reference shapes",
         "Torsion ID — geometric identification of torsional modes"],
    )
    flow = ["mac", "torsion"][flow_idx]

    if flow == "torsion":
        _run_torsion_id(model, variant)
    else:
        cfg = _interactive_config_mac(model, variant)
        ok  = _ask_yes("\nRun analysis with this configuration?", default=True)
        if not ok:
            print("Analysis cancelled.")
            sys.exit(0)
        if model == "simple":
            _run_simple(cfg)
        else:
            _run_ansa(cfg)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
