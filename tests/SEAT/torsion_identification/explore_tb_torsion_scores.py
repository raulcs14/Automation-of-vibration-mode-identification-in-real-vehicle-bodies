"""
Sweep of torsion_score_v2 for all modes of the TB model.

Run:  py tests/SEAT/torsion_identification/explore_tb_torsion_scores.py

combined = linearity * centering * antisym * uniformity
  linearity : theta_x(X) profile is linear with real amplitude (not flat)
  centering : rotation centre near the geometric centre of the vehicle
  antisym   : mean of the two torsion fingerprints (score_lr and score_tb)

Classification from the antisymmetry fingerprints (see common.torsion_analysis
and common.visualization.torsion_plots.classify_scores):
  TORSION   : score_lr > 0.5 AND score_tb > 0.5  (left/right opposite in Z
              and top/bottom opposite in Y — a full rigid rotation about X)
  BENDING-V : score_lr < -0.5                    (sides in phase in Z)
  ROLLING   : score_tb < -0.5, U_y uniform along X (rigid lateral roll)
  BENDING-L : score_tb/score_ly < -0.5, U_y curves along X (score_xvar large)
  LOCAL/MIXED : otherwise
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from seat_model.reader import read_hdf5_modal
from common.torsion_analysis import scan_torsion_scores_v2, theta_x_profile
from common.visualization.torsion_plots import _classify_row, plot_mode_map

# ---------------------------------------------------------------------------
MODAL_H5 = Path("data/seat_model/TB/ansa/modal/output/000_Header_TB_modal_run.h5")
N_TOP    = 6
N_SLICES = 100

# ---------------------------------------------------------------------------
print("Loading TB modes...")
data     = read_hdf5_modal(MODAL_H5)
node_xyz = data["node_xyz"]
modes    = data["modes"]
freq     = data["freq"]
nNodes   = len(node_xyz)

print("Computing torsion scores v2 for all modes...")
results = scan_torsion_scores_v2(node_xyz, modes, freq, n_slices=N_SLICES, skip_rigid=True)

# ---------------------------------------------------------------------------
# Classification — uses the shared classifier so this script always matches
# main.py and the plotting module (single source of truth).
# ---------------------------------------------------------------------------
COLORS = {
    "TORSION":    "firebrick",
    "ROLLING":    "darkorange",
    "BENDING-V":  "steelblue",
    "BENDING-L":  "seagreen",
    "LOCAL/MIXTO":"gray",
}
THR = 0.5  # classification threshold

# ---------------------------------------------------------------------------
# Console table
# ---------------------------------------------------------------------------
print()
print(f"  {'rank':>4}  {'mode':>4}  {'freq':>9}  {'combined':>8}  "
      f"{'linear':>6}  {'center':>6}  {'antisym':>7}  {'unifrm':>6}  {'x0(mm)':>7}  "
      f"{'sc_lr':>6}  {'sc_tb':>6}  {'xvar':>5}  type")
print("-" * 116)
for rank, row in enumerate(results, 1):
    m    = int(row["mode_idx"])
    f    = float(row["freq_hz"])
    c    = float(row["combined"])
    lin  = float(row["linearity"])
    cen  = float(row["centering"])
    ant  = float(row["antisym"])
    unif = float(row["uniformity"])
    x0   = float(row["x0"])
    slr  = float(row["score_lr"])
    stb  = float(row["score_tb"])
    xvar = float(row["score_xvar"])
    x0s  = f"{x0:7.0f}" if not np.isnan(x0) else "    nan"
    mtype = _classify_row(row, THR)
    print(f"  {rank:4d}  {m:4d}  {f:9.3f}  {c:8.4f}  {lin:6.3f}  "
          f"{cen:6.3f}  {ant:7.3f}  {unif:6.3f}  {x0s}  {slr:+6.3f}  {stb:+6.3f}  "
          f"{xvar:5.2f}  {mtype}")

# ---------------------------------------------------------------------------
# Figure 1 — score_lr vs score_tb classification map (shared helper)
# ---------------------------------------------------------------------------
plot_mode_map(results, model_label="TB", thr=THR)

# ---------------------------------------------------------------------------
# Figure 2 — theta_x profiles for the N_TOP torsion modes with highest combined
# ---------------------------------------------------------------------------
top = [r for r in results if _classify_row(r, THR) == "TORSION"][:N_TOP]
Ux_idx = np.arange(0, 6 * nNodes, 6)
Uy_idx = np.arange(1, 6 * nNodes, 6)
Uz_idx = np.arange(2, 6 * nNodes, 6)

X_min = float(node_xyz[:, 0].min())
X_max = float(node_xyz[:, 0].max())

fig2, axes2 = plt.subplots(2, 3, figsize=(14, 7))
axes2 = axes2.flatten()

for i, row in enumerate(top):
    mi   = int(row["mode_idx"]) - 1
    fhz  = float(row["freq_hz"])
    comb = float(row["combined"])
    lin  = float(row["linearity"])
    cen  = float(row["centering"])
    ant  = float(row["antisym"])
    unif = float(row["uniformity"])
    x0   = float(row["x0"])
    mtype = _classify_row(row, THR)

    uy = modes[Uy_idx, mi]
    uz = modes[Uz_idx, mi]
    x_c, th = theta_x_profile(node_xyz, uy, uz, n_slices=N_SLICES)

    ax = axes2[i]
    ax.plot(x_c, th, "o-", color=COLORS.get(mtype, "steelblue"),
            lw=1.5, ms=4, label="theta_x")

    if len(x_c) >= 4:
        coeffs = np.polyfit(x_c, th, 1)
        x_fit  = np.array([x_c.min(), x_c.max()])
        ax.plot(x_fit, np.polyval(coeffs, x_fit),
                "--", color="tomato", lw=1.3,
                label=f"linear (lin={lin:.2f})")
        if X_min <= x0 <= X_max:
            ax.axvline(x0, color="tomato", lw=0.9, ls=":",
                       label=f"x0={x0:.0f} mm")

    ax.axhline(0, color="k", lw=0.7, ls=":")

    ax.set_xlabel("X (mm)", fontsize=8)
    ax.set_ylabel("theta_x (rad/unit)", fontsize=8)
    ax.set_title(
        f"Mode {int(row['mode_idx'])}  {fhz:.2f} Hz  [{mtype}]\n"
        f"comb={comb:.3f}  lin={lin:.2f}  cen={cen:.2f}\n"
        f"ant={ant:.2f}  unif={unif:.3f}",
        fontsize=7.5
    )
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)

fig2.suptitle(
    f"Top {N_TOP} TORSION/ROLLING modes by combined score  (linearity x centering x antisym,  {N_SLICES} slices)",
    fontsize=10
)
fig2.tight_layout()

plt.show()
