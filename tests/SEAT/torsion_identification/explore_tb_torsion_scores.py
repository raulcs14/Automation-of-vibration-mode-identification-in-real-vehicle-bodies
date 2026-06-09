"""
Sweep of torsion_score_v2 for all modes of the TB model.

Run:  py tests/SEAT/torsion_identification/explore_tb_torsion_scores.py

combined = linearity * centering * antisym
  linearity : theta_x(X) profile is linear with real amplitude (not flat)
  centering : rotation centre near the geometric centre of the vehicle
  antisym   : left/right sides move in opposite directions (torsion) vs same (bending)

Classification into 5 types from score_Uz and score_Uy:
  TORSION      : sc_Uz > 0.5  (opposite sides in Z — body torsion)
  ROLLING      : sc_Uy > 0.5  (opposite sides in Y — rolling)
  BENDING-V    : sc_Uz < -0.5 (same sides in Z     — vertical bending)
  BENDING-L    : sc_Uy < -0.5 (same sides in Y     — lateral bending)
  LOCAL/MIXED  : otherwise    (local mode, no clear lateral correlation)

Thresholds are softer (0.5 instead of 0.7) to capture mixed modes.
The classification figure uses colour = type and size = combined, so that
small modes with a good score_Uz but low combined are still distinguishable.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from seat_model.reader import read_hdf5_modal
from common.torsion_analysis import scan_torsion_scores_v2, theta_x_profile

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
# Classification
# ---------------------------------------------------------------------------
COLORS = {
    "TORSION":    "firebrick",
    "ROLLING":    "darkorange",
    "BENDING-V":  "steelblue",
    "BENDING-L":  "seagreen",
    "LOCAL/MIXED":"gray",
}
THR = 0.5  # classification threshold

def classify(suz: float, suy: float) -> str:
    # Priority: the larger of score_Uz / score_Uy decides the type.
    # If both are small -> LOCAL/MIXED.
    if suz >= suy and suz > THR:
        return "TORSION"
    if suy > suz and suy > THR:
        return "ROLLING"
    if suz < -THR:
        return "BENDING-V"
    if suy < -THR:
        return "BENDING-L"
    return "LOCAL/MIXED"

# ---------------------------------------------------------------------------
# Console table
# ---------------------------------------------------------------------------
print()
print(f"  {'rank':>4}  {'mode':>4}  {'freq':>9}  {'combined':>8}  "
      f"{'linear':>6}  {'center':>6}  {'antisym':>7}  {'unifrm':>6}  {'x0(mm)':>7}  "
      f"{'sc_Uz':>6}  {'sc_Uy':>6}  type")
print("-" * 110)
for rank, row in enumerate(results, 1):
    m    = int(row["mode_idx"])
    f    = float(row["freq_hz"])
    c    = float(row["combined"])
    lin  = float(row["linearity"])
    cen  = float(row["centering"])
    ant  = float(row["antisym"])
    unif = float(row["uniformity"])
    x0   = float(row["x0"])
    suz  = float(row["score_Uz"])
    suy  = float(row["score_Uy"])
    x0s  = f"{x0:7.0f}" if not np.isnan(x0) else "    nan"
    mtype = classify(suz, suy)
    print(f"  {rank:4d}  {m:4d}  {f:9.3f}  {c:8.4f}  {lin:6.3f}  "
          f"{cen:6.3f}  {ant:7.3f}  {unif:6.3f}  {x0s}  {suz:+6.3f}  {suy:+6.3f}  {mtype}")

# ---------------------------------------------------------------------------
# Figure 1 — score_Uz vs score_Uy map
#
# X axis = score_Uz: +1 pure torsion, -1 vertical bending.
# Y axis = score_Uy: +1 pure rolling, -1 lateral bending.
# Point size = combined: only modes with linear, centred, antisymmetric
#   profiles appear large. A mode can be well classified (high score_Uz)
#   but small if linearity or centering are low.
# ---------------------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(9, 8))

# Soft quadrant background
ax1.axhspan( THR,  1.1, xmin=(THR + 1.1) / 2.2, alpha=0.04, color="firebrick")
ax1.fill_between([-1.1, 1.1], [THR, THR], [1.1, 1.1], alpha=0.04, color="darkorange")

for row in results:
    suz  = float(row["score_Uz"])
    suy  = float(row["score_Uy"])
    comb = float(row["combined"])
    mtype = classify(suz, suy)
    size = 12 + 300 * comb   # minimum visible point even with combined~0
    ax1.scatter(suz, suy, color=COLORS[mtype], s=size, zorder=3,
                alpha=0.75, edgecolors="white", linewidths=0.3)
    ax1.annotate(
        str(int(row["mode_idx"])),
        xy=(suz, suy), xytext=(3, 2), textcoords="offset points",
        fontsize=6, color=COLORS[mtype],
    )

# Reference lines
for v in [-THR, 0, THR]:
    ax1.axhline(v, color="gray", lw=0.6, ls="--" if v == 0 else ":")
    ax1.axvline(v, color="gray", lw=0.6, ls="--" if v == 0 else ":")

# Quadrant labels
kw = dict(fontsize=7, alpha=0.5, ha="center")
ax1.text( 0.85,  0.85, "TORSION\n+ ROLLING",  color="purple",     **kw)
ax1.text( 0.85,  0.0,  "TORSION",              color="firebrick",  **kw)
ax1.text( 0.0,   0.85, "ROLLING",              color="darkorange", **kw)
ax1.text(-0.85,  0.0,  "BENDING-V",            color="steelblue",  **kw)
ax1.text( 0.0,  -0.85, "BENDING-L",            color="seagreen",   **kw)
ax1.text(-0.85, -0.85, "LOCAL",                color="gray",       **kw)

# Legend
patches = [mpatches.Patch(color=c, label=t) for t, c in COLORS.items()]
# Reference sizes for combined
for s, lbl in [(12, "combined~0"), (62, "combined=0.15"), (162, "combined=0.5")]:
    ax1.scatter([], [], color="k", s=s, alpha=0.6, label=lbl)
ax1.legend(handles=patches + ax1.get_legend_handles_labels()[0][-3:],
           loc="upper left", fontsize=7, ncol=2)

ax1.set_xlabel("score_Uz  (+1 = body torsion,  -1 = vertical bending)", fontsize=9)
ax1.set_ylabel("score_Uy  (+1 = rolling,  -1 = lateral bending)", fontsize=9)
ax1.set_title("TB mode classification  —  size proportional to combined score\n"
              f"(combined = linearity x centering x antisym,  threshold={THR})")
ax1.set_xlim(-1.1, 1.1); ax1.set_ylim(-1.1, 1.1)
ax1.set_aspect("equal")
ax1.grid(False)
fig1.tight_layout()

# ---------------------------------------------------------------------------
# Figure 2 — theta_x profiles for the N_TOP torsional modes with highest combined
# ---------------------------------------------------------------------------
torsion_types = {"TORSION", "ROLLING"}
top = [r for r in results
       if classify(float(r["score_Uz"]), float(r["score_Uy"])) in torsion_types][:N_TOP]
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
    suz  = float(row["score_Uz"])
    mtype = classify(suz, float(row["score_Uy"]))

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
