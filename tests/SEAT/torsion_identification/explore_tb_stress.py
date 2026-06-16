"""
Interactive exploration of ShearXY stresses (tau_xy) per mode — TB model.

Run:  py tests/SEAT/torsion_identification/explore_tb_stress.py

Figures:
  1. 3D scatter with real vehicle proportions (forced aspect ratio)
  2. Lateral antisymmetry: left vs right symmetric comparison
  3. Plan view (X-Y) and elevation (X-Z), both with signed tau_xy
  4. theta_x(X) profile with linear fit, linearity, centering and antisym
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from seat_model.reader import read_hdf5_element_stress, read_hdf5_modal
from common.torsion_analysis import torsion_score_v2
from common.interaction import ask_int

# ---------------------------------------------------------------------------
MODAL_H5 = Path("data/seat_model/TB/ansa/modal/output/000_Header_TB_modal_run.h5")
N_SLICES  = 30

# ---------------------------------------------------------------------------
modal   = read_hdf5_modal(MODAL_H5)
freq    = modal["freq"]
n_modes = len(freq)

print("\n=== Available modes (TB) ===")
for i, f in enumerate(freq):
    tag = " [rigid]" if f < 0.5 else ""
    print(f"  {i+1:3d}  {f:8.3f} Hz{tag}")

mode_num = ask_int(f"\nSelect mode (1-{n_modes})", default=7)
freq_sel = freq[mode_num - 1]
print(f"\nMode {mode_num}  ->  {freq_sel:.3f} Hz")

# ---------------------------------------------------------------------------
print("Reading stresses from H5...")
st = read_hdf5_element_stress(MODAL_H5, mode=mode_num)

cx      = st["centroid"][:, 0]
cy      = st["centroid"][:, 1]
cz      = st["centroid"][:, 2]
tau_avg = st["tau_xy_avg"]   # always >= 0  (mean of |top fibre| + |bottom fibre|)
tau1    = st["tau_xy1"]      # top fibre, signed
tau2    = st["tau_xy2"]      # bottom fibre, signed

print(f"  nElem = {len(cx)}")

vmax_avg = np.percentile(tau_avg, 98)
vmax_sgn = max(np.percentile(np.abs(tau1), 98), np.percentile(np.abs(tau2), 98))
norm_avg = mcolors.Normalize(vmin=0,         vmax=vmax_avg)
norm_sgn = mcolors.TwoSlopeNorm(vmin=-vmax_sgn, vcenter=0, vmax=vmax_sgn)

# ---------------------------------------------------------------------------
# Figure 1 — 3D scatter with real vehicle proportions
# ---------------------------------------------------------------------------
fig1 = plt.figure(figsize=(14, 6))
ax3d = fig1.add_subplot(111, projection="3d")
sc = ax3d.scatter(cx, cy, cz, c=tau_avg, cmap="hot_r", norm=norm_avg,
                  s=5, linewidths=0, depthshade=False)
fig1.colorbar(sc, ax=ax3d, pad=0.04, shrink=0.6,
              label="|tau_xy| mean fibres (normalised units)")

ax3d.set_xlabel("X (mm)"); ax3d.set_ylabel("Y (mm)"); ax3d.set_zlabel("Z (mm)")
ax3d.set_title(f"Mode {mode_num} — {freq_sel:.2f} Hz  |  Shear tau_xy (intensity)")

# Force real proportions: equalise range of all three axes
x_range = cx.max() - cx.min()
y_range = cy.max() - cy.min()
z_range = cz.max() - cz.min()
max_range = max(x_range, y_range, z_range)
x_mid = (cx.max() + cx.min()) / 2
y_mid = (cy.max() + cy.min()) / 2
z_mid = (cz.max() + cz.min()) / 2
ax3d.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
ax3d.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
ax3d.set_zlim(z_mid - max_range / 2, z_mid + max_range / 2)

# Isometric side view
ax3d.view_init(elev=20, azim=-60)
fig1.tight_layout()

# ---------------------------------------------------------------------------
# Figure 2 — Lateral antisymmetry: left vs right
#
# For torsion, elements on the left side (Y > 0) should have the opposite
# sign to those on the right (Y < 0). We plot signed tau_xy1 with a
# divergent colormap, split visually into Y+/Y- halves.
# ---------------------------------------------------------------------------
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: tau_xy1 (signed) vs Y
# Split by side to reveal the antisymmetry pattern
mask_left  = cy > 50
mask_right = cy < -50

ax_ant = axes2[0]
ax_ant.scatter(cy[mask_left],  tau1[mask_left],
               s=3, alpha=0.4, color="steelblue", label="Left side (Y > 0)")
ax_ant.scatter(cy[mask_right], tau1[mask_right],
               s=3, alpha=0.4, color="tomato",    label="Right side (Y < 0)")
ax_ant.axhline(0, color="k", lw=0.8, ls="--")
ax_ant.axvline(0, color="gray", lw=0.8, ls=":")
ax_ant.set_xlabel("Y (mm)")
ax_ant.set_ylabel("tau_xy1 top fibre (normalised units)")
ax_ant.set_title("Lateral antisymmetry\n(torsion: left/right have opposite sign)")
ax_ant.legend(markerscale=3, fontsize=8)
ax_ant.grid(True, alpha=0.2)

# Right panel: mean tau1 per X slice for each side
# Directly checks whether left and right move in opposite directions
bins   = np.linspace(cx.min(), cx.max(), N_SLICES + 1)
bi     = np.digitize(cx, bins)
x_mid_slices, left_means, right_means = [], [], []
for b in range(1, N_SLICES + 1):
    ml = (bi == b) & mask_left
    mr = (bi == b) & mask_right
    if ml.sum() > 2 and mr.sum() > 2:
        x_mid_slices.append(float(cx[(bi == b)].mean()))
        left_means.append(float(tau1[ml].mean()))
        right_means.append(float(tau1[mr].mean()))

x_mid_slices = np.array(x_mid_slices)
left_means   = np.array(left_means)
right_means  = np.array(right_means)

ax_sym = axes2[1]
if len(x_mid_slices) > 0:
    ax_sym.bar(x_mid_slices - 30, left_means,  width=55, color="steelblue",
               alpha=0.7, label="Left (Y > 0)")
    ax_sym.bar(x_mid_slices + 30, right_means, width=55, color="tomato",
               alpha=0.7, label="Right (Y < 0)")
    ax_sym.axhline(0, color="k", lw=0.8)
    if len(x_mid_slices) >= 3:
        corr = float(np.corrcoef(left_means, right_means)[0, 1])
        antisym_tau = -corr
        ax_sym.set_title(
            f"Mean tau_xy1 per X slice  (left vs right)\n"
            f"corr(left,right)={corr:+.2f}  ->  antisym={antisym_tau:+.2f}"
            f"  ({'TORSION' if antisym_tau > 0.5 else 'BENDING' if antisym_tau < -0.5 else 'MIXED'})"
        )
    else:
        ax_sym.set_title("Mean tau_xy1 per X slice")
    ax_sym.set_xlabel("X (mm)")
    ax_sym.set_ylabel("Mean tau_xy1 (normalised units)")
    ax_sym.legend(fontsize=8)
    ax_sym.grid(True, alpha=0.2)

fig2.suptitle(f"Mode {mode_num} — {freq_sel:.2f} Hz  |  Shear tau_xy antisymmetry")
fig2.tight_layout()

# ---------------------------------------------------------------------------
# Figure 3 — Orthogonal views with SIGNED tau_xy
#
# Why signed tau in both views:
#   - Plan view (X-Y): reveals whether symmetric elements in Y have opposite
#     sign (typical torsion pattern seen from above).
#   - Elevation (X-Z): reveals whether top/bottom zones share sign or not
#     (diagnoses vertical bending vs torsion).
#   Using unsigned avg in the plan view hides this antisymmetry pattern.
# ---------------------------------------------------------------------------
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))

ax_plan = axes3[0]
sc_p = ax_plan.scatter(cx, cy, c=tau1, cmap="RdBu_r", norm=norm_sgn, s=4)
fig3.colorbar(sc_p, ax=ax_plan, label="tau_xy1 top fibre (signed)")
ax_plan.set_xlabel("X (mm)"); ax_plan.set_ylabel("Y (mm)")
ax_plan.set_title("Plan view (X-Y)  |  signed tau_xy\n"
                  "(torsion: left/right have opposite colour)")
ax_plan.axhline(0, color="gray", lw=0.6, ls=":")
ax_plan.set_aspect("equal")

ax_alz = axes3[1]
sc_a = ax_alz.scatter(cx, cz, c=tau1, cmap="RdBu_r", norm=norm_sgn, s=4)
fig3.colorbar(sc_a, ax=ax_alz, label="tau_xy1 top fibre (signed)")
ax_alz.set_xlabel("X (mm)"); ax_alz.set_ylabel("Z (mm)")
ax_alz.set_title("Elevation (X-Z)  |  signed tau_xy\n"
                 "(vertical bending: roof/floor same colour; torsion: diagonal pattern)")
ax_alz.set_aspect("equal")

fig3.suptitle(f"Mode {mode_num} — {freq_sel:.2f} Hz  |  tau_xy distribution in orthogonal views")
fig3.tight_layout()

# ---------------------------------------------------------------------------
# Figure 4 — theta_x profile with linearity, centering and antisym
# ---------------------------------------------------------------------------
node_xyz = modal["node_xyz"]
nNodes   = len(node_xyz)
Ux_idx   = np.arange(0, 6 * nNodes, 6)
Uy_idx   = np.arange(1, 6 * nNodes, 6)
Uz_idx   = np.arange(2, 6 * nNodes, 6)
ux_mode  = modal["modes"][Ux_idx, mode_num - 1]
uy_mode  = modal["modes"][Uy_idx, mode_num - 1]
uz_mode  = modal["modes"][Uz_idx, mode_num - 1]

res  = torsion_score_v2(node_xyz, ux_mode, uy_mode, uz_mode, n_slices=N_SLICES)
x_c  = res["x_centers"]
th   = res["theta_means"]
lin  = res["linearity"]
cen  = res["centering"]
ant  = res["antisym"]
comb = res["combined"]
unif = res["uniformity"]
x0   = res["x0"]
slr  = res["score_lr"]
stb  = res["score_tb"]

fig4, ax4 = plt.subplots(figsize=(9, 4))
ax4.plot(x_c, th, "o-", color="steelblue", lw=2, ms=6, label="theta_x per slice")

if len(x_c) >= 2:
    coeffs = np.polyfit(x_c, th, 1)
    x_fit  = np.array([x_c.min(), x_c.max()])
    ax4.plot(x_fit, np.polyval(coeffs, x_fit), "--", color="tomato", lw=1.5,
             label=f"Linear fit (R²·SNR = {lin:.2f})")
    if not np.isnan(x0):
        X_min, X_max = float(node_xyz[:, 0].min()), float(node_xyz[:, 0].max())
        if X_min <= x0 <= X_max:
            ax4.axvline(x0, color="tomato", lw=1.0, ls=":", alpha=0.7,
                        label=f"Rotation centre x0={x0:.0f} mm (centering={cen:.2f})")

ax4.axhline(0, color="k", lw=0.7, ls="--")
ax4.set_xlabel("X (mm)")
ax4.set_ylabel("Mean theta_x per slice (rad/unit)")
ax4.set_title(
    f"Mode {mode_num}  —  {freq_sel:.2f} Hz\n"
    f"linearity={lin:.2f}   centering={cen:.2f}   antisym={ant:.2f}   uniformity={unif:.3f}\n"
    f"combined={comb:.3f}   (= lin x cen x ant x unif)\n"
    f"score_lr={slr:+.3f}   score_tb={stb:+.3f}"
)
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)
fig4.tight_layout()

plt.show()
