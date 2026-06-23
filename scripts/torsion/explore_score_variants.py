"""
[EXPLORATION] Compare candidate torsion-score formulas on the TB model.

Goal (Raul's brief): make the first torsion mode stand out from the rest,
kill false positives, use the full [0,1] range, and never drop a real torsion
mode — WITHOUT adding net parameters.

This script does NOT modify common/torsion_analysis.py.  It re-reads the raw
sub-scores already produced by scan_torsion_scores_v2 (linearity, centering,
antisym, uniformity, peak, score_lr, score_tb) and recombines them several
ways, then reports a separability diagnostic for each variant so you can pick
the formula with evidence instead of intuition.

Run:  py scripts/torsion/explore_score_variants.py

Variants
--------
  baseline   lin * cen * antisym * unif * veto          (current)
  no_unif    lin * cen * antisym * veto                 (drop Shannon; veto stays)
  gate       antisym  if (lin>=L and cen>=C and veto)   (gate + rank by antisym)
  soft_gate  antisym * sigmoid(lin) * sigmoid(cen) * veto (soft quality gates)
  sharp_anti lin * cen * antisym^2 * veto               (sharpen the discriminant)

Separability diagnostic (higher = more "differential")
  margin_12  : (best - second) / best        gap of the #1 over the #2 torsion mode
  sep_t_nt   : min(torsion combined) - max(non-torsion combined)
               > 0  => torsion modes fully separated from everything else
  recall     : # torsion modes kept (combined > 0) / # torsion modes (by classifier)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
import _bootstrap  # noqa: F401  -- puts repo root (and scripts/) on sys.path
import numpy as np

from seat_model.reader import read_hdf5_modal
from common.torsion_analysis import scan_torsion_scores_v2
from common.visualization.torsion_plots import _classify_row

# ---------------------------------------------------------------------------
MODAL_H5 = Path("data/seat_model/TB/ansa/modal/output/000_Header_TB_modal_run.h5")
N_SLICES = 100
THR      = 0.5          # classifier threshold (same as the rest of the pipeline)
L_GATE   = 0.30         # linearity gate
C_GATE   = 0.40         # centering gate
PEAK_THR = 0.6          # local-mode veto (same default as torsion_score_v2)

# ---------------------------------------------------------------------------
print("Loading TB modes...")
data     = read_hdf5_modal(MODAL_H5)
results  = scan_torsion_scores_v2(
    data["node_xyz"], data["modes"], data["freq"],
    n_slices=N_SLICES, peak_thr=PEAK_THR, skip_rigid=True,
)

lin  = results["linearity"]
cen  = results["centering"]
ant  = results["antisym"]
unif = results["uniformity"]
peak = results["peak"]
veto = (peak <= PEAK_THR).astype(float)


def _sigmoid(x, x0, k):
    """Smooth 0->1 gate centred at x0 with slope k."""
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


VARIANTS = {
    "baseline":   lin * cen * ant * unif * veto,
    "no_unif":    lin * cen * ant * veto,
    "gate":       ant * ((lin >= L_GATE) & (cen >= C_GATE)).astype(float) * veto,
    "soft_gate":  ant * _sigmoid(lin, L_GATE, 12) * _sigmoid(cen, C_GATE, 12) * veto,
    "sharp_anti": lin * cen * ant**2 * veto,
}

# classifier label per mode (single source of truth, unchanged)
is_torsion = np.array([_classify_row(r, THR) == "TORSION" for r in results])


def diagnostics(score: np.ndarray) -> dict:
    tors_scores = score[is_torsion]
    nt_scores   = score[~is_torsion]
    order       = np.argsort(-tors_scores)
    best        = float(tors_scores[order[0]]) if len(order) else 0.0
    second      = float(tors_scores[order[1]]) if len(order) > 1 else 0.0
    margin_12   = (best - second) / best if best > 1e-12 else 0.0
    sep_t_nt    = (float(tors_scores.min()) - float(nt_scores.max())
                   if len(tors_scores) and len(nt_scores) else float("nan"))
    recall      = float(np.mean(tors_scores > 1e-12)) if len(tors_scores) else 0.0
    return dict(best=best, margin_12=margin_12, sep_t_nt=sep_t_nt, recall=recall)


# ---------------------------------------------------------------------------
# Diagnostic table
# ---------------------------------------------------------------------------
print(f"\n{len(results)} modes,  {int(is_torsion.sum())} classified TORSION "
      f"(thr={THR})\n")
print(f"  {'variant':>11}  {'best':>6}  {'margin_12':>9}  {'sep_t_nt':>9}  "
      f"{'recall':>6}   #1 mode")
print("-" * 72)
for name, score in VARIANTS.items():
    d = diagnostics(score)
    top_idx = int(results["mode_idx"][np.argmax(score)])
    top_freq = float(results["freq_hz"][np.argmax(score)])
    print(f"  {name:>11}  {d['best']:6.3f}  {d['margin_12']:9.3f}  "
          f"{d['sep_t_nt']:+9.3f}  {d['recall']:6.2f}   "
          f"mode {top_idx} ({top_freq:.2f} Hz)")

print("\nmargin_12 : gap of #1 over #2 torsion mode  (higher = #1 stands out)")
print("sep_t_nt  : min(torsion) - max(non-torsion)  (>0 = no false positives)")
print("recall    : fraction of torsion modes kept    (1.0 = none dropped)")

# ---------------------------------------------------------------------------
# Per-mode comparison for the torsion modes (top 8 by baseline)
# ---------------------------------------------------------------------------
tors_rows = np.where(is_torsion)[0]
tors_rows = tors_rows[np.argsort(-VARIANTS["baseline"][tors_rows])][:8]

print(f"\nTorsion modes — score under each variant (sorted by baseline):")
header = (f"  {'mode':>4}  {'freq':>7}  {'sc_lr':>6}  "
          + "  ".join(f"{n:>10}" for n in VARIANTS))
print(header)
print("-" * len(header))
for ri in tors_rows:
    m   = int(results["mode_idx"][ri])
    f   = float(results["freq_hz"][ri])
    slr = float(results["score_lr"][ri])
    cells = "  ".join(f"{VARIANTS[n][ri]:10.4f}" for n in VARIANTS)
    print(f"  {m:4d}  {f:7.2f}  {slr:+6.3f}  {cells}")

# ---------------------------------------------------------------------------
# False-positive autopsy: which NON-torsion modes beat the weakest torsion
# mode under no_unif?  (sep_t_nt < 0 means the bands overlap.)
# ---------------------------------------------------------------------------
score = VARIANTS["no_unif"]
tors_min = float(score[is_torsion].min())
worst_tors = int(results["mode_idx"][is_torsion][np.argmin(score[is_torsion])])
offenders = np.where((~is_torsion) & (score >= tors_min))[0]
offenders = offenders[np.argsort(-score[offenders])]

print(f"\nFalse-positive autopsy (no_unif): weakest torsion mode = "
      f"{worst_tors} at score {tors_min:.4f}")
print(f"  {len(offenders)} non-torsion modes score >= that:")
print(f"  {'mode':>4}  {'freq':>7}  {'score':>7}  {'lin':>5}  {'cen':>5}  "
      f"{'ant':>5}  {'sc_lr':>6}  {'sc_tb':>6}  type")
for ri in offenders[:10]:
    print(f"  {int(results['mode_idx'][ri]):4d}  {results['freq_hz'][ri]:7.2f}  "
          f"{score[ri]:7.4f}  {lin[ri]:5.2f}  {cen[ri]:5.2f}  {ant[ri]:5.2f}  "
          f"{results['score_lr'][ri]:+6.3f}  {results['score_tb'][ri]:+6.3f}  "
          f"{_classify_row(results[ri], THR)}")
