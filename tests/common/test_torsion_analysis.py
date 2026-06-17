"""
Unit tests for the geometric torsion-identification criterion
(common.torsion_analysis).

The strategy is to build SYNTHETIC modal fields with known physics on a simple
box-shaped node cloud and assert that each metric responds as designed:

  * pca_body_frame      recovers the longitudinal axis and a ground-level frame
                        regardless of how the box is oriented.
  * rigid_rotation_fit  gives ~1 for a pure rigid rotation about X, ~0 for a
                        pure translation, and is sensitive to lever-arm scaling.
  * spatial_uniformity / peak_concentration separate a spread field from a
    localised one.
  * torsion_score_v2 / scan_torsion_scores_v2 rank a clean torsion mode above a
    bending mode and a local mode.

All fields are analytic, so the tests need no FE data and run fast.
"""

import sys
from pathlib import Path

# Make the project root importable so this file works both under pytest (any
# working directory) and when run directly, matching the explore_* scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest

from common.torsion_analysis import (
    pca_body_frame,
    rigid_rotation_fit,
    spatial_uniformity,
    peak_concentration,
    _soft_gate,
    torsion_score_v2,
    scan_torsion_scores_v2,
)


# ---------------------------------------------------------------------------
# Synthetic geometry: a long box (a crude "vehicle") elongated along X.
# ---------------------------------------------------------------------------

@pytest.fixture
def box_nodes():
    """
    A long box, 4000 mm in X, 1600 in Y, 1200 in Z, on a regular grid.
    Returns (nNodes, 3) coordinates with the centroid offset from the origin
    so tests exercise the centring logic.
    """
    xs = np.linspace(0.0, 4000.0, 21)
    ys = np.linspace(-800.0, 800.0, 9)
    zs = np.linspace(0.0, 1200.0, 7)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])


def rigid_rotation_field(node_xyz, axis_y=0.0, axis_z=600.0, theta=1e-3):
    """Displacement of a rigid rotation about the X line (Y=axis_y, Z=axis_z)."""
    Y = node_xyz[:, 1] - axis_y
    Z = node_xyz[:, 2] - axis_z
    ux = np.zeros(len(node_xyz))
    uy = -theta * Z
    uz = +theta * Y
    return ux, uy, uz


# ---------------------------------------------------------------------------
# pca_body_frame
# ---------------------------------------------------------------------------

def test_pca_frame_axes_are_orthonormal(box_nodes):
    _, R, _ = pca_body_frame(box_nodes)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert pytest.approx(np.linalg.det(R), abs=1e-9) == 1.0   # proper rotation


def test_pca_frame_longitudinal_axis_is_x(box_nodes):
    """The box is longest in X, so e_long must be ~ global X."""
    _, R, _ = pca_body_frame(box_nodes)
    e_long = R[0]
    assert abs(e_long[0]) > 0.999
    assert abs(e_long[2]) < 1e-6          # levelled: no vertical component


def test_pca_frame_is_levelled(box_nodes):
    """Vertical axis is exactly global Z (axis parallel to the ground)."""
    _, R, _ = pca_body_frame(box_nodes)
    assert np.allclose(R[2], [0.0, 0.0, 1.0], atol=1e-9)


def test_pca_centre_is_robust_midpoint(box_nodes):
    centre, _, _ = pca_body_frame(box_nodes)
    # median of a symmetric regular grid is its geometric centre
    assert centre[0] == pytest.approx(2000.0, abs=1.0)
    assert centre[1] == pytest.approx(0.0, abs=1.0)
    assert centre[2] == pytest.approx(600.0, abs=1.0)


def test_pca_detects_axis_when_box_oriented_along_y():
    """A box elongated along Y must yield e_long ~ global Y (orientation-free)."""
    xs = np.linspace(-800.0, 800.0, 9)
    ys = np.linspace(0.0, 4000.0, 21)
    zs = np.linspace(0.0, 1200.0, 7)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    _, R, _ = pca_body_frame(nodes)
    assert abs(R[0, 1]) > 0.999           # longitudinal axis points along Y
    assert abs(R[0, 2]) < 1e-6            # still levelled


# ---------------------------------------------------------------------------
# rigid_rotation_fit
# ---------------------------------------------------------------------------

def test_rigid_fit_pure_rotation_is_one(box_nodes):
    """A perfect rigid rotation about the box axis fits with R^2 ~ 1."""
    # express in the body frame so the axis passes through the centre
    centre, R, _ = pca_body_frame(box_nodes)
    xyz_b = (box_nodes - centre) @ R.T
    # rotation about the body X axis (centre at origin of body frame)
    _, uy, uz = rigid_rotation_field(xyz_b, axis_y=0.0, axis_z=0.0)
    rigid_uz, rigid_uzuy = rigid_rotation_fit(xyz_b, uy, uz, n_slices=10)
    assert rigid_uz > 0.99
    assert rigid_uzuy > 0.99


def test_rigid_fit_pure_translation_is_zero(box_nodes):
    """A uniform translation in Z is not a rotation -> R^2 ~ 0."""
    uy = np.zeros(len(box_nodes))
    uz = np.full(len(box_nodes), 1.0)          # constant Uz, no lever-arm scaling
    rigid_uz, _ = rigid_rotation_fit(box_nodes, uy, uz, n_slices=10)
    assert rigid_uz < 0.05


def test_rigid_fit_offset_axis_lowers_score(box_nodes):
    """
    Fitting a rotation about a wrong (origin) axis when the true axis is offset
    in Z must score lower than fitting about the correct centred axis.
    """
    centre, R, _ = pca_body_frame(box_nodes)
    xyz_b = (box_nodes - centre) @ R.T
    _, uy, uz = rigid_rotation_field(xyz_b, axis_y=0.0, axis_z=0.0)

    # correct: body frame (axis through centre)
    r2_correct, _ = rigid_rotation_fit(xyz_b, uy, uz, n_slices=10)
    # wrong: same field but coordinates not centred in Z (axis at Z=0 global)
    r2_offset, _ = rigid_rotation_fit(box_nodes, uy, uz, n_slices=10)
    assert r2_correct > r2_offset


# ---------------------------------------------------------------------------
# energy-distribution metrics
# ---------------------------------------------------------------------------

def test_uniformity_spread_vs_local(box_nodes):
    n = len(box_nodes)
    spread = np.full(n, 1.0)
    local = np.zeros(n); local[0] = 1.0
    u_spread = spatial_uniformity(spread, spread, spread)
    u_local = spatial_uniformity(local, local, local)
    assert u_spread > 0.99
    assert u_local < 0.05


def test_peak_concentration_local_vs_spread(box_nodes):
    n = len(box_nodes)
    spread = np.full(n, 1.0)
    local = np.zeros(n); local[0] = 1.0
    assert peak_concentration(spread, spread, spread) < 0.05
    assert peak_concentration(local, local, local) > 0.9


# ---------------------------------------------------------------------------
# soft gate
# ---------------------------------------------------------------------------

def test_soft_gate_is_monotonic_and_centred():
    assert _soft_gate(0.30, 0.30) == pytest.approx(0.5, abs=1e-9)
    assert _soft_gate(0.10, 0.30) < 0.2          # below threshold -> suppressed
    assert _soft_gate(0.50, 0.30) > 0.8          # above threshold -> passed
    # strictly increasing
    vals = [_soft_gate(v, 0.30) for v in np.linspace(0, 1, 11)]
    assert all(b > a for a, b in zip(vals, vals[1:]))


# ---------------------------------------------------------------------------
# torsion_score_v2 — composite behaviour
# ---------------------------------------------------------------------------

def test_torsion_score_high_for_clean_torsion(box_nodes):
    """A linear theta(X) rigid-rotation field scores high antisym and combined."""
    centre, R, _ = pca_body_frame(box_nodes)
    xyz_b = (box_nodes - centre) @ R.T
    X = xyz_b[:, 0]
    # theta grows linearly along X (first torsion mode): theta(X) = k*X
    k = 1e-6
    theta = k * X
    uy = -theta * xyz_b[:, 2]
    uz = +theta * xyz_b[:, 1]
    ux = np.zeros(len(xyz_b))
    res = torsion_score_v2(xyz_b, ux, uy, uz, n_slices=10)
    assert res["antisym"] > 0.8
    assert res["linearity"] > 0.8
    assert res["combined"] > 0.4


def test_torsion_score_low_for_bending(box_nodes):
    """
    A vertical bending field (both sides move together in Z, no rotation) must
    score a low antisym / combined.
    """
    centre, R, _ = pca_body_frame(box_nodes)
    xyz_b = (box_nodes - centre) @ R.T
    X = xyz_b[:, 0]
    uz = np.sin(np.pi * (X - X.min()) / (X.max() - X.min()))   # same sign L/R
    uy = np.zeros(len(xyz_b))
    ux = np.zeros(len(xyz_b))
    res = torsion_score_v2(xyz_b, ux, uy, uz, n_slices=10)
    assert res["antisym"] < 0.3
    assert res["combined"] < 0.2


def test_local_mode_is_vetoed(box_nodes):
    """A field concentrated in one node is vetoed (combined == 0)."""
    n = len(box_nodes)
    ux = np.zeros(n); uy = np.zeros(n); uz = np.zeros(n)
    uz[0] = 1.0
    res = torsion_score_v2(box_nodes, ux, uy, uz, n_slices=10)
    assert res["peak"] > 0.6
    assert res["combined"] == 0.0


# ---------------------------------------------------------------------------
# scan_torsion_scores_v2 — ranking and structure
# ---------------------------------------------------------------------------

def test_scan_ranks_torsion_first(box_nodes):
    """
    Build a 3-mode set {clean torsion, bending, local} and check the scan ranks
    the torsion mode first and returns the documented fields.
    """
    n = len(box_nodes)
    centre, R, _ = pca_body_frame(box_nodes)
    xyz_b = (box_nodes - centre) @ R.T
    X, Y, Z = xyz_b[:, 0], xyz_b[:, 1], xyz_b[:, 2]

    # mode 1: clean first-torsion (linear theta along X)
    theta = 1e-6 * X
    m_tors = np.zeros(6 * n)
    m_tors[1::6] = -theta * Z          # Uy
    m_tors[2::6] = +theta * Y          # Uz

    # mode 2: vertical bending (both sides same sign in Z)
    m_bend = np.zeros(6 * n)
    m_bend[2::6] = np.sin(np.pi * (X - X.min()) / (X.max() - X.min()))

    # mode 3: local spike
    m_local = np.zeros(6 * n)
    m_local[2] = 1.0

    modes = np.column_stack([m_tors, m_bend, m_local])
    freq = np.array([40.0, 30.0, 120.0])

    # already in body frame -> disable the internal projection to test directly
    res = scan_torsion_scores_v2(xyz_b, modes, freq, n_slices=10,
                                 use_body_frame=False)
    # sorted by combined desc: torsion (mode_idx 1) must be on top
    assert int(res["mode_idx"][0]) == 1
    # all documented fields present
    for field in ("mode_idx", "freq_hz", "combined", "antisym", "linearity",
                  "centering", "uniformity", "peak", "rigid_uzuy",
                  "score_lr", "score_tb", "score_ly", "score_xvar"):
        assert field in res.dtype.names


def test_scan_body_frame_matches_manual_projection(box_nodes):
    """
    use_body_frame=True should reproduce the result of projecting the field to
    the body frame by hand and scoring with use_body_frame=False.
    """
    n = len(box_nodes)
    centre, R, _ = pca_body_frame(box_nodes)
    xyz_b = (box_nodes - centre) @ R.T
    X, Y, Z = xyz_b[:, 0], xyz_b[:, 1], xyz_b[:, 2]
    theta = 1e-6 * X
    mode = np.zeros(6 * n)
    mode[1::6] = -theta * Z
    mode[2::6] = +theta * Y
    modes = mode[:, None]
    freq = np.array([40.0])

    auto = scan_torsion_scores_v2(box_nodes, modes, freq, n_slices=10,
                                  use_body_frame=True)
    manual = scan_torsion_scores_v2(xyz_b, modes, freq, n_slices=10,
                                    use_body_frame=False)
    # box is axis-aligned, so the body frame is ~identity; scores must match
    assert auto["combined"][0] == pytest.approx(manual["combined"][0], abs=1e-6)


# ---------------------------------------------------------------------------
# Allow running this file directly (IDE "Run" button / `py thisfile.py`):
# delegate to pytest so all tests above execute and report.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
