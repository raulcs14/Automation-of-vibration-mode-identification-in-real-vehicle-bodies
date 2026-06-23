"""
[VERIFICATION] Unit tests for the generic MAC core (common.mac_core).

All inputs are small analytic vectors with known correlations, so the tests need
no FE data and run instantly.  They assert the mathematical properties the MAC
must satisfy:

  * compute_mac      : values in [0, 1]; self-MAC = 1 on the diagonal; orthogonal
                       shapes -> 0; collinear shapes -> 1; sign/scale invariance;
                       a weighting matrix W changes the inner product as expected.
  * best_mac_per_mode: picks the row-wise maximum over references.
  * select_top_modes : returns the indices of the n best modes across variants,
                       sorted ascending.

The interactive/visual MAC explorer lives at scripts/seat/view_mac.py.
"""

import sys
from pathlib import Path

# Repo root on sys.path so `common` imports work under pytest from any CWD.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest

from common.mac_core import compute_mac, best_mac_per_mode, select_top_modes


# ---------------------------------------------------------------------------
# compute_mac — core mathematical properties
# ---------------------------------------------------------------------------

def test_self_mac_is_identity():
    """MAC of a set of orthogonal shapes against itself is the identity."""
    Phi = np.eye(4)
    mac = compute_mac(Phi, Phi)
    assert mac.shape == (4, 4)
    np.testing.assert_allclose(mac, np.eye(4), atol=1e-12)


def test_mac_values_within_unit_interval():
    """Every MAC value lies in [0, 1] for arbitrary real shapes."""
    rng = np.random.default_rng(0)
    Phi = rng.standard_normal((30, 5))
    Psi = rng.standard_normal((30, 4))
    mac = compute_mac(Phi, Psi)
    assert mac.shape == (5, 4)
    assert np.all(mac >= -1e-12) and np.all(mac <= 1.0 + 1e-12)


def test_collinear_shapes_give_mac_one():
    """A mode and a scaled+negated copy of it are perfectly correlated (MAC=1)."""
    rng = np.random.default_rng(1)
    v = rng.standard_normal((20, 1))
    mac = compute_mac(v, -3.7 * v)
    assert mac[0, 0] == pytest.approx(1.0, abs=1e-12)


def test_orthogonal_shapes_give_mac_zero():
    """Orthogonal shapes are uncorrelated (MAC=0)."""
    a = np.array([[1.0], [0.0], [0.0]])
    b = np.array([[0.0], [1.0], [0.0]])
    assert compute_mac(a, b)[0, 0] == pytest.approx(0.0, abs=1e-12)


def test_mac_is_scale_and_sign_invariant():
    """Scaling either argument leaves the MAC unchanged."""
    rng = np.random.default_rng(2)
    Phi = rng.standard_normal((15, 3))
    Psi = rng.standard_normal((15, 3))
    base   = compute_mac(Phi, Psi)
    scaled = compute_mac(5.0 * Phi, -2.0 * Psi)
    np.testing.assert_allclose(base, scaled, atol=1e-12)


def test_weighting_changes_inner_product():
    """A non-identity weighting W reproduces the W-weighted MAC by hand."""
    rng = np.random.default_rng(3)
    Phi = rng.standard_normal((6, 2))
    Psi = rng.standard_normal((6, 2))
    d = np.abs(rng.standard_normal(6)) + 0.1
    W = np.diag(d)

    mac = compute_mac(Phi, Psi, W)

    # reference: explicit W-weighted MAC for entry (i, j)
    for i in range(2):
        for j in range(2):
            num = (Phi[:, i] @ W @ Psi[:, j]) ** 2
            den = (Phi[:, i] @ W @ Phi[:, i]) * (Psi[:, j] @ W @ Psi[:, j])
            assert mac[i, j] == pytest.approx(num / den, rel=1e-10)


def test_identity_weight_matches_unweighted():
    """W = I gives the same result as W = None."""
    rng = np.random.default_rng(4)
    Phi = rng.standard_normal((8, 3))
    Psi = rng.standard_normal((8, 3))
    np.testing.assert_allclose(
        compute_mac(Phi, Psi, np.eye(8)),
        compute_mac(Phi, Psi),
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# best_mac_per_mode / select_top_modes
# ---------------------------------------------------------------------------

def test_best_mac_per_mode_takes_row_max():
    mac = np.array([[0.1, 0.9, 0.3],
                    [0.8, 0.2, 0.5],
                    [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(best_mac_per_mode(mac), [0.9, 0.8, 0.0])


def test_select_top_modes_picks_best_across_variants_sorted():
    # mode 4 best in variant A, mode 1 best in variant B; n=2 -> indices [1, 4]
    variants = {
        "A": np.array([0.1, 0.2, 0.3, 0.4, 0.95]),
        "B": np.array([0.9, 0.05, 0.1, 0.2, 0.3]),
    }
    top = select_top_modes(variants, n=2)
    assert top.tolist() == [0, 4]
    # result is sorted ascending
    assert list(top) == sorted(top)


# ---------------------------------------------------------------------------
# Allow running this file directly (IDE "Run" button / `py thisfile.py`):
# delegate to pytest so all tests above execute and report.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
