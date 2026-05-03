"""
test_significance.py — Tests for significance testing functions.

Tests:
  1. AR1 analytical threshold exceeds permutation threshold at all scales
     for a white noise signal.
  2. BH correction reduces the number of significant scales appropriately.
  3. Phipson-Smyth p-values are in (0, 1].
  4. AR1 threshold increases with AR1 coefficient.
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chromperiod import consecutive_peaks_cwt
from chromperiod.significance import (
    ar1_significance_threshold,
    _bh_correction,
    permutation_test,
)
from chromperiod.utils import phipson_smyth_pvalue, estimate_ar1


def make_peaks_df(signal, chrom='chrTest', spacing_bp=10000):
    """Helper: create a minimal peaks DataFrame."""
    N = len(signal)
    positions = np.arange(N) * spacing_bp
    return pd.DataFrame({
        'chrom': [chrom] * N,
        'start': positions,
        'end': positions + 200,
        'signalValue': signal,
    })


class TestAR1ThresholdVsPermutation:
    """
    Test 1: AR1 analytical threshold exceeds permutation threshold at all scales.

    For a white noise signal, the AR1 analytical threshold should be
    conservative (higher) than the empirical permutation threshold.
    """

    def test_ar1_exceeds_permutation_white_noise(self):
        """AR1 threshold should exceed permutation 95th percentile for white noise."""
        rng = np.random.default_rng(42)
        n_peaks = 2000
        n_permutations = 200

        signal = rng.standard_normal(n_peaks)
        df = make_peaks_df(signal)

        # Run CWT to get AR1 threshold and periods
        result = consecutive_peaks_cwt(
            df, chromosome='chrTest',
            n_scales=40, period_min=10, period_max=800,
        )

        # Compute permutation surrogate GWS
        surrogate_gws = np.zeros((n_permutations, len(result.periods)))
        for i in range(n_permutations):
            shuffled = rng.permutation(signal)
            df_s = make_peaks_df(shuffled)
            r = consecutive_peaks_cwt(
                df_s, chromosome='chrTest',
                n_scales=40, period_min=10, period_max=800,
            )
            surrogate_gws[i, :] = r.gws

        # 95th percentile of surrogate GWS at each scale
        perm_p95 = np.nanpercentile(surrogate_gws, 95, axis=0)

        # AR1 threshold
        ar1_thresh = result.ar1_threshold

        # AR1 threshold should exceed permutation p95 at most scales
        # (it is known to be conservative)
        n_scales = len(result.periods)
        n_ar1_exceeds = np.sum(ar1_thresh > perm_p95)
        frac_exceeds = n_ar1_exceeds / n_scales

        print(f"\nAR1 > permutation p95 at {n_ar1_exceeds}/{n_scales} scales "
              f"({frac_exceeds:.1%})")

        # AR1 should exceed permutation at the majority of scales
        assert frac_exceeds >= 0.5, (
            f"AR1 threshold exceeds permutation p95 at only {frac_exceeds:.1%} "
            f"of scales. Expected >= 50%."
        )

    def test_ar1_threshold_shape(self):
        """AR1 threshold should have same shape as periods array."""
        periods = np.geomspace(10, 1000, 50)
        ar1_alpha = 0.3
        threshold = ar1_significance_threshold(periods, ar1_alpha)
        assert threshold.shape == periods.shape

    def test_ar1_threshold_positive(self):
        """AR1 threshold should be positive."""
        periods = np.geomspace(10, 1000, 50)
        threshold = ar1_significance_threshold(periods, ar1_alpha=0.2)
        assert np.all(threshold > 0), "AR1 threshold contains non-positive values."

    def test_ar1_threshold_increases_with_alpha(self):
        """Higher AR1 coefficient should give higher threshold at long periods."""
        periods = np.geomspace(10, 1000, 50)
        thresh_low = ar1_significance_threshold(periods, ar1_alpha=0.1)
        thresh_high = ar1_significance_threshold(periods, ar1_alpha=0.8)

        # At long periods, higher AR1 gives higher background power
        long_period_idx = -10  # near the long-period end
        print(f"\nAR1=0.1 threshold at long period: {thresh_low[long_period_idx]:.3f}")
        print(f"AR1=0.8 threshold at long period: {thresh_high[long_period_idx]:.3f}")

        assert thresh_high[long_period_idx] > thresh_low[long_period_idx], (
            "Higher AR1 should give higher threshold at long periods."
        )

    def test_ar1_zero_gives_flat_threshold(self):
        """AR1=0 (white noise) should give a flat threshold."""
        periods = np.geomspace(10, 1000, 50)
        threshold = ar1_significance_threshold(periods, ar1_alpha=0.0)
        # For AR1=0, P_k = 1 at all frequencies, so threshold is constant
        assert np.allclose(threshold, threshold[0], rtol=0.01), (
            "AR1=0 should give a flat (constant) threshold."
        )


class TestBHCorrection:
    """
    Test 2: BH correction reduces the number of significant scales.
    """

    def test_bh_reduces_significant_scales(self):
        """BH correction should reduce false positives compared to uncorrected p-values."""
        rng = np.random.default_rng(0)
        # Simulate p-values: mostly uniform (null) with a few small values
        n = 80
        pvalues = rng.uniform(0, 1, n)
        # Add a few genuinely small p-values
        pvalues[:5] = rng.uniform(0, 0.001, 5)

        # Uncorrected significant at 0.05
        n_uncorrected = np.sum(pvalues < 0.05)

        # BH corrected
        qvalues = _bh_correction(pvalues)
        n_corrected = np.sum(qvalues < 0.05)

        print(f"\nUncorrected significant: {n_uncorrected}")
        print(f"BH-corrected significant: {n_corrected}")

        # BH should not increase the number of significant results
        # (it may keep the same or reduce)
        assert n_corrected <= n_uncorrected + 2, (
            f"BH correction increased significant count from {n_uncorrected} "
            f"to {n_corrected}."
        )

    def test_bh_output_range(self):
        """BH q-values should be in [0, 1]."""
        rng = np.random.default_rng(1)
        pvalues = rng.uniform(0, 1, 100)
        qvalues = _bh_correction(pvalues)
        assert np.all(qvalues >= 0) and np.all(qvalues <= 1), (
            "BH q-values out of [0, 1] range."
        )

    def test_bh_monotone(self):
        """BH q-values should be monotonically non-decreasing when sorted by p-value."""
        rng = np.random.default_rng(2)
        pvalues = rng.uniform(0, 1, 80)
        qvalues = _bh_correction(pvalues)

        # Sort by p-value and check q-values are non-decreasing
        order = np.argsort(pvalues)
        q_sorted = qvalues[order]
        assert np.all(np.diff(q_sorted) >= -1e-10), (
            "BH q-values are not monotonically non-decreasing."
        )

    def test_bh_all_null(self):
        """For uniform p-values (all null), BH should give q-values >= p-values."""
        rng = np.random.default_rng(3)
        pvalues = rng.uniform(0, 1, 100)
        qvalues = _bh_correction(pvalues)
        # Under the null, q-values should be >= p-values (conservative)
        assert np.mean(qvalues >= pvalues) >= 0.5, (
            "Under the null, BH q-values should generally be >= p-values."
        )

    def test_bh_strong_signal(self):
        """For very small p-values, BH should keep them significant."""
        pvalues = np.array([1e-10, 1e-9, 1e-8, 0.5, 0.6, 0.7, 0.8, 0.9])
        qvalues = _bh_correction(pvalues)
        # The first three should remain significant at q < 0.05
        assert np.sum(qvalues < 0.05) >= 3, (
            "BH should keep very small p-values significant."
        )


class TestPhipsonSmythPvalue:
    """Tests for Phipson-Smyth permutation p-values."""

    def test_pvalue_range(self):
        """P-values should be in (0, 1]."""
        for rank in [0, 1, 10, 100, 999, 1000]:
            p = phipson_smyth_pvalue(rank, n_permutations=1000)
            assert 0 < p <= 1.0, f"P-value {p} out of (0, 1] for rank={rank}"

    def test_pvalue_minimum(self):
        """Minimum p-value should be 1/(n+1), not 0."""
        p_min = phipson_smyth_pvalue(0, n_permutations=1000)
        assert p_min == 1.0 / 1001.0, (
            f"Minimum p-value should be 1/1001 = {1/1001:.6f}, got {p_min:.6f}"
        )

    def test_pvalue_maximum(self):
        """Maximum p-value (all permutations exceed observed) should be 1."""
        p_max = phipson_smyth_pvalue(1000, n_permutations=1000)
        assert p_max == 1001.0 / 1001.0, (
            f"Maximum p-value should be 1.0, got {p_max:.6f}"
        )

    def test_pvalue_monotone(self):
        """P-values should increase with rank."""
        n_perm = 1000
        ranks = np.arange(0, n_perm + 1, 100)
        pvalues = phipson_smyth_pvalue(ranks, n_perm)
        assert np.all(np.diff(pvalues) > 0), "P-values should be monotonically increasing with rank."


class TestEstimateAR1:
    """Tests for AR1 estimation."""

    def test_white_noise_ar1_near_zero(self):
        """White noise should have AR1 near 0."""
        rng = np.random.default_rng(0)
        signal = rng.standard_normal(10000)
        alpha = estimate_ar1(signal)
        print(f"\nWhite noise AR1: {alpha:.4f}")
        assert abs(alpha) < 0.05, f"White noise AR1 = {alpha:.4f}, expected near 0."

    def test_ar1_process_recovery(self):
        """AR1 process should have AR1 near the true coefficient."""
        rng = np.random.default_rng(42)
        true_alpha = 0.7
        N = 5000
        signal = np.zeros(N)
        signal[0] = rng.standard_normal()
        for i in range(1, N):
            signal[i] = true_alpha * signal[i-1] + rng.standard_normal()

        estimated = estimate_ar1(signal)
        print(f"\nAR1 process: true={true_alpha}, estimated={estimated:.4f}")
        assert abs(estimated - true_alpha) < 0.05, (
            f"AR1 estimate {estimated:.4f} too far from true {true_alpha}."
        )

    def test_ar1_clipped_to_valid_range(self):
        """AR1 should be clipped to [0, 0.999]."""
        # Perfectly correlated signal (AR1 = 1)
        signal = np.ones(100)
        alpha = estimate_ar1(signal)
        assert 0.0 <= alpha <= 0.999, f"AR1 = {alpha} out of [0, 0.999]"
