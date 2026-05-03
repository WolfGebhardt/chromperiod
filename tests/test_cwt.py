"""
test_cwt.py — Tests for the core CWT implementation.

Tests:
  1. White noise false positive rate: mean sig95 should be within 3-7%
     (targeting 5%) for white noise signals.
  2. Sine wave recovery: dominant period should be within 10% of true period.
  3. Randomization control: shuffling a real-like signal reduces sig95 to ~5%.
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chromperiod import consecutive_peaks_cwt


def make_peaks_df(signal, chrom='chrTest', spacing_bp=10000):
    """Helper: create a minimal peaks DataFrame from a signal array."""
    N = len(signal)
    positions = np.arange(N) * spacing_bp
    return pd.DataFrame({
        'chrom': [chrom] * N,
        'start': positions,
        'end': positions + 200,
        'signalValue': signal,
    })


class TestWhiteNoiseFalsePositiveRate:
    """
    Test 1: White noise false positive rate.

    Generate 100 white noise signals (N=8082), run CWT, verify mean sig95
    is within 3-7% (targeting 5% for significance_level=0.95).
    """

    def test_white_noise_fp_rate(self):
        rng = np.random.default_rng(0)
        n_signals = 100
        n_peaks = 8082
        sig95_values = []

        for i in range(n_signals):
            signal = rng.standard_normal(n_peaks)
            df = make_peaks_df(signal)
            result = consecutive_peaks_cwt(
                df,
                chromosome='chrTest',
                wavelet='paul',
                order=2,
                n_scales=80,
                period_min=10,
                period_max=7000,
                significance_level=0.95,
            )
            sig95_values.append(result.sig95)

        mean_sig95 = np.mean(sig95_values)
        std_sig95 = np.std(sig95_values)

        print(f"\nWhite noise FP rate: mean={mean_sig95:.3f}, std={std_sig95:.3f}")
        print(f"Expected: ~0.05 (5%)")

        # Mean sig95 should be within 3-7% (3 sigma range around 5%)
        assert 0.03 <= mean_sig95 <= 0.07, (
            f"White noise mean sig95 = {mean_sig95:.3f}, expected 0.03-0.07. "
            f"This indicates the AR1 significance threshold is miscalibrated."
        )

    def test_white_noise_fp_rate_small(self):
        """Quick version with 20 signals for faster CI testing."""
        rng = np.random.default_rng(42)
        n_signals = 20
        n_peaks = 2000
        sig95_values = []

        for i in range(n_signals):
            signal = rng.standard_normal(n_peaks)
            df = make_peaks_df(signal)
            result = consecutive_peaks_cwt(
                df,
                chromosome='chrTest',
                wavelet='paul',
                order=2,
                n_scales=60,
                period_min=10,
                period_max=1000,
                significance_level=0.95,
            )
            sig95_values.append(result.sig95)

        mean_sig95 = np.mean(sig95_values)
        print(f"\nSmall white noise FP rate: mean={mean_sig95:.3f}")

        # Looser bounds for small sample
        assert 0.01 <= mean_sig95 <= 0.12, (
            f"White noise mean sig95 = {mean_sig95:.3f}, expected 0.01-0.12."
        )


class TestSineWaveRecovery:
    """
    Test 2: Sine wave recovery.

    Generate a sine wave with period 500 peaks in Gaussian noise (SNR=1),
    verify recovered dominant period is within 10% of 500.
    """

    def test_sine_wave_recovery(self):
        rng = np.random.default_rng(123)
        n_peaks = 4000
        true_period = 500  # peak-index units
        snr = 1.0  # signal-to-noise ratio

        x = np.arange(n_peaks)
        signal = np.sin(2 * np.pi * x / true_period) + (1.0 / snr) * rng.standard_normal(n_peaks)

        df = make_peaks_df(signal)
        result = consecutive_peaks_cwt(
            df,
            chromosome='chrTest',
            wavelet='paul',
            order=2,
            n_scales=80,
            period_min=10,
            period_max=2000,
            significance_level=0.95,
        )

        recovered_period = result.dominant_period
        relative_error = abs(recovered_period - true_period) / true_period

        print(f"\nSine wave recovery: true={true_period}, recovered={recovered_period:.1f}, "
              f"error={relative_error:.1%}")

        assert relative_error <= 0.10, (
            f"Recovered period {recovered_period:.1f} is more than 10% from "
            f"true period {true_period}. Relative error: {relative_error:.1%}"
        )

    def test_sine_wave_significant(self):
        """Sine wave should produce significant periodicity."""
        rng = np.random.default_rng(456)
        n_peaks = 3000
        true_period = 300

        x = np.arange(n_peaks)
        signal = 2.0 * np.sin(2 * np.pi * x / true_period) + rng.standard_normal(n_peaks)

        df = make_peaks_df(signal)
        result = consecutive_peaks_cwt(
            df,
            chromosome='chrTest',
            wavelet='paul',
            order=2,
            n_scales=60,
            period_min=10,
            period_max=1500,
        )

        print(f"\nSine wave sig95: {result.sig95:.1%}")
        assert result.sig95 > 0.10, (
            f"Sine wave sig95 = {result.sig95:.1%}, expected > 10%. "
            f"The CWT should detect the periodic signal."
        )

    def test_multiple_wavelets(self):
        """Sine wave recovery should work for all three wavelet types."""
        rng = np.random.default_rng(789)
        n_peaks = 3000
        true_period = 400

        x = np.arange(n_peaks)
        signal = np.sin(2 * np.pi * x / true_period) + rng.standard_normal(n_peaks)
        df = make_peaks_df(signal)

        for wavelet, order in [('paul', 4), ('morlet', 6), ('dog', 2)]:
            result = consecutive_peaks_cwt(
                df,
                chromosome='chrTest',
                wavelet=wavelet,
                order=order,
                n_scales=60,
                period_min=10,
                period_max=2000,
            )
            recovered = result.dominant_period
            error = abs(recovered - true_period) / true_period
            print(f"\n{wavelet} m={order}: recovered={recovered:.1f}, error={error:.1%}")
            assert error <= 0.20, (
                f"{wavelet} wavelet: recovered period {recovered:.1f} is more than "
                f"20% from true period {true_period}."
            )


class TestRandomizationControl:
    """
    Test 3: Randomization control.

    Verify that shuffling a real-like signal (sine wave + noise) reduces
    sig95 to near 5%.
    """

    def test_randomization_reduces_sig95(self):
        rng = np.random.default_rng(999)
        n_peaks = 3000
        true_period = 300

        x = np.arange(n_peaks)
        signal = 2.0 * np.sin(2 * np.pi * x / true_period) + rng.standard_normal(n_peaks)
        df = make_peaks_df(signal)

        # Original signal
        result_orig = consecutive_peaks_cwt(
            df, chromosome='chrTest', n_scales=60, period_min=10, period_max=1500
        )

        # Shuffled signal
        df_rand = df.copy()
        df_rand['signalValue'] = rng.permutation(signal)
        result_rand = consecutive_peaks_cwt(
            df_rand, chromosome='chrTest', n_scales=60, period_min=10, period_max=1500
        )

        print(f"\nOriginal sig95: {result_orig.sig95:.1%}")
        print(f"Randomized sig95: {result_rand.sig95:.1%}")

        # Original should be substantially more significant than randomized
        assert result_orig.sig95 > result_rand.sig95 + 0.05, (
            f"Original sig95 ({result_orig.sig95:.1%}) should be substantially "
            f"higher than randomized sig95 ({result_rand.sig95:.1%})."
        )

        # Randomized should be near 5%
        assert result_rand.sig95 <= 0.15, (
            f"Randomized sig95 = {result_rand.sig95:.1%}, expected <= 15%. "
            f"Shuffling should destroy the periodic structure."
        )

    def test_randomization_flat_gws(self):
        """Randomized signal should have a flat GWS (no dominant peak)."""
        rng = np.random.default_rng(111)
        n_peaks = 2000
        true_period = 250

        x = np.arange(n_peaks)
        signal = 3.0 * np.sin(2 * np.pi * x / true_period) + rng.standard_normal(n_peaks)

        df_rand = pd.DataFrame({
            'chrom': ['chrTest'] * n_peaks,
            'start': np.arange(n_peaks) * 10000,
            'end': np.arange(n_peaks) * 10000 + 200,
            'signalValue': rng.permutation(signal),
        })

        result = consecutive_peaks_cwt(
            df_rand, chromosome='chrTest', n_scales=60, period_min=10, period_max=1000
        )

        # GWS should be relatively flat: max/mean ratio should be small
        valid_gws = result.gws[np.isfinite(result.gws)]
        if len(valid_gws) > 0:
            gws_ratio = valid_gws.max() / (valid_gws.mean() + 1e-10)
            print(f"\nRandomized GWS max/mean ratio: {gws_ratio:.2f}")
            # For white noise, this ratio should be modest (< 5)
            assert gws_ratio < 10, (
                f"Randomized GWS max/mean ratio = {gws_ratio:.2f}, expected < 10. "
                f"The GWS should be relatively flat for shuffled data."
            )


class TestCWTProperties:
    """Additional tests for CWT implementation correctness."""

    def test_result_attributes(self):
        """CWTResult should have all required attributes."""
        rng = np.random.default_rng(0)
        signal = rng.standard_normal(500)
        df = make_peaks_df(signal)
        result = consecutive_peaks_cwt(df, chromosome='chrTest',
                                        n_scales=40, period_min=10, period_max=200)

        required_attrs = [
            'power', 'significance', 'gws', 'periods', 'dominant_period',
            'sig95', 'ar1_alpha', 'coi', 'coi_frac', 'scales', 'signal',
            'ar1_threshold', 'wavelet', 'order', 'chromosome', 'n_peaks',
            'dom_period_near_coi_edge',
        ]
        for attr in required_attrs:
            assert hasattr(result, attr), f"CWTResult missing attribute: {attr}"

    def test_power_shape(self):
        """Power matrix should have shape (n_scales, n_peaks)."""
        rng = np.random.default_rng(0)
        n_peaks = 300
        n_scales = 40
        signal = rng.standard_normal(n_peaks)
        df = make_peaks_df(signal)
        result = consecutive_peaks_cwt(df, chromosome='chrTest',
                                        n_scales=n_scales, period_min=10, period_max=100)
        assert result.power.shape == (n_scales, n_peaks), (
            f"Power shape {result.power.shape} != ({n_scales}, {n_peaks})"
        )

    def test_power_nonnegative(self):
        """Power should be non-negative."""
        rng = np.random.default_rng(0)
        signal = rng.standard_normal(500)
        df = make_peaks_df(signal)
        result = consecutive_peaks_cwt(df, chromosome='chrTest',
                                        n_scales=40, period_min=10, period_max=200)
        assert np.all(result.power >= 0), "Power matrix contains negative values."

    def test_sig95_range(self):
        """sig95 should be in [0, 1]."""
        rng = np.random.default_rng(0)
        signal = rng.standard_normal(500)
        df = make_peaks_df(signal)
        result = consecutive_peaks_cwt(df, chromosome='chrTest',
                                        n_scales=40, period_min=10, period_max=200)
        assert 0.0 <= result.sig95 <= 1.0, f"sig95 = {result.sig95} out of [0, 1]"

    def test_ar1_range(self):
        """AR1 coefficient should be in [0, 1)."""
        rng = np.random.default_rng(0)
        signal = rng.standard_normal(500)
        df = make_peaks_df(signal)
        result = consecutive_peaks_cwt(df, chromosome='chrTest',
                                        n_scales=40, period_min=10, period_max=200)
        assert 0.0 <= result.ar1_alpha < 1.0, f"AR1 = {result.ar1_alpha} out of [0, 1)"

    def test_dominant_period_in_range(self):
        """Dominant period should be within [period_min, period_max]."""
        rng = np.random.default_rng(0)
        signal = rng.standard_normal(1000)
        df = make_peaks_df(signal)
        period_min, period_max = 20, 400
        result = consecutive_peaks_cwt(df, chromosome='chrTest',
                                        n_scales=40, period_min=period_min,
                                        period_max=period_max)
        assert period_min <= result.dominant_period <= period_max, (
            f"Dominant period {result.dominant_period:.1f} outside "
            f"[{period_min}, {period_max}]"
        )

    def test_median_spacing_computed(self):
        """Median spacing should be computed from peak positions."""
        rng = np.random.default_rng(0)
        n_peaks = 200
        signal = rng.standard_normal(n_peaks)
        spacing = 15000  # bp
        df = pd.DataFrame({
            'chrom': ['chrTest'] * n_peaks,
            'start': np.arange(n_peaks) * spacing,
            'end': np.arange(n_peaks) * spacing + 200,
            'signalValue': signal,
        })
        result = consecutive_peaks_cwt(df, chromosome='chrTest',
                                        n_scales=30, period_min=10, period_max=100)
        assert result.median_spacing_bp is not None
        assert abs(result.median_spacing_bp - spacing) < spacing * 0.1, (
            f"Median spacing {result.median_spacing_bp:.0f} != expected {spacing}"
        )
