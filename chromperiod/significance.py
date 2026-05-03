"""
significance.py — Significance testing for chromperiod CWT analysis.

Implements:
  - AR1 red-noise analytical significance (T&C 1998 eq. 18)
  - Monte Carlo permutation testing with BH FDR correction
  - Phipson-Smyth p-values for permutation tests
"""

import numpy as np
from scipy.stats import chi2
from typing import Optional, Callable

from .utils import phipson_smyth_pvalue


def ar1_significance_threshold(periods, ar1_alpha, significance_level=0.95):
    """
    Compute the AR1 red-noise significance threshold at each period scale.

    Following T&C 1998 eq. 18:
        threshold(s) = P_k(s) * chi2_crit / 2

    where P_k(s) is the AR1 background power at the frequency corresponding
    to period s, and chi2_crit is the chi^2 critical value with 2 dof.

    Parameters
    ----------
    periods : np.ndarray
        Period array in peak-index units.
    ar1_alpha : float
        Lag-1 autocorrelation coefficient.
    significance_level : float
        Significance level (default 0.95 for 95%).

    Returns
    -------
    np.ndarray
        Significance threshold at each scale.
    """
    chi2_crit = chi2.ppf(significance_level, df=2)
    thresholds = np.zeros(len(periods))
    for j, period in enumerate(periods):
        freq = 1.0 / period
        P_k = (1.0 - ar1_alpha**2) / (
            1.0 - 2.0 * ar1_alpha * np.cos(2.0 * np.pi * freq) + ar1_alpha**2
        )
        thresholds[j] = P_k * chi2_crit / 2.0
    return thresholds


def permutation_test(
    signal,
    cwt_func: Callable,
    n_permutations=1000,
    seed=42,
    fdr_method='bh',
    significance_level=0.05,
):
    """
    Monte Carlo permutation test for CWT significance.

    Generates n_permutations random shuffles of the signal, computes the
    GWS for each, and compares the observed GWS to the surrogate distribution.

    Parameters
    ----------
    signal : np.ndarray
        Standardized input signal (zero mean, unit variance).
    cwt_func : callable
        Function that takes a signal array and returns (gws, periods).
        Typically a partial of consecutive_peaks_cwt with fixed parameters.
    n_permutations : int
        Number of permutation surrogates (default 1000).
    seed : int
        Random seed for reproducibility.
    fdr_method : str
        Multiple testing correction method ('bh' for Benjamini-Hochberg).
    significance_level : float
        FDR significance threshold (default 0.05).

    Returns
    -------
    dict with keys:
        'observed_gws' : np.ndarray — observed GWS
        'surrogate_mean' : np.ndarray — mean surrogate GWS
        'surrogate_p95' : np.ndarray — 95th percentile surrogate GWS
        'surrogate_p99' : np.ndarray — 99th percentile surrogate GWS
        'pvalues' : np.ndarray — Phipson-Smyth p-values per scale
        'qvalues' : np.ndarray — BH-corrected q-values per scale
        'significant' : np.ndarray of bool — significant scales at FDR threshold
        'n_significant_bh05' : int — number of significant scales at BH q<0.05
        'n_significant_bh01' : int — number of significant scales at BH q<0.01
        'surrogate_matrix' : np.ndarray — full surrogate GWS matrix (n_perm x n_scales)
    """
    rng = np.random.default_rng(seed)

    # Observed GWS
    observed_gws, periods = cwt_func(signal)
    n_scales = len(observed_gws)

    # Surrogate GWS matrix
    surrogate_matrix = np.zeros((n_permutations, n_scales))
    for i in range(n_permutations):
        shuffled = rng.permutation(signal)
        surrogate_gws, _ = cwt_func(shuffled)
        surrogate_matrix[i, :] = surrogate_gws

    # Compute p-values (Phipson-Smyth)
    pvalues = np.zeros(n_scales)
    for j in range(n_scales):
        rank = np.sum(surrogate_matrix[:, j] >= observed_gws[j])
        pvalues[j] = phipson_smyth_pvalue(rank, n_permutations)

    # BH FDR correction
    qvalues = _bh_correction(pvalues)

    significant = qvalues < significance_level

    return {
        'observed_gws': observed_gws,
        'surrogate_mean': np.mean(surrogate_matrix, axis=0),
        'surrogate_p95': np.percentile(surrogate_matrix, 95, axis=0),
        'surrogate_p99': np.percentile(surrogate_matrix, 99, axis=0),
        'pvalues': pvalues,
        'qvalues': qvalues,
        'significant': significant,
        'n_significant_bh05': int(np.sum(qvalues < 0.05)),
        'n_significant_bh01': int(np.sum(qvalues < 0.01)),
        'surrogate_matrix': surrogate_matrix,
    }


def _bh_correction(pvalues):
    """
    Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    pvalues : np.ndarray
        Array of p-values.

    Returns
    -------
    np.ndarray
        BH-adjusted q-values.
    """
    n = len(pvalues)
    order = np.argsort(pvalues)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)

    qvalues = pvalues * n / ranks
    # Enforce monotonicity (cumulative minimum from the right)
    qvalues_sorted = qvalues[order]
    for i in range(n - 2, -1, -1):
        qvalues_sorted[i] = min(qvalues_sorted[i], qvalues_sorted[i + 1])
    qvalues[order] = qvalues_sorted

    return np.minimum(qvalues, 1.0)


def compute_sig95(power, significance_mask, coi_mask):
    """
    Compute sig95: fraction of outside-COI points exceeding significance threshold.

    Parameters
    ----------
    power : np.ndarray, shape (n_scales, n_peaks)
        Wavelet power matrix.
    significance_mask : np.ndarray of bool, shape (n_scales, n_peaks)
        True where power exceeds threshold.
    coi_mask : np.ndarray of bool, shape (n_scales, n_peaks)
        True where point is outside the COI.

    Returns
    -------
    float
        Fraction of outside-COI points that are significant.
    """
    n_outside = coi_mask.sum()
    if n_outside == 0:
        return 0.0
    n_sig = (significance_mask & coi_mask).sum()
    return float(n_sig) / float(n_outside)


def white_noise_false_positive_rate(
    n_signals=100,
    n_peaks=8082,
    wavelet='paul',
    order=2,
    n_scales=80,
    period_min=10,
    period_max=7000,
    significance_level=0.95,
    seed=0,
):
    """
    Estimate the false positive rate of the AR1 significance test on white noise.

    Generates n_signals independent white noise signals and computes the mean
    sig95 value. For a well-calibrated test, this should be approximately
    (1 - significance_level) = 5% for significance_level=0.95.

    Parameters
    ----------
    n_signals : int
        Number of white noise signals to test (default 100).
    n_peaks : int
        Length of each signal (default 8082, matching MCF-7 chrX).
    (other parameters as in consecutive_peaks_cwt)
    seed : int
        Random seed.

    Returns
    -------
    dict with keys:
        'mean_sig95' : float — mean sig95 across all signals
        'std_sig95' : float — standard deviation of sig95
        'all_sig95' : list of float — sig95 for each signal
        'expected_fp_rate' : float — 1 - significance_level
    """
    from .cwt import consecutive_peaks_cwt
    import pandas as pd

    rng = np.random.default_rng(seed)
    all_sig95 = []

    for i in range(n_signals):
        signal = rng.standard_normal(n_peaks)
        # Create a minimal DataFrame
        df = pd.DataFrame({
            'chrom': ['chrSim'] * n_peaks,
            'start': np.arange(n_peaks) * 10000,
            'end': np.arange(n_peaks) * 10000 + 200,
            'signalValue': signal,
        })
        result = consecutive_peaks_cwt(
            df,
            chromosome='chrSim',
            wavelet=wavelet,
            order=order,
            n_scales=n_scales,
            period_min=period_min,
            period_max=period_max,
            significance_level=significance_level,
        )
        all_sig95.append(result.sig95)

    return {
        'mean_sig95': float(np.mean(all_sig95)),
        'std_sig95': float(np.std(all_sig95)),
        'all_sig95': all_sig95,
        'expected_fp_rate': 1.0 - significance_level,
    }
