"""
utils.py — Utility functions for chromperiod.

Includes:
  - COI computation (T&C eq. 25)
  - AR1 estimation
  - Run-length analysis for A/B compartment periodicity
  - Miscellaneous helpers
"""

import numpy as np
from typing import Optional


def compute_coi(N, delta_t, coi_factor):
    """
    Compute the cone of influence (COI) for a signal of length N.

    The COI at position n is the maximum period that is not edge-affected,
    following T&C eq. 25:
        COI(n) = coi_factor * delta_t * min(n, N-1-n)

    where coi_factor = sqrt(2) * lambda_psi for Paul wavelets.

    Parameters
    ----------
    N : int
        Signal length (number of peaks).
    delta_t : float
        Sampling interval (1.0 for consecutive-peak representation).
    coi_factor : float
        Wavelet-specific COI factor (lambda_psi * sqrt(2) for Paul).

    Returns
    -------
    coi : np.ndarray, shape (N,)
        COI in period units at each position.
    """
    n = np.arange(N)
    coi = coi_factor * delta_t * np.maximum(1, np.minimum(n + 1, N - n))
    return coi


def compute_coi_frac(N, n_scales, coi, periods):
    """
    Compute the COI-accessible fraction of the scalogram.

    Defined as the proportion of (scale × position) points that lie
    outside the cone of influence.

    Parameters
    ----------
    N : int
        Number of peaks.
    n_scales : int
        Number of scales.
    coi : np.ndarray, shape (N,)
        COI in period units at each position.
    periods : np.ndarray, shape (n_scales,)
        Period array.

    Returns
    -------
    float
        Fraction of scalogram area outside COI.
    """
    total = N * n_scales
    outside = 0
    for j in range(n_scales):
        outside += np.sum(periods[j] < coi)
    return float(outside) / float(total)


def estimate_ar1(signal):
    """
    Estimate the lag-1 autocorrelation coefficient of a signal.

    Uses the biased estimator (T&C eq. 2):
        alpha = r1 / r0
    where r1 is the lag-1 autocovariance and r0 is the variance.

    Parameters
    ----------
    signal : np.ndarray
        Input signal (should be zero-mean).

    Returns
    -------
    float
        Lag-1 autocorrelation coefficient, clipped to [0, 0.999].
    """
    N = len(signal)
    if N < 3:
        return 0.0
    x = signal - np.mean(signal)
    r0 = np.dot(x, x) / N
    r1 = np.dot(x[:-1], x[1:]) / N
    if r0 == 0:
        return 0.0
    alpha = r1 / r0
    return float(np.clip(alpha, 0.0, 0.999))


def runlength_analysis(signal_values, compartment_labels, median_spacing_bp=None):
    """
    Compute A/B compartment run-length statistics.

    Counts consecutive same-compartment peaks between each A->B or B->A
    transition. The mean A+B run-length period provides a wavelet-independent
    estimate of the compartmental repeat unit in consecutive-peak space.

    Parameters
    ----------
    signal_values : np.ndarray
        Accessibility signal values (not used for run counting, kept for
        compatibility).
    compartment_labels : np.ndarray of str or int
        Compartment label for each peak. Use 'A'/'B' or 1/0 or True/False.
        Peaks labeled 'A' or 1 or True are compartment A.
    median_spacing_bp : float or None
        Median inter-peak spacing in bp. If provided, converts run lengths
        to Mbp.

    Returns
    -------
    dict with keys:
        'mean_A_run' : float — mean length of A-compartment runs (peaks)
        'mean_B_run' : float — mean length of B-compartment runs (peaks)
        'mean_AB_period' : float — mean A+B period (peaks)
        'mean_AB_period_mbp' : float or None — mean A+B period in Mbp
        'n_transitions' : int — number of A<->B transitions
        'A_runs' : list of int — lengths of all A runs
        'B_runs' : list of int — lengths of all B runs
    """
    labels = np.asarray(compartment_labels)

    # Normalize to boolean: True = A compartment
    if labels.dtype.kind in ('U', 'S', 'O'):
        is_A = np.array([str(l).upper() == 'A' for l in labels])
    else:
        is_A = labels.astype(bool)

    # Find runs using run-length encoding
    A_runs = []
    B_runs = []

    if len(is_A) == 0:
        return {
            'mean_A_run': np.nan, 'mean_B_run': np.nan,
            'mean_AB_period': np.nan, 'mean_AB_period_mbp': None,
            'n_transitions': 0, 'A_runs': [], 'B_runs': []
        }

    current_label = is_A[0]
    current_count = 1

    for i in range(1, len(is_A)):
        if is_A[i] == current_label:
            current_count += 1
        else:
            if current_label:
                A_runs.append(current_count)
            else:
                B_runs.append(current_count)
            current_label = is_A[i]
            current_count = 1

    # Don't forget the last run
    if current_label:
        A_runs.append(current_count)
    else:
        B_runs.append(current_count)

    mean_A = float(np.mean(A_runs)) if A_runs else np.nan
    mean_B = float(np.mean(B_runs)) if B_runs else np.nan
    mean_AB = mean_A + mean_B if (not np.isnan(mean_A) and not np.isnan(mean_B)) else np.nan

    mean_AB_mbp = None
    if mean_AB is not None and not np.isnan(mean_AB) and median_spacing_bp is not None:
        mean_AB_mbp = mean_AB * median_spacing_bp / 1e6

    n_transitions = len(A_runs) + len(B_runs) - 1

    return {
        'mean_A_run': mean_A,
        'mean_B_run': mean_B,
        'mean_AB_period': mean_AB,
        'mean_AB_period_mbp': mean_AB_mbp,
        'n_transitions': n_transitions,
        'A_runs': A_runs,
        'B_runs': B_runs,
    }


def assign_compartment_labels(peaks_df, pc1_values, pc1_threshold=0.0):
    """
    Assign A/B compartment labels to peaks based on Hi-C PC1 values.

    Parameters
    ----------
    peaks_df : pd.DataFrame
        DataFrame with peak positions (must have 'start', 'end' columns).
    pc1_values : np.ndarray or pd.Series
        PC1 eigenvector values at each peak position.
    pc1_threshold : float
        Threshold for A/B assignment. Peaks with PC1 > threshold are A.

    Returns
    -------
    np.ndarray of str
        Array of 'A' or 'B' labels for each peak.
    """
    pc1 = np.asarray(pc1_values)
    labels = np.where(pc1 > pc1_threshold, 'A', 'B')
    return labels


def bandpass_filter(signal, periods, target_period, bandwidth_log10=0.15):
    """
    Apply a band-pass filter in period space around a target period.

    Parameters
    ----------
    signal : np.ndarray
        Input signal (standardized).
    periods : np.ndarray
        Period array from CWT result.
    target_period : float
        Center period for the band-pass filter (in same units as periods).
    bandwidth_log10 : float
        Half-bandwidth in log10 period units (default 0.15, matching manuscript).

    Returns
    -------
    np.ndarray
        Band-pass filtered signal (real part of inverse CWT).

    Notes
    -----
    This is a simplified band-pass using the period range only.
    For full phase reconstruction, use phase.reconstruct_bandpass().
    """
    log_periods = np.log10(periods)
    log_target = np.log10(target_period)
    in_band = np.abs(log_periods - log_target) <= bandwidth_log10
    return in_band


def smooth_signal(signal, window=11):
    """
    Apply a uniform moving average to a signal.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    window : int
        Window size (must be odd).

    Returns
    -------
    np.ndarray
        Smoothed signal.
    """
    if window < 2:
        return signal.copy()
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode='same')


def phipson_smyth_pvalue(rank, n_permutations):
    """
    Compute Phipson-Smyth permutation p-value.

    p = (rank + 1) / (n_permutations + 1)

    This avoids p=0 for observed values exceeding all permutations.

    Parameters
    ----------
    rank : int or np.ndarray
        Number of permutations with statistic >= observed.
    n_permutations : int
        Total number of permutations.

    Returns
    -------
    float or np.ndarray
        P-value(s).
    """
    return (np.asarray(rank) + 1.0) / (n_permutations + 1.0)
