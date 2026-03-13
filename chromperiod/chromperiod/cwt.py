"""
cwt.py — Core continuous wavelet transform analysis for chromperiod.

Implements the consecutive-peaks CWT following Torrence & Compo (1998),
Bulletin of the American Meteorological Society, 79(1), 61-78.

All equations referenced as T&C eq. N.

Key normalization choices (matching the manuscript analyses):
  - Power = |W_n(s)|^2 / sigma^2  (NO division by scale)
  - AR1 significance via chi^2(2) distribution (T&C eq. 18)
  - GWS-peak dominant period = argmax of mean power outside COI
  - Signal standardized to zero mean and unit variance before CWT
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Union
import pandas as pd

from .io import load_peaks
from .utils import compute_coi, compute_coi_frac, estimate_ar1


# ---------------------------------------------------------------------------
# Wavelet definitions
# ---------------------------------------------------------------------------

def _paul_psi_hat(s, omega, m=4):
    """
    Fourier transform of the Paul wavelet of order m (T&C eq. 6).
    psi_hat(s*omega) = (2^m / sqrt(m * (2m-1)!)) * H(omega) * (s*omega)^m * exp(-s*omega)
    where H(omega) is the Heaviside step function.
    """
    import math
    norm = (2.0**m) / np.sqrt(m * float(math.factorial(2 * m - 1)))
    # Clip exponent argument to avoid overflow; exp(-x) -> 0 for large x
    exponent = np.clip(-s * omega, -500.0, 0.0)
    result = np.where(omega > 0,
                      norm * (s * omega)**m * np.exp(exponent),
                      0.0)
    return result


def _morlet_psi_hat(s, omega, omega0=6.0):
    """
    Fourier transform of the Morlet wavelet (T&C eq. 4).
    psi_hat(s*omega) = pi^(-1/4) * H(omega) * exp(-0.5*(s*omega - omega0)^2)
    """
    result = np.where(omega > 0,
                      np.pi**(-0.25) * np.exp(-0.5 * (s * omega - omega0)**2),
                      0.0)
    return result


def _dog_psi_hat(s, omega, m=2):
    """
    Fourier transform of the DOG (Derivative of Gaussian) wavelet of order m (T&C eq. 14).
    psi_hat(s*omega) = -(-i)^m / sqrt(Gamma(m+0.5)) * (s*omega)^m * exp(-0.5*(s*omega)^2)
    """
    from scipy.special import gamma
    norm = (-1.0)**(m + 1) / np.sqrt(gamma(m + 0.5))
    result = norm * (s * omega)**m * np.exp(-0.5 * (s * omega)**2)
    return result


def _get_wavelet_params(wavelet, order):
    """
    Return (psi_hat_func, fourier_factor, coi_factor) for the given wavelet.

    fourier_factor: converts scale s to equivalent Fourier period (T&C Table 1)
    coi_factor: lambda_psi in T&C eq. 25 (COI e-folding factor)
    """
    wavelet = wavelet.lower()
    if wavelet == 'paul':
        m = order
        fourier_factor = 4.0 * np.pi / (2.0 * m + 1.0)
        coi_factor = fourier_factor / np.sqrt(2.0)
        psi_hat = lambda s, omega: _paul_psi_hat(s, omega, m=m)
    elif wavelet == 'morlet':
        omega0 = float(order)  # order used as omega0
        fourier_factor = 4.0 * np.pi / (omega0 + np.sqrt(2.0 + omega0**2))
        coi_factor = fourier_factor * np.sqrt(2.0)
        psi_hat = lambda s, omega: _morlet_psi_hat(s, omega, omega0=omega0)
    elif wavelet == 'dog':
        m = order
        from scipy.special import gamma
        fourier_factor = 2.0 * np.pi / np.sqrt(m + 0.5)
        coi_factor = fourier_factor * np.sqrt(2.0)
        psi_hat = lambda s, omega: _dog_psi_hat(s, omega, m=m)
    else:
        raise ValueError(f"Unknown wavelet '{wavelet}'. Choose 'paul', 'morlet', or 'dog'.")
    return psi_hat, fourier_factor, coi_factor


# ---------------------------------------------------------------------------
# CWT result container
# ---------------------------------------------------------------------------

@dataclass
class CWTResult:
    """
    Container for CWT analysis results.

    Attributes
    ----------
    power : np.ndarray, shape (n_scales, n_peaks)
        Normalized wavelet power |W_n(s)|^2 / sigma^2.
    significance : np.ndarray of bool, shape (n_scales, n_peaks)
        True where power exceeds the AR1 95% significance threshold.
    gws : np.ndarray, shape (n_scales,)
        Global wavelet spectrum (time-averaged power outside COI).
    periods : np.ndarray, shape (n_scales,)
        Period array in peak-index units.
    periods_mbp : np.ndarray or None, shape (n_scales,)
        Period array in Mbp (if median_spacing_bp was provided).
    dominant_period : float
        GWS-peak dominant period in peak-index units.
    dominant_period_mbp : float or None
        GWS-peak dominant period in Mbp (if median_spacing_bp was provided).
    sig95 : float
        Fraction of time-scale points outside COI that exceed AR1 95% threshold.
    ar1_alpha : float
        Lag-1 autocorrelation coefficient of the input signal.
    coi : np.ndarray, shape (n_peaks,)
        Cone of influence in period units at each peak position.
    coi_frac : float
        Fraction of scalogram area outside the COI.
    scales : np.ndarray, shape (n_scales,)
        Scale array (in peak-index units).
    signal : np.ndarray, shape (n_peaks,)
        Standardized input signal.
    ar1_threshold : np.ndarray, shape (n_scales,)
        AR1 significance threshold at each scale.
    wavelet : str
        Wavelet type used.
    order : int or float
        Wavelet order used.
    chromosome : str or None
        Chromosome analyzed.
    n_peaks : int
        Number of peaks analyzed.
    median_spacing_bp : float or None
        Median inter-peak spacing in bp.
    dom_period_near_coi_edge : bool
        True if dominant period is within the outermost 10% of COI-accessible range.
    """
    power: np.ndarray
    significance: np.ndarray
    gws: np.ndarray
    periods: np.ndarray
    periods_mbp: Optional[np.ndarray]
    dominant_period: float
    dominant_period_mbp: Optional[float]
    sig95: float
    ar1_alpha: float
    coi: np.ndarray
    coi_frac: float
    scales: np.ndarray
    signal: np.ndarray
    ar1_threshold: np.ndarray
    wavelet: str
    order: Union[int, float]
    chromosome: Optional[str]
    n_peaks: int
    median_spacing_bp: Optional[float]
    dom_period_near_coi_edge: bool = False

    def __repr__(self):
        period_str = (f"{self.dominant_period_mbp:.1f} Mbp"
                      if self.dominant_period_mbp is not None
                      else f"{self.dominant_period:.1f} peaks")
        return (f"CWTResult(chrom={self.chromosome}, n_peaks={self.n_peaks}, "
                f"dominant_period={period_str}, sig95={self.sig95:.1%}, "
                f"ar1={self.ar1_alpha:.3f})")


# ---------------------------------------------------------------------------
# Main CWT function
# ---------------------------------------------------------------------------

def consecutive_peaks_cwt(
    peaks_file,
    chromosome=None,
    wavelet='paul',
    order=4,
    n_scales=80,
    period_min=10,
    period_max=7000,
    significance_level=0.95,
    signal_column='signalValue',
):
    """
    Run the consecutive-peaks CWT analysis on a narrowPeak or BED file.

    Parameters
    ----------
    peaks_file : str or pd.DataFrame
        Path to narrowPeak/BED file, or a pre-loaded DataFrame with columns
        ['chrom', 'start', 'end', signal_column].
    chromosome : str or None
        Chromosome to analyze (e.g. 'chrX'). If None, analyzes all peaks
        concatenated (use with caution for multi-chromosome files).
    wavelet : str
        Wavelet type: 'paul', 'morlet', or 'dog'.
    order : int or float
        Wavelet order. For Paul: m=4 (default). For Morlet: omega0=6.
        For DOG: m=2.
    n_scales : int
        Number of log-spaced scales (default 80).
    period_min : float
        Minimum period in peak-index units (default 10).
    period_max : float
        Maximum period in peak-index units (default 7000).
    significance_level : float
        AR1 red-noise significance level (default 0.95).
    signal_column : str or int
        Column name or index for the analysis signal (default 'signalValue').

    Returns
    -------
    CWTResult
        Object with all analysis results. See CWTResult docstring for attributes.

    Notes
    -----
    Implements T&C 1998 equations:
      - CWT via FFT convolution (eq. 3, 4)
      - Power normalization: |W_n(s)|^2 / sigma^2 (no scale division)
      - AR1 significance: chi^2(2) distribution (eq. 18)
      - COI: T&C eq. 25
      - GWS: time-average of power outside COI (eq. 7)
    """
    # --- Load data ---
    if isinstance(peaks_file, pd.DataFrame):
        df = peaks_file.copy()
    else:
        df = load_peaks(peaks_file, signal_column=signal_column)

    if chromosome is not None:
        df = df[df['chrom'] == chromosome].copy()
        if len(df) == 0:
            raise ValueError(f"No peaks found for chromosome '{chromosome}'.")

    df = df.sort_values(['chrom', 'start']).reset_index(drop=True)

    # Extract signal
    if isinstance(signal_column, int):
        signal_raw = df.iloc[:, signal_column].values.astype(float)
    else:
        signal_raw = df[signal_column].values.astype(float)

    N = len(signal_raw)
    if N < 2 * period_min:
        raise ValueError(f"Too few peaks ({N}) for period_min={period_min}. "
                         f"Need at least {2 * period_min} peaks.")

    # Median inter-peak spacing
    midpoints = ((df['start'].values + df['end'].values) / 2.0)
    if len(midpoints) > 1:
        spacings = np.diff(midpoints)
        spacings = spacings[spacings > 0]
        median_spacing_bp = float(np.median(spacings)) if len(spacings) > 0 else None
    else:
        median_spacing_bp = None

    # --- Standardize signal ---
    sigma = np.std(signal_raw, ddof=1)
    if sigma == 0:
        raise ValueError("Signal has zero variance — cannot run CWT.")
    signal = (signal_raw - np.mean(signal_raw)) / sigma

    # --- Estimate AR1 ---
    ar1_alpha = estimate_ar1(signal)

    # --- Build scale array ---
    # Scales are in peak-index units (delta_t = 1)
    delta_t = 1.0
    s0 = period_min / _get_wavelet_params(wavelet, order)[1]  # convert period to scale
    psi_hat_func, fourier_factor, coi_factor = _get_wavelet_params(wavelet, order)

    # s0 such that period_min = s0 * fourier_factor
    s0 = period_min / fourier_factor
    # smax such that period_max = smax * fourier_factor
    smax = period_max / fourier_factor
    # Log-spaced scales
    scales = np.geomspace(s0, smax, n_scales)
    periods = scales * fourier_factor  # in peak-index units

    # --- FFT of signal ---
    # Zero-pad to next power of 2 for efficiency
    N_pad = int(2 ** np.ceil(np.log2(N)))
    x_hat = np.fft.fft(signal, n=N_pad)

    # Angular frequencies (T&C eq. 5)
    omega_pos = 2.0 * np.pi * np.arange(1, N_pad // 2 + 1) / (N_pad * delta_t)
    omega_neg = -2.0 * np.pi * np.arange(N_pad // 2 - 1, 0, -1) / (N_pad * delta_t)
    omega = np.concatenate([[0.0], omega_pos, omega_neg])

    # --- Compute CWT via FFT convolution (T&C eq. 4) ---
    # W_n(s) = sum_k x_hat(k) * psi_hat*(s*omega_k) * exp(i*omega_k*n*delta_t)
    power = np.zeros((n_scales, N), dtype=float)

    for j, s in enumerate(scales):
        # Normalized wavelet transform in Fourier space
        psi_vals = np.conj(psi_hat_func(s, omega))
        # Normalization factor: sqrt(2*pi*s/delta_t) (T&C eq. 4)
        norm = np.sqrt(2.0 * np.pi * s / delta_t)
        W_hat = norm * x_hat * psi_vals
        # Inverse FFT to get W_n(s) for all n
        W_n = np.fft.ifft(W_hat)[:N]
        # Power = |W_n(s)|^2 / sigma^2
        # Since signal is already standardized (sigma=1), sigma^2=1
        power[j, :] = np.abs(W_n)**2

    # --- COI ---
    coi = compute_coi(N, delta_t, coi_factor)  # in period units

    # --- AR1 significance threshold (T&C eq. 18) ---
    # P_k = (1 - alpha^2) / (1 - 2*alpha*cos(2*pi*k/N) + alpha^2)
    # Threshold = P_k * chi2_95 / 2  (chi^2 with 2 dof)
    from scipy.stats import chi2
    chi2_crit = chi2.ppf(significance_level, df=2)

    # Background spectrum at each scale (T&C eq. 16)
    # For each scale s, the dominant frequency is 1/period = 1/(s*fourier_factor)
    # Use the AR1 background at that frequency
    ar1_threshold = np.zeros(n_scales)
    for j, s in enumerate(scales):
        period_j = periods[j]
        # Frequency in cycles per peak-index unit
        freq_j = 1.0 / period_j
        # AR1 background power at this frequency (T&C eq. 16)
        P_k = (1.0 - ar1_alpha**2) / (1.0 - 2.0 * ar1_alpha * np.cos(2.0 * np.pi * freq_j) + ar1_alpha**2)
        ar1_threshold[j] = P_k * chi2_crit / 2.0

    # Significance mask
    significance = power > ar1_threshold[:, np.newaxis]

    # --- GWS (T&C eq. 7): time-average of power outside COI ---
    # For each scale j, average over peaks n where period[j] < coi[n]
    gws = np.zeros(n_scales)
    outside_coi_mask = np.zeros((n_scales, N), dtype=bool)
    for j in range(n_scales):
        mask = periods[j] < coi  # outside COI
        outside_coi_mask[j, :] = mask
        if mask.sum() > 0:
            gws[j] = np.mean(power[j, mask])
        else:
            gws[j] = np.nan

    # --- COI-accessible fraction ---
    coi_frac = compute_coi_frac(N, n_scales, coi, periods)

    # --- sig95: fraction of outside-COI points that are significant ---
    n_outside = outside_coi_mask.sum()
    n_sig_outside = (significance & outside_coi_mask).sum()
    sig95 = float(n_sig_outside) / float(n_outside) if n_outside > 0 else 0.0

    # --- Dominant period: argmax of GWS (outside COI) ---
    valid_gws = np.where(np.isfinite(gws), gws, -np.inf)
    dom_idx = int(np.argmax(valid_gws))
    dominant_period = float(periods[dom_idx])

    # --- Convert to Mbp if spacing available ---
    if median_spacing_bp is not None:
        periods_mbp = periods * median_spacing_bp / 1e6
        dominant_period_mbp = dominant_period * median_spacing_bp / 1e6
    else:
        periods_mbp = None
        dominant_period_mbp = None

    # --- Near-COI-edge flag ---
    # Find max accessible period (where at least 1 peak is outside COI)
    accessible_periods = [periods[j] for j in range(n_scales)
                          if outside_coi_mask[j, :].sum() > 0]
    if accessible_periods:
        max_accessible = max(accessible_periods)
        dom_period_near_coi_edge = dominant_period > (max_accessible / 1.11)
    else:
        dom_period_near_coi_edge = True

    return CWTResult(
        power=power,
        significance=significance,
        gws=gws,
        periods=periods,
        periods_mbp=periods_mbp,
        dominant_period=dominant_period,
        dominant_period_mbp=dominant_period_mbp,
        sig95=sig95,
        ar1_alpha=ar1_alpha,
        coi=coi,
        coi_frac=coi_frac,
        scales=scales,
        signal=signal,
        ar1_threshold=ar1_threshold,
        wavelet=wavelet,
        order=order,
        chromosome=chromosome,
        n_peaks=N,
        median_spacing_bp=median_spacing_bp,
        dom_period_near_coi_edge=dom_period_near_coi_edge,
    )


def run_genome_wide_cwt(
    peaks_file,
    chromosomes=None,
    wavelet='paul',
    order=4,
    n_scales=80,
    period_min=10,
    period_max=7000,
    significance_level=0.95,
    signal_column='signalValue',
    min_peaks=100,
):
    """
    Run CWT on all chromosomes in a peaks file.

    Parameters
    ----------
    peaks_file : str or pd.DataFrame
        Path to narrowPeak/BED file or pre-loaded DataFrame.
    chromosomes : list of str or None
        Chromosomes to analyze. If None, analyzes all chromosomes present.
    min_peaks : int
        Minimum number of peaks required to analyze a chromosome (default 100).
    (other parameters as in consecutive_peaks_cwt)

    Returns
    -------
    dict
        Dictionary mapping chromosome name to CWTResult.
    """
    if isinstance(peaks_file, pd.DataFrame):
        df = peaks_file.copy()
    else:
        df = load_peaks(peaks_file, signal_column=signal_column)

    if chromosomes is None:
        chromosomes = sorted(df['chrom'].unique())

    results = {}
    for chrom in chromosomes:
        chrom_df = df[df['chrom'] == chrom]
        if len(chrom_df) < min_peaks:
            print(f"  Skipping {chrom}: only {len(chrom_df)} peaks (min={min_peaks})")
            continue
        try:
            result = consecutive_peaks_cwt(
                chrom_df,
                chromosome=chrom,
                wavelet=wavelet,
                order=order,
                n_scales=n_scales,
                period_min=period_min,
                period_max=period_max,
                significance_level=significance_level,
                signal_column=signal_column,
            )
            results[chrom] = result
        except Exception as e:
            print(f"  Error on {chrom}: {e}")

    return results
