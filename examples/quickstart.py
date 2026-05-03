"""
quickstart.py — Minimal example: load peaks -> run CWT -> plot

This script demonstrates the core chromperiod workflow using a synthetic
sine wave signal. To use with real data, replace the synthetic signal
with a call to load_peaks() on a narrowPeak file.

Usage:
    python quickstart.py

To use with real ENCODE data (see examples/example_data/README.md):
    python quickstart.py --file ENCFF250GOB.narrowPeak --chrom chrX
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import sys
import os

# Add parent directory to path for development installs
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chromperiod import consecutive_peaks_cwt
from chromperiod.plotting import plot_scalogram, plot_gws, plot_triple_comparison


def make_synthetic_peaks(n_peaks=2000, period=400, noise_level=1.0, seed=42):
    """
    Create a synthetic narrowPeak-like DataFrame with a known periodic signal.

    Parameters
    ----------
    n_peaks : int
        Number of peaks.
    period : int
        True period in peak-index units.
    noise_level : float
        Standard deviation of additive Gaussian noise.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: chrom, start, end, signalValue
    """
    rng = np.random.default_rng(seed)

    # Sine wave signal with noise
    x = np.arange(n_peaks)
    signal = np.sin(2 * np.pi * x / period) + noise_level * rng.standard_normal(n_peaks)

    # Simulate genomic positions (median spacing ~10 kb)
    spacings = rng.exponential(scale=10000, size=n_peaks)
    positions = np.cumsum(spacings).astype(int)

    df = pd.DataFrame({
        'chrom': ['chrSim'] * n_peaks,
        'start': positions,
        'end': positions + 200,
        'name': ['.'] * n_peaks,
        'score': [0] * n_peaks,
        'strand': ['.'] * n_peaks,
        'signalValue': signal,
        'pValue': [-1.0] * n_peaks,
        'qValue': [-1.0] * n_peaks,
        'peak': [-1] * n_peaks,
    })
    return df


def main():
    parser = argparse.ArgumentParser(
        description='chromperiod quickstart example'
    )
    parser.add_argument('--file', type=str, default=None,
                        help='Path to narrowPeak file (default: use synthetic data)')
    parser.add_argument('--chrom', type=str, default='chrSim',
                        help='Chromosome to analyze (default: chrSim)')
    parser.add_argument('--output', type=str, default='quickstart_output',
                        help='Output directory for figures')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Load or generate peaks
    # -------------------------------------------------------------------------
    if args.file is not None:
        print(f"Loading peaks from {args.file}...")
        from chromperiod import load_peaks
        peaks_df = load_peaks(args.file, chromosome=args.chrom)
        print(f"  Loaded {len(peaks_df)} peaks on {args.chrom}")
    else:
        print("No file provided — using synthetic sine wave signal (period=400 peaks)")
        peaks_df = make_synthetic_peaks(n_peaks=2000, period=400, noise_level=1.0)
        args.chrom = 'chrSim'

    # -------------------------------------------------------------------------
    # Step 2: Run CWT
    # -------------------------------------------------------------------------
    print(f"\nRunning CWT on {args.chrom}...")
    result = consecutive_peaks_cwt(
        peaks_df,
        chromosome=args.chrom,
        wavelet='paul',
        order=2,
        n_scales=80,
        period_min=10,
        period_max=min(7000, len(peaks_df) // 2),
        significance_level=0.95,
    )
    print(f"  {result}")
    print(f"  Dominant period: {result.dominant_period:.1f} peak-index units")
    if result.dominant_period_mbp is not None:
        print(f"  Dominant period: {result.dominant_period_mbp:.2f} Mbp")
    print(f"  sig95: {result.sig95:.1%}")
    print(f"  AR1 alpha: {result.ar1_alpha:.3f}")
    print(f"  COI-accessible fraction: {result.coi_frac:.3f}")
    print(f"  Near COI edge: {result.dom_period_near_coi_edge}")

    # -------------------------------------------------------------------------
    # Step 3: Plot scalogram
    # -------------------------------------------------------------------------
    print("\nGenerating scalogram...")
    fig, ax = plot_scalogram(result, period_units='peaks')
    scalogram_path = os.path.join(args.output, 'scalogram.png')
    fig.savefig(scalogram_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {scalogram_path}")

    # -------------------------------------------------------------------------
    # Step 4: Plot GWS
    # -------------------------------------------------------------------------
    print("Generating GWS plot...")
    fig2, ax2 = plot_gws(result, period_units='peaks')
    gws_path = os.path.join(args.output, 'gws.png')
    fig2.savefig(gws_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Saved: {gws_path}")

    # -------------------------------------------------------------------------
    # Step 5: Run randomized control and compare
    # -------------------------------------------------------------------------
    print("\nRunning randomized control...")
    rng = np.random.default_rng(42)
    rand_df = peaks_df.copy()
    rand_df['signalValue'] = rng.permutation(peaks_df['signalValue'].values)

    result_rand = consecutive_peaks_cwt(
        rand_df,
        chromosome=args.chrom,
        wavelet='paul',
        order=2,
        n_scales=80,
        period_min=10,
        period_max=min(7000, len(peaks_df) // 2),
    )
    print(f"  Randomized: sig95={result_rand.sig95:.1%} (expected ~5%)")

    print("\nDone! Output files:")
    for f in os.listdir(args.output):
        fpath = os.path.join(args.output, f)
        print(f"  {fpath} ({os.path.getsize(fpath):,} bytes)")


if __name__ == '__main__':
    main()
