"""
chromperiod — Detecting periodic chromatin organization from accessibility data.

A Python package implementing the consecutive-peaks continuous wavelet transform
(CWT) analysis pipeline for detecting periodic chromatin accessibility patterns
from DNase-seq, ATAC-seq, or ChIP-seq narrowPeak files.

Key functions
-------------
consecutive_peaks_cwt : Run CWT on a single chromosome
run_genome_wide_cwt   : Run CWT on all chromosomes in a file

Key classes
-----------
CWTResult : Container for CWT analysis results

Quick start
-----------
>>> from chromperiod import consecutive_peaks_cwt
>>> result = consecutive_peaks_cwt('peaks.narrowPeak', chromosome='chrX')
>>> print(result)
CWTResult(chrom=chrX, n_peaks=8082, dominant_period=28.4 Mbp, sig95=39.4%, ar1=0.150)

References
----------
Torrence C, Compo GP (1998) A practical guide to wavelet analysis.
    Bull Am Meteorol Soc 79(1):61-78.

Gebhardt WH (2026) Consecutive chromatin accessibility peaks reveal periodic
    compartmental organization of eukaryotic chromosomes. bioRxiv.
"""

__version__ = '0.1.0'
__author__ = 'Wolf H. Gebhardt'
__email__ = 'w.gebhardt@protonmail.com'
__license__ = 'MIT'

from .cwt import consecutive_peaks_cwt, run_genome_wide_cwt, CWTResult
from .io import load_peaks
from .utils import runlength_analysis, estimate_ar1, compute_coi
from .significance import (
    ar1_significance_threshold,
    permutation_test,
    white_noise_false_positive_rate,
)
from .phase import (
    reconstruct_bandpass,
    reconstruct_bandpass_full,
    classify_phase,
    harmonize_phase,
    phase_compartment_concordance,
)

__all__ = [
    # Core CWT
    'consecutive_peaks_cwt',
    'run_genome_wide_cwt',
    'CWTResult',
    # I/O
    'load_peaks',
    # Utilities
    'runlength_analysis',
    'estimate_ar1',
    'compute_coi',
    # Significance
    'ar1_significance_threshold',
    'permutation_test',
    'white_noise_false_positive_rate',
    # Phase
    'reconstruct_bandpass',
    'reconstruct_bandpass_full',
    'classify_phase',
    'harmonize_phase',
    'phase_compartment_concordance',
]
