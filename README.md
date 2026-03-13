# chromperiod

**Detecting periodic chromatin organization from accessibility data**

chromperiod applies continuous wavelet transforms (CWT) to consecutive chromatin accessibility peaks — ordered by chromosomal position but stripped of inter-peak distance information — to reveal periodic compartmental organization across eukaryotic genomes.

This coordinate-independent representation performs a nonlinear compression of the genome that exposes multi-megabase periodic structure invisible to conventional coordinate-based analysis. The detected periodicity corresponds to Hi-C A/B compartments, is conserved from *Drosophila* to human, persists upon CTCF depletion, and is dynamically remodeled during stem cell differentiation.

## Key Findings

- **Genome-wide periodicity** across all human (23), mouse (20), and *Drosophila* (5) chromosomes
- **Hi-C compartment concordance**: Fisher's OR = 10.88 on human chrX (21/23 chromosomes genome-wide)
- **Cross-species conservation**: comparable periodicity in organisms separated by ~800 million years of evolution
- **CTCF-independent**: periodicity persists in *Drosophila* (no CTCF-dependent loops) and after CTCF depletion in human cells
- **Developmentally dynamic**: 2.7-fold period increase during H7 hESC cardiac differentiation

## Installation

```bash
# From source
git clone https://github.com/wolfgebhardt/chromperiod.git
cd chromperiod
pip install -e .

# Or directly
pip install git+https://github.com/wolfgebhardt/chromperiod.git
```

**Requirements:** Python ≥ 3.8, numpy ≥ 1.21, scipy ≥ 1.7, matplotlib ≥ 3.5, pandas ≥ 1.3

No external wavelet libraries are required — the CWT is implemented from scratch following Torrence & Compo (1998).

## Quickstart

```python
from chromperiod import consecutive_peaks_cwt
from chromperiod.plotting import plot_scalogram, plot_gws

# Run CWT on a narrowPeak file
result = consecutive_peaks_cwt("peaks.narrowPeak", chromosome="chrX")

# Print key metrics
print(f"Dominant period: {result.dominant_period:.0f} peaks ({result.dominant_period_mbp:.1f} Mbp)")
print(f"sig95: {result.sig95:.1%}")
print(f"AR1 alpha: {result.ar1_alpha:.3f}")
print(f"COI-accessible fraction: {result.coi_frac:.3f}")

# Generate publication-quality figures
plot_scalogram(result, output="scalogram.png")
plot_gws(result, output="gws.png")
```

## What It Does

1. **Parses** chromatin accessibility peaks from narrowPeak (DNase-seq, ATAC-seq, ChIP-seq) or Hotspot3 BED files
2. **Orders** peaks consecutively by genomic position, using signal values as the analysis signal
3. **Applies** the continuous wavelet transform (Paul wavelet m=4 by default; Morlet and DOG also supported)
4. **Tests** significance against an AR1 red-noise null model (χ² test, 95% confidence) and optionally by Monte Carlo permutation
5. **Classifies** peaks by wavelet phase (high-phase vs. low-phase) at the dominant period band
6. **Computes** compartment concordance when Hi-C eigenvector data is provided
7. **Measures** A/B run lengths as a wavelet-independent periodicity estimate

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `wavelet` | `'paul'` | Wavelet family: `'paul'`, `'morlet'`, or `'dog'` |
| `order` | `4` | Wavelet order (m=4 for Paul, ω₀=6 for Morlet, m=2 for DOG) |
| `n_scales` | `80` | Number of log-spaced scales |
| `period_min` | `10` | Minimum period in peak-index units |
| `period_max` | `7000` | Maximum period in peak-index units |
| `significance_level` | `0.95` | AR1 red-noise significance level |
| `signal_column` | `'signalValue'` | Column name or index for the analysis signal |

## Method

The analysis follows Torrence & Compo (1998) exactly:

- **CWT computation**: FFT convolution (T&C eq. 4)
- **Power**: |W_n(s)|² / σ² — no division by scale
- **Significance**: χ²(2) test against AR1 red-noise background (T&C eq. 18)
- **Cone of influence**: √2 · λ · δt · min(n, N−n) (T&C eq. 25)
- **Dominant period**: argmax of global wavelet spectrum (mean power outside COI)
- **Phase reconstruction**: band-pass filtered real part of CWT coefficients (T&C eq. 11)

The key innovation is the **consecutive-peak representation**: peaks are analyzed in their sequential genomic order with inter-peak distances discarded. This nonlinear compression collapses inactive chromatin regions and amplifies periodic fluctuations in chromatin state that are masked in coordinate space.

## Additional Tools

```python
from chromperiod.significance import permutation_test
from chromperiod.phase import classify_phases, run_length_analysis
from chromperiod.plotting import plot_triple_comparison

# Monte Carlo permutation testing (1000 surrogates, BH FDR correction)
perm_result = permutation_test(result, n_surrogates=1000, seed=42)

# Phase classification at dominant period band
phases = classify_phases(result, threshold_sd=0.5)

# Wavelet-independent run-length analysis (requires PC1 values)
runs = run_length_analysis(peaks_df, pc1_column='PC1')

# Triple comparison: consecutive vs. randomized vs. linear interpolation
plot_triple_comparison("peaks.narrowPeak", chromosome="chrX", output="triple.png")
```

## Example Data

The manuscript analyses use publicly available ENCODE data. To reproduce the primary result (MCF-7 chrX):

```bash
# Download MCF-7 DNase-seq narrowPeak from ENCODE
wget https://www.encodeproject.org/files/ENCFF250GOB/@@download/ENCFF250GOB.bed.gz
gunzip ENCFF250GOB.bed.gz
```

```python
result = consecutive_peaks_cwt("ENCFF250GOB.bed", chromosome="chrX")
# Expected: dominant period ~28 Mbp, sig95 ~39%, AR1 ~0.15
```

## Tests

```bash
python -m pytest tests/ -v
```

31 tests covering: white noise false positive rates, sine wave period recovery, randomization controls, wavelet family consistency, AR1 estimation accuracy, BH FDR correction, and Phipson-Smyth p-value validity.

## Citation

```
Gebhardt WH (2026) Consecutive chromatin accessibility peaks reveal periodic
compartmental organization of eukaryotic chromosomes. bioRxiv [DOI to be added]
```

## License

MIT License. Copyright (c) 2026 Wolf H. Gebhardt.

## References

- Torrence C, Compo GP (1998) A practical guide to wavelet analysis. *Bull Am Meteorol Soc* 79:61–78.
- Owen JA, Osmanović D, Mirny LA (2023) Design principles of 3D epigenetic memory systems. *Science* 382:eadg3053.
- Lieberman-Aiden E et al. (2009) Comprehensive mapping of long-range interactions reveals folding principles of the human genome. *Science* 326:289–293.
