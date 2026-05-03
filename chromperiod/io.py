"""
io.py — File I/O for chromperiod.

Parsers for:
  - narrowPeak format (ENCODE standard, signalValue in column 7)
  - BED format (generic, configurable signal column)
  - Hotspot3 BED format (max_density in column 5)
"""

import numpy as np
import pandas as pd
from typing import Union, Optional


# narrowPeak column names (0-indexed)
NARROWPEAK_COLS = [
    'chrom', 'start', 'end', 'name', 'score',
    'strand', 'signalValue', 'pValue', 'qValue', 'peak'
]

# BED3 minimum columns
BED3_COLS = ['chrom', 'start', 'end']


def load_peaks(
    filepath: str,
    signal_column: Union[str, int] = 'signalValue',
    file_format: Optional[str] = None,
    chromosome: Optional[str] = None,
    min_signal: Optional[float] = None,
) -> pd.DataFrame:
    """
    Load a narrowPeak or BED file into a DataFrame.

    Automatically detects file format based on extension or column count.
    Returns a DataFrame with at minimum columns: ['chrom', 'start', 'end',
    signal_column].

    Parameters
    ----------
    filepath : str
        Path to the input file. Supports .narrowPeak, .bed, .bed.gz,
        .narrowPeak.gz, and plain text files.
    signal_column : str or int
        Column name or 0-based index for the signal to analyze.
        - 'signalValue' (default): column 7 in narrowPeak format
        - 'score': column 5 (BED score)
        - int: 0-based column index
        - 'max_density': column 5 in Hotspot3 BED format
    file_format : str or None
        Force file format: 'narrowpeak', 'bed', or 'hotspot3'.
        If None, auto-detected from extension and column count.
    chromosome : str or None
        If provided, return only peaks on this chromosome.
    min_signal : float or None
        If provided, filter out peaks with signal < min_signal.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns including 'chrom', 'start', 'end', and
        the signal column (named by signal_column if str, or 'signal' if int).

    Raises
    ------
    ValueError
        If the file cannot be parsed or the signal column is not found.
    FileNotFoundError
        If the file does not exist.
    """
    import os
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # Detect format
    fmt = _detect_format(filepath, file_format)

    if fmt == 'narrowpeak':
        df = _load_narrowpeak(filepath)
    elif fmt == 'hotspot3':
        df = _load_hotspot3(filepath)
    else:
        df = _load_bed(filepath, signal_column)

    # Resolve signal column
    if isinstance(signal_column, int):
        # Already loaded; rename the column
        col_name = df.columns[signal_column]
        if col_name not in ('chrom', 'start', 'end'):
            df = df.rename(columns={col_name: 'signal'})
            signal_col_name = 'signal'
        else:
            signal_col_name = col_name
    else:
        signal_col_name = signal_column
        if signal_col_name not in df.columns:
            # Try to find it by position
            available = [c for c in df.columns if c not in ('chrom', 'start', 'end')]
            if available:
                df = df.rename(columns={available[0]: signal_col_name})
            else:
                raise ValueError(
                    f"Signal column '{signal_col_name}' not found. "
                    f"Available columns: {list(df.columns)}"
                )

    # Ensure required columns exist
    for col in ['chrom', 'start', 'end']:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in file.")

    # Convert types
    df['start'] = pd.to_numeric(df['start'], errors='coerce').astype(int)
    df['end'] = pd.to_numeric(df['end'], errors='coerce').astype(int)
    df[signal_col_name] = pd.to_numeric(df[signal_col_name], errors='coerce')

    # Drop rows with NaN signal
    df = df.dropna(subset=[signal_col_name])

    # Filter by chromosome
    if chromosome is not None:
        df = df[df['chrom'] == chromosome].copy()

    # Filter by minimum signal
    if min_signal is not None:
        df = df[df[signal_col_name] >= min_signal].copy()

    # Sort by position
    df = df.sort_values(['chrom', 'start']).reset_index(drop=True)

    return df


def _detect_format(filepath, file_format=None):
    """Auto-detect file format."""
    if file_format is not None:
        return file_format.lower()

    fp_lower = filepath.lower()
    if fp_lower.endswith('.narrowpeak') or fp_lower.endswith('.narrowpeak.gz'):
        return 'narrowpeak'
    elif 'hotspot' in fp_lower:
        return 'hotspot3'
    else:
        # Try to detect by column count
        try:
            sample = pd.read_csv(filepath, sep='\t', header=None, nrows=5,
                                  comment='#')
            if sample.shape[1] >= 10:
                return 'narrowpeak'
            else:
                return 'bed'
        except Exception:
            return 'bed'


def _load_narrowpeak(filepath):
    """Load a narrowPeak file."""
    try:
        df = pd.read_csv(
            filepath, sep='\t', header=None,
            names=NARROWPEAK_COLS[:10],
            usecols=range(min(10, 10)),
            comment='#',
        )
    except Exception:
        # Try with fewer columns
        df = pd.read_csv(filepath, sep='\t', header=None, comment='#')
        n_cols = min(len(df.columns), len(NARROWPEAK_COLS))
        df.columns = NARROWPEAK_COLS[:n_cols] + [f'col{i}' for i in range(n_cols, len(df.columns))]

    return df


def _load_hotspot3(filepath):
    """
    Load a Hotspot3 BED file with max_density in column 5 (0-indexed: col 4).

    Hotspot3 format: chrom start end name max_density [...]
    """
    df = pd.read_csv(filepath, sep='\t', header=None, comment='#')
    col_names = ['chrom', 'start', 'end', 'name', 'max_density']
    n = min(len(df.columns), len(col_names))
    df.columns = col_names[:n] + [f'col{i}' for i in range(n, len(df.columns))]

    if 'max_density' not in df.columns and len(df.columns) >= 5:
        df = df.rename(columns={df.columns[4]: 'max_density'})

    # Add signalValue alias for compatibility
    if 'max_density' in df.columns:
        df['signalValue'] = df['max_density']

    return df


def _load_bed(filepath, signal_column):
    """Load a generic BED file."""
    df = pd.read_csv(filepath, sep='\t', header=None, comment='#')

    # Assign column names
    base_cols = ['chrom', 'start', 'end']
    extra_cols = [f'col{i}' for i in range(3, len(df.columns))]
    df.columns = base_cols + extra_cols

    # Map common signal column names to positions
    signal_col_map = {
        'score': 'col3',
        'signalValue': 'col6',
        'max_density': 'col4',
        'name': 'col3',
    }

    if isinstance(signal_column, str) and signal_column in signal_col_map:
        target = signal_col_map[signal_column]
        if target in df.columns:
            df = df.rename(columns={target: signal_column})

    return df


def load_bigwig_at_peaks(bigwig_file, peaks_df, summary='mean'):
    """
    Extract BigWig values at peak positions.

    Parameters
    ----------
    bigwig_file : str
        Path to BigWig file.
    peaks_df : pd.DataFrame
        DataFrame with 'chrom', 'start', 'end' columns.
    summary : str
        Summary statistic: 'mean', 'max', or 'min'.

    Returns
    -------
    np.ndarray
        Array of BigWig values at each peak.
    """
    try:
        import pyBigWig
    except ImportError:
        raise ImportError("pyBigWig is required for BigWig loading. "
                          "Install with: pip install pyBigWig")

    bw = pyBigWig.open(bigwig_file)
    values = []
    for _, row in peaks_df.iterrows():
        try:
            if summary == 'mean':
                val = bw.stats(row['chrom'], int(row['start']), int(row['end']),
                               type='mean')[0]
            elif summary == 'max':
                val = bw.stats(row['chrom'], int(row['start']), int(row['end']),
                               type='max')[0]
            else:
                val = bw.stats(row['chrom'], int(row['start']), int(row['end']),
                               type='min')[0]
            values.append(val if val is not None else np.nan)
        except Exception:
            values.append(np.nan)
    bw.close()
    return np.array(values)


def write_phase_bed(peaks_df, phase_labels, output_file, phase='high'):
    """
    Write phase-classified peaks to a BED file.

    Parameters
    ----------
    peaks_df : pd.DataFrame
        DataFrame with 'chrom', 'start', 'end' columns.
    phase_labels : np.ndarray of str
        Phase labels ('high', 'low', 'background').
    output_file : str
        Output BED file path.
    phase : str
        Which phase to write ('high', 'low', or 'background').
    """
    mask = np.asarray(phase_labels) == phase
    out_df = peaks_df[mask][['chrom', 'start', 'end']].copy()
    out_df.to_csv(output_file, sep='\t', header=False, index=False)
    return len(out_df)
