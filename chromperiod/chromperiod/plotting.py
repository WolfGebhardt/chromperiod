"""
plotting.py — Publication-quality plots for chromperiod CWT analysis.

All plots match the style of the manuscript figures (Torrence & Compo 1998
conventions, seaborn ticks theme, colorblind-friendly palettes).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from typing import Optional, Tuple, List


# Default style
STYLE_DEFAULTS = {
    'scalogram_cmap': 'YlOrRd',
    'sig_contour_color': 'white',
    'coi_hatch': '//',
    'coi_alpha': 0.3,
    'gws_color': '#0279EE',
    'threshold_color': '#FF9400',
    'dominant_period_color': '#75A025',
    'figsize_scalogram': (10, 6),
    'figsize_gws': (6, 4),
    'figsize_triple': (14, 8),
    'dpi': 150,
}


def _setup_style():
    """Apply consistent matplotlib style."""
    import seaborn as sns
    sns.set_theme(style='ticks', font_scale=1.0)
    plt.rcParams.update({
        'axes.spines.top': False,
        'axes.spines.right': False,
        'font.family': 'sans-serif',
    })


def plot_scalogram(
    result,
    ax=None,
    title=None,
    show_significance=True,
    show_coi=True,
    show_dominant_period=True,
    period_units='peaks',
    cmap=None,
    figsize=None,
):
    """
    Plot the wavelet power scalogram with significance contours and COI.

    Parameters
    ----------
    result : CWTResult
        CWT analysis result.
    ax : matplotlib.axes.Axes or None
        Axes to plot on. If None, creates a new figure.
    title : str or None
        Plot title.
    show_significance : bool
        If True, overlay significance contours.
    show_coi : bool
        If True, shade the cone of influence.
    show_dominant_period : bool
        If True, mark the dominant period with a horizontal line.
    period_units : str
        'peaks' or 'mbp'. Use 'mbp' if result.periods_mbp is available.
    cmap : str or None
        Colormap for power. Default: 'YlOrRd'.
    figsize : tuple or None
        Figure size.

    Returns
    -------
    fig, ax
    """
    _setup_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or STYLE_DEFAULTS['figsize_scalogram'])
    else:
        fig = ax.get_figure()

    if cmap is None:
        cmap = STYLE_DEFAULTS['scalogram_cmap']

    # Choose period axis
    if period_units == 'mbp' and result.periods_mbp is not None:
        periods = result.periods_mbp
        period_label = 'Period (Mbp)'
        dom_period = result.dominant_period_mbp
        coi_periods = result.coi * (result.median_spacing_bp / 1e6) if result.median_spacing_bp else result.coi
    else:
        periods = result.periods
        period_label = 'Period (peak-index units)'
        dom_period = result.dominant_period
        coi_periods = result.coi

    N = result.n_peaks
    x = np.arange(N)

    # Log-scale power for display
    power_log = np.log2(result.power + 1e-10)

    # Plot scalogram
    im = ax.contourf(
        x, np.log2(periods), power_log,
        levels=20, cmap=cmap, extend='both'
    )

    # Significance contours
    if show_significance:
        # Threshold in log2 space
        threshold_log = np.log2(result.ar1_threshold + 1e-10)
        for j in range(len(periods)):
            sig_line = np.full(N, threshold_log[j])
        # Use contour at significance boundary
        sig_ratio = result.power / (result.ar1_threshold[:, np.newaxis] + 1e-10)
        ax.contour(x, np.log2(periods), sig_ratio, levels=[1.0],
                   colors=STYLE_DEFAULTS['sig_contour_color'], linewidths=1.0)

    # COI shading
    if show_coi:
        coi_log = np.log2(np.maximum(coi_periods, periods[0]))
        # Shade above COI (edge-affected region)
        ax.fill_between(x, coi_log, np.log2(periods[-1]),
                        alpha=STYLE_DEFAULTS['coi_alpha'],
                        color='gray',
                        hatch=STYLE_DEFAULTS['coi_hatch'],
                        label='COI')

    # Dominant period line
    if show_dominant_period and dom_period is not None:
        ax.axhline(np.log2(dom_period), color=STYLE_DEFAULTS['dominant_period_color'],
                   linestyle='--', linewidth=1.5,
                   label=f'Dominant: {dom_period:.1f}')

    # Axes formatting
    yticks_log2 = ax.get_yticks()
    ax.set_yticks(yticks_log2)
    ax.set_yticklabels([f'{2**y:.1f}' for y in yticks_log2])
    ax.set_xlabel('Peak index')
    ax.set_ylabel(period_label)

    if title is None:
        chrom_str = result.chromosome or 'all'
        title = (f'{chrom_str} | N={result.n_peaks} | '
                 f'dom={dom_period:.1f} | sig95={result.sig95:.1%}')
    ax.set_title(title, fontsize=10)

    plt.colorbar(im, ax=ax, label='log₂ power')

    return fig, ax


def plot_gws(
    result,
    ax=None,
    title=None,
    period_units='peaks',
    show_threshold=True,
    show_dominant_period=True,
    figsize=None,
    color=None,
):
    """
    Plot the global wavelet spectrum (GWS) with AR1 significance threshold.

    Parameters
    ----------
    result : CWTResult
        CWT analysis result.
    ax : matplotlib.axes.Axes or None
        Axes to plot on.
    title : str or None
        Plot title.
    period_units : str
        'peaks' or 'mbp'.
    show_threshold : bool
        If True, plot the AR1 significance threshold.
    show_dominant_period : bool
        If True, mark the dominant period.
    figsize : tuple or None
        Figure size.
    color : str or None
        Line color for GWS.

    Returns
    -------
    fig, ax
    """
    _setup_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or STYLE_DEFAULTS['figsize_gws'])
    else:
        fig = ax.get_figure()

    if color is None:
        color = STYLE_DEFAULTS['gws_color']

    # Choose period axis
    if period_units == 'mbp' and result.periods_mbp is not None:
        periods = result.periods_mbp
        period_label = 'Period (Mbp)'
        dom_period = result.dominant_period_mbp
    else:
        periods = result.periods
        period_label = 'Period (peak-index units)'
        dom_period = result.dominant_period

    gws = result.gws
    threshold = result.ar1_threshold

    ax.semilogx(periods, gws, color=color, linewidth=2, label='GWS')

    if show_threshold:
        ax.semilogx(periods, threshold, color=STYLE_DEFAULTS['threshold_color'],
                    linestyle=':', linewidth=1.5, label='AR1 95%')

    if show_dominant_period and dom_period is not None:
        ax.axvline(dom_period, color=STYLE_DEFAULTS['dominant_period_color'],
                   linestyle='--', linewidth=1.5,
                   label=f'Dominant: {dom_period:.1f}')

    ax.set_xlabel(period_label)
    ax.set_ylabel('Mean power')
    ax.legend(fontsize=8)

    if title is None:
        chrom_str = result.chromosome or 'all'
        title = f'GWS — {chrom_str} | sig95={result.sig95:.1%}'
    ax.set_title(title, fontsize=10)

    return fig, ax


def plot_triple_comparison(
    consecutive_result,
    randomized_result,
    linear_result,
    period_units='peaks',
    figsize=None,
    suptitle=None,
):
    """
    Three-panel figure comparing consecutive, randomized, and linear CWT results.

    Matches the manuscript Figure 1 style: signal trace (top) + scalogram
    (bottom) for each of the three conditions.

    Parameters
    ----------
    consecutive_result : CWTResult
        CWT result for the consecutive-peak signal.
    randomized_result : CWTResult
        CWT result for the randomized control.
    linear_result : CWTResult
        CWT result for the linear interpolation control.
    period_units : str
        'peaks' or 'mbp'.
    figsize : tuple or None
        Figure size.
    suptitle : str or None
        Overall figure title.

    Returns
    -------
    fig
    """
    _setup_style()

    if figsize is None:
        figsize = STYLE_DEFAULTS['figsize_triple']

    fig, axes = plt.subplots(2, 3, figsize=figsize,
                              gridspec_kw={'height_ratios': [1, 3]})

    results = [consecutive_result, randomized_result, linear_result]
    labels = ['A: Consecutive peaks', 'B: Randomized control', 'C: Linear interpolation']

    for col, (res, label) in enumerate(zip(results, labels)):
        ax_sig = axes[0, col]
        ax_cwt = axes[1, col]

        # Signal trace
        ax_sig.plot(res.signal, color='#333333', linewidth=0.5, alpha=0.8)
        ax_sig.set_title(label, fontsize=9, fontweight='bold')
        ax_sig.set_ylabel('Standardized signal' if col == 0 else '')
        ax_sig.set_xlabel('')
        ax_sig.tick_params(labelbottom=False)

        # Scalogram
        plot_scalogram(res, ax=ax_cwt, period_units=period_units,
                       title='', show_dominant_period=True)
        ax_cwt.set_xlabel('Peak index')
        if col > 0:
            ax_cwt.set_ylabel('')

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight='bold', y=1.01)

    plt.tight_layout()
    return fig


def plot_phase_signal(
    result,
    peaks_df,
    phase_labels,
    compartment_labels=None,
    period_units='peaks',
    figsize=None,
):
    """
    Plot the band-pass reconstructed phase signal with peak classifications.

    Parameters
    ----------
    result : CWTResult
        CWT analysis result.
    peaks_df : pd.DataFrame
        DataFrame with peak positions.
    phase_labels : np.ndarray of str
        Phase labels ('high', 'low', 'background').
    compartment_labels : np.ndarray of str or None
        Compartment labels ('A' or 'B') for coloring.
    period_units : str
        'peaks' or 'mbp'.
    figsize : tuple or None
        Figure size.

    Returns
    -------
    fig, axes
    """
    from .phase import reconstruct_bandpass

    _setup_style()

    if figsize is None:
        figsize = (12, 5)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    x = np.arange(result.n_peaks)

    # Top panel: original signal
    ax1.plot(x, result.signal, color='#333333', linewidth=0.5, alpha=0.7)
    ax1.set_ylabel('Standardized signal')
    ax1.set_title(f'Phase classification — {result.chromosome or "all"} | '
                  f'dom={result.dominant_period:.1f} | sig95={result.sig95:.1%}',
                  fontsize=10)

    # Bottom panel: phase labels
    phase = np.asarray(phase_labels)
    colors = {'high': '#E74C3C', 'low': '#3498DB', 'background': '#AAAAAA'}

    for label, color in colors.items():
        mask = phase == label
        if mask.sum() > 0:
            ax2.scatter(x[mask], result.signal[mask], c=color, s=2,
                        alpha=0.6, label=label, rasterized=True)

    ax2.set_xlabel('Peak index')
    ax2.set_ylabel('Standardized signal')
    ax2.legend(markerscale=4, fontsize=8)

    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_genome_wide_summary(
    results_dict,
    period_units='mbp',
    figsize=None,
    n_cols=5,
):
    """
    Plot a grid of GWS panels for all chromosomes.

    Parameters
    ----------
    results_dict : dict
        Dictionary mapping chromosome name to CWTResult.
    period_units : str
        'peaks' or 'mbp'.
    figsize : tuple or None
        Figure size.
    n_cols : int
        Number of columns in the grid.

    Returns
    -------
    fig
    """
    _setup_style()

    chroms = sorted(results_dict.keys())
    n = len(chroms)
    n_rows = int(np.ceil(n / n_cols))

    if figsize is None:
        figsize = (n_cols * 3, n_rows * 2.5)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).flatten()

    for i, chrom in enumerate(chroms):
        res = results_dict[chrom]
        plot_gws(res, ax=axes[i], period_units=period_units,
                 title=f'{chrom}\n{res.sig95:.1%}', show_threshold=True)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    return fig
