"""Publication plotting helpers for BASS/Fixed simulation figures.

The defaults here are intentionally plain: clean axes, readable labels, a
consistent palette, and vector-friendly exports.  Notebooks should call
``mplstyle()`` once near the top and ``save_figure()`` for final assets.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from math import ceil

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter

COLORS = {
    "blue": "#0C5DA5",
    "red": "#FF6B6B",
    "turquoise": "#2EC4B6",
    "violet": "#8E6DB0",
    "purple": "#AB3A62",
    "yellow": "#F2C46F",
    "lightblue": "#64AEE3",
    "green": "#1C875C",
    "brown": "#A17E33",
}

SERIES_COLORS = {
    "trusts": COLORS["blue"],
    "bass": COLORS["red"],
    "bassv3": COLORS["red"],
    "bassv4": COLORS["green"],
    "mps": COLORS["violet"],
    "exact": "#222222",
    "gamma": COLORS["turquoise"],
}

MARKERS = {
    "trusts": "o",
    "bass": "s",
    "bassv3": "s",
    "bassv4": "^",
    "mps": "D",
    "exact": "X",
    "gamma": "o",
}


def mplstyle(*, usetex: bool = False, dpi: int = 150) -> None:
    """Set a consistent Matplotlib style for paper figures."""
    plt.rcParams.update(
        {
            "xtick.direction": "in",
            "xtick.major.size": 4.5,
            "xtick.minor.size": 2.5,
            "xtick.major.width": 1.0,
            "xtick.minor.width": 0.8,
            "xtick.labelsize": 10,
            "xtick.minor.visible": True,
            "ytick.direction": "in",
            "ytick.major.size": 4.5,
            "ytick.minor.size": 2.5,
            "ytick.major.width": 1.0,
            "ytick.minor.width": 0.8,
            "ytick.labelsize": 10,
            "ytick.minor.visible": True,
            "axes.facecolor": "white",
            "axes.grid": True,
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "axes.linewidth": 1.0,
            "axes.prop_cycle": cycler("color", COLORS.values()),
            "axes.xmargin": 0.03,
            "axes.ymargin": 0.05,
            "grid.color": "gray",
            "grid.linestyle": ":",
            "grid.alpha": 0.28,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.title_fontsize": 9,
            "legend.labelspacing": 0.35,
            "legend.handlelength": 1.5,
            "legend.columnspacing": 1.2,
            "legend.borderpad": 0.4,
            "figure.facecolor": "white",
            "figure.dpi": dpi,
            "figure.figsize": (7.0, 7.0 / 1.6),
            "figure.constrained_layout.use": True,
            "savefig.facecolor": "white",
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "font.size": 11,
            "font.family": "sans-serif",
            "lines.linewidth": 2.0,
            "lines.markersize": 5.5,
            "scatter.marker": "o",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "text.usetex": usetex,
        }
    )
    if usetex:
        mpl.rcParams["text.latex.preamble"] = r"\usepackage{braket}"


def figax(w: float = 7.0, h: float | None = None, **kwargs) -> tuple[Figure, Axes]:
    """Return a single-axis figure with a paper-friendly aspect ratio."""
    if h is None:
        h = w / 1.6
    return plt.subplots(1, 1, figsize=(w, h), constrained_layout=True, **kwargs)


def grid(
    n: int,
    nrows: int = 1,
    *,
    w: float = 3.2,
    h: float | None = None,
    sharexy: bool = False,
    **kwargs,
) -> tuple[Figure, Iterable[Axes]]:
    """Return a constrained-layout grid and an iterator over its axes."""
    h = w if h is None else h
    ncols = ceil(n / nrows)
    if sharexy:
        kwargs["sharex"] = True
        kwargs["sharey"] = True
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(w * ncols, h * nrows),
        constrained_layout=True,
        **kwargs,
    )
    axs_list = axs.flatten() if nrows != 1 or ncols != 1 else [axs]
    return fig, iter(axs_list)


def polish_axes(ax: Axes, *, logx: bool = False, logy: bool = False) -> None:
    """Apply final axis polish that notebooks often forget."""
    ax.tick_params(which="both", top=True, right=True)
    if logx:
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
        ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    if logy:
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.yaxis.set_minor_locator(AutoMinorLocator())


def polish_figure(fig: Figure) -> None:
    """Polish every axis in a figure."""
    for ax in fig.axes:
        if isinstance(ax, Axes):
            polish_axes(
                ax,
                logx=ax.get_xscale() == "log",
                logy=ax.get_yscale() == "log",
            )


def save_figure(
    fig: Figure,
    path: str,
    *,
    dpi: int = 600,
    formats: tuple[str, ...] = ("pdf", "svg", "png"),
) -> None:
    """Save a figure as vector assets plus a high-resolution PNG."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    polish_figure(fig)
    for ext in formats:
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.02}
        if ext.lower() in {"png", "jpg", "jpeg", "tif", "tiff"}:
            kwargs["dpi"] = dpi
        fig.savefig(f"{path}.{ext}", **kwargs)
    print(f"Saved {path}." + "/.".join(formats))
