"""Shared matplotlib styling so every figure in the repository reads as one system.

Colour policy
-------------
Series colours are assigned by *entity*, in a fixed slot order, and never
recycled by rank -- the binomial lattice is always blue, CN-PSOR always orange,
LSM always aqua, regardless of how many series a given figure shows.  The first
three slots of the palette are validated for colour-vision-deficient separation
under an all-pairs comparison, which is the relevant test here because the three
methods routinely appear together as scatter points rather than as stacked or
adjacent marks.  Figures needing more than three entities use the extended slot
order and always carry a legend, so identity is never conveyed by colour alone.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Fixed categorical slots (light surface).
PALETTE = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "green": "#008300",
    "violet": "#4a3aa7",
    "red": "#e34948",
}
SLOTS = list(PALETTE.values())

#: Method -> colour.  Stable across the whole repository.
METHOD_COLORS = {
    "binomial": PALETTE["blue"],
    "crr": PALETTE["blue"],
    "cn_psor": PALETTE["orange"],
    "cn-psor": PALETTE["orange"],
    "lsm": PALETTE["aqua"],
    "lsm_naive": PALETTE["aqua"],
    "lsm_antithetic": PALETTE["violet"],
    "lsm_control": PALETTE["magenta"],
    "black_scholes": PALETTE["red"],
    "reference": "#52514e",
}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e3e2dd"


def use_style() -> None:
    """Install the repository's matplotlib rcParams.  Idempotent."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.dpi": 200,
            "figure.dpi": 110,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.titleweight": "medium",
            "axes.titlelocation": "left",
            "axes.titlepad": 9,
            "axes.labelsize": 9.5,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "text.color": INK,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.linewidth": 2.0,
            "lines.markersize": 5.0,
            "lines.solid_capstyle": "round",
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "legend.handlelength": 1.6,
            "legend.labelspacing": 0.35,
            "figure.constrained_layout.use": True,
        }
    )


def color(name: str) -> str:
    """Colour for a named method, falling back to the slot order."""
    return METHOD_COLORS.get(name.lower(), SLOTS[0])


def annotate(ax, x, y, text, dx=6, dy=6, fontsize=8, color=INK_2, **kw):
    """Direct label in text ink (never the series colour)."""
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=fontsize,
        color=color,
        **kw,
    )


def reference_slope(ax, x, y, slope, label, color=INK_MUTED, offset=2.0):
    """Draw a dashed guide line of a given log-log slope through ``(x, y)``.

    Used on convergence plots to show the theoretical rate against which the
    measured slope is compared.
    """
    import numpy as np

    xs = np.asarray(ax.get_xlim(), dtype=float)
    # Lift the guide clear of the data so neither the line nor its label lands on
    # a marker; only the slope is being communicated, not the intercept.
    ys = offset * y * (xs / x) ** slope
    ax.plot(xs, ys, ls=(0, (4, 3)), lw=1.2, color=color, zorder=1)
    ax.set_xlim(*xs)
    # Label at the geometric midpoint, offset off the line, so it never lands on
    # a data marker at either end of the guide.
    xm = float(np.sqrt(xs[0] * xs[1]))
    ax.annotate(
        label,
        xy=(xm, offset * y * (xm / x) ** slope),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        fontsize=8,
        color=color,
    )


def save(fig, path, caption: str | None = None):
    """Save a figure, optionally stamping a one-line caption under the axes."""
    if caption:
        fig.supxlabel(caption, fontsize=8, color=INK_MUTED, ha="left", x=0.0)
    fig.savefig(path)
    plt.close(fig)
    return path
