"""Shared look of the Identity Ladder paper figures: one font family, sizes, rung palette, markers and labels.

Every figure is drawn at the full text width (TEXTWIDTH_IN) and included at width=\\textwidth, so the sizes below
are the printed sizes. Matplotlib figures use Arial; PyMuPDF-assembled figures use Base-14 Helvetica, which has the
same metrics. Panel letters are bold capitals, as in the captions ("A,", "B,").
"""
import matplotlib.pyplot as plt

TEXTWIDTH_IN = 6.875          # CVPR two-column \textwidth
TEXTWIDTH_PT = 495.0
FONT = "Arial"                # matplotlib; PyMuPDF uses "helv"/"hebo" (Helvetica)
SIZE = 7.0                    # tick labels, legends, annotations
SIZE_SMALL = 6.3              # notes under tiles
SIZE_TITLE = 7.5              # panel titles
SIZE_LETTER = 9.0             # panel letters A, B, C
INK = "#1a1d21"
MUTED = "#5f6368"
RULE = "#d0d3d6"

# Rungs and controls: key -> (plain label, color, marker). One entry per fill, same everywhere.
# Ladder rungs are cool colors, controls (manipulated fills) warm.
RUNG = {
    "ref": ("same person", "#2b5fb8", "o"),
    "lookalike_oracle": ("best-matching person", "#38b2d8", "^"),
    "lookalike16": ("16 nearest people", "#6a4fd0", "h"),
    "dataset_mean": ("dataset average", "#0f9a8a", "s"),
    "stranger": ("random person", "#555555", "X"),
    "shrink_ref": ("same person, softened", "#e8890c", "D"),
    "noise": ("average + noise", "#c62828", "P"),
    "wrong_detail": ("average + random detail", "#e36fa8", "v"),
}
LADDER = ["ref", "lookalike_oracle", "lookalike16", "dataset_mean", "stranger"]
CONTROLS = ["shrink_ref", "noise", "wrong_detail"]
ORDER = LADDER + CONTROLS
MS, EDGE = 4.6, 0.4           # marker size and white edge


def label(key):
    return RUNG[key][0]


def color(key):
    return RUNG[key][1]


def marker(key):
    return RUNG[key][2]


def apply():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [FONT, "Helvetica", "DejaVu Sans"],
        "font.size": SIZE, "axes.titlesize": SIZE_TITLE, "axes.labelsize": SIZE, "xtick.labelsize": SIZE,
        "ytick.labelsize": SIZE, "legend.fontsize": SIZE, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": "#444444", "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42, "savefig.dpi": 300,
    })


def letter(fig, x, y, text):
    """Panel letter in figure coordinates, bold capital."""
    fig.text(x, y, text, fontsize=SIZE_LETTER, fontweight="bold", ha="left", va="baseline", color=INK)


def title(fig, x, y, text):
    fig.text(x, y, text, fontsize=SIZE_TITLE, fontweight="bold", ha="left", va="baseline", color=INK)


def handles(keys, ms=None):
    return [plt.Line2D([], [], marker=marker(k), ms=ms or MS, color=color(k), mec="white", mew=EDGE, ls="")
            for k in keys]
