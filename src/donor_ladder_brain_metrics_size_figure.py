"""Window size in object units (Amendment 21): R of each donor at cube edge 8, 16 and 32 mm (f = 1/16, 1/8, 1/4 of
the brain extent L = 127 mm), random cubes, both sides pooled. R: 0 = as close as the repeat scan, 1 = as far as a
random stranger. One panel per metric; one line per donor; a size not run is left out (never drawn as 0). The panel
title gives the Spearman correlation of the donor order between sizes. Aggregates only:
results/donor_ladder_brain_metrics.json (16 mm) and results/donor_ladder_brain_metrics_size.json. Pre-QC.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
REF = os.path.join(ROOT, "results", "donor_ladder_brain_metrics.json")
SRC = os.path.join(ROOT, "results", "donor_ladder_brain_metrics_size.json")
OUT = os.path.join(ROOT, "results", "figures", "fig_brain_window_sizes")
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# dataviz reference categorical order; dashed = controls built from the truth
DONORS = [("mirror", "own other side", "#2a78d6", "-"), ("mirror_block", "own other side, best fit", "#eb6834", "-"),
          ("lookalike_oracle", "best look-alike (knows answer)", "#1baf7a", "-"),
          ("lookalike", "look-alike by context", "#9b5de5", "-"), ("sibling", "sibling", "#e0b000", "-"),
          ("dataset_mean", "dataset mean", "#d6336c", "-"), ("mean_k2", "mean of 2 strangers", "#5c6b73", "-"),
          ("noise", "mean + white noise", "#0b0b0b", "--")]
METRICS = [("mse", "MSE"), ("ssim", "SSIM"), ("lddt", "lDDT-like")]
VIEW = "random_patches/LR"


def series(views, metric, fill):
    """[(size_mm, R, lo, hi)] over the sizes where this donor and metric exist."""
    out = []
    for w in sorted(views):
        R = (views[w]["ladder_R"].get(metric) or {}).get(fill)
        if R is not None:
            out.append((w, R["R"], *R["ci95"]))
    return out


def load():
    views = {16: json.load(open(REF, encoding="utf-8"))["views"][VIEW]}
    d = json.load(open(SRC, encoding="utf-8"))
    for k, v in d["sizes"].items():
        views[int(k)] = v["views"][VIEW]
    return views, d


def main():
    views, d = load()
    stab = (d.get("rank_stability") or {}).get("metrics", {})
    kept = {16: 32, **{int(k): v["positions_kept"] for k, v in d["sizes"].items()}}
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), sharey=True, facecolor=SURFACE)
    for ax, (m, title) in zip(axes, METRICS):
        ax.set_facecolor(SURFACE)
        for y in (0, 1):
            ax.axhline(y, color=GRID, lw=1.2, zorder=0)
        for fill, label, col, ls in DONORS:
            s = series(views, m, fill)
            if not s:
                continue
            x = [w for w, *_ in s]
            ax.fill_between(x, [lo for *_, lo, _ in s], [hi for *_, hi in s], color=col, alpha=0.12, lw=0)
            ax.plot(x, [r for _, r, *_ in s], ls, color=col, lw=2, marker="o", ms=5, mec=SURFACE, label=label)
        rho = stab.get(m, {})
        sub = "   ".join(f"{k.replace('_vs_', '-')}: {v:.2f}" for k, v in rho.items() if "_vs_" in k and v is not None)
        ax.set_title(title + (f"\ndonor order kept (Spearman, mm)\n{sub}" if sub else ""), fontsize=9, color=INK)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(views), [f"{w} mm\nf 1/{round(127 / w)}\n{kept[w]} cubes" for w in sorted(views)])
        ax.tick_params(colors=MUTED, labelsize=8)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    axes[0].set_ylabel("R   (0 = repeat scan, 1 = stranger)", color=MUTED, fontsize=9)
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8)
    fig.suptitle("Brain, random cubes, 221 targets: donor position vs window size (pre-QC, affine registration only)",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.text(0.02, 0.01, "Bands: 95% interval (families x positions). Block search stays +-8 mm at every size; "
             "32 mm keeps only the cubes whose window lies inside the brain.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=150, facecolor=SURFACE)
    print(f"saved {OUT}.png/.pdf")


if __name__ == "__main__":
    main()
