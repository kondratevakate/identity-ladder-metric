"""Face metric audit figures (docs/PROTOCOL_face_metrics.md), in the style of the brain figures:
blocks figure (R per rung by block, CelebA and IMDb-WIKI), reality-B expected-vs-measured figure (Amendment 14
axiomatic target) and the shrinkage table. Reuses constants and scoring from the brain figure scripts without
editing them. Aggregates only (results/face_ladder_metrics.json). Pre-QC.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import donor_ladder_metrics_blocks_figure as BF  # noqa: E402
import donor_ladder_metrics_expected_figure as EF  # noqa: E402
import donor_ladder_metrics_shrink_table as SH  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
VIEWS = [("celeba", "CelebA", "#2a78d6", "o"), ("imdb", "IMDb-WIKI", "#eb6834", "s")]
BLOCKS = [("same photo, acquisition changed", [("aug_shift", "same photo, shifted"),
                                              ("aug_light", "same photo, retest light")]),
          ("same person", [("retest", "another photo"), ("mirror", "own mirror half")]),
          ("relatives", [("sibling", "sibling (IMDb)")]),
          ("selected by context (achievable)", [("lookalike", "most similar person\n(by surroundings)"),
                                                ("lookalike_p219", "same, pool 219")]),
          ("no selection (averages)", [("stranger", "unrelated person"), ("mean_k2", "average of\n2 people"),
                                       ("cohort_mean", "sex-age mean"), ("dataset_mean", "average of\nall people")]),
          ("oracle bound (unreachable)", [("lookalike_oracle", "best person in pool\n(oracle)"),
                                          ("lookalike_oracle_p219", "oracle, pool 219")]),
          ("controls", [("wrong_detail", "wrong detail"), ("noise", "white noise,\nsame amplitude")])]
CORE = {"aug_shift", "aug_light", "retest", "mirror", "sibling", "lookalike", "stranger", "mean_k2", "dataset_mean", "lookalike_oracle",
        "noise"}
LABEL = dict(BF.LABEL, rmse="RMSE", tm_like="TM-like", gdt_like="GDT-like", lddt_rep="lDDT, repeat tolerance",
             rmse_rep="RMSE / repeat tolerance", arcface_cos="ArcFace identity (reference)")
YLIM = (-1.6, 1.8)


def layout(full):
    xs, fills, labels, spans, x = [], [], [], [], 0.0
    for name, rungs in BLOCKS:
        rungs = [(f, lab) for f, lab in rungs if full or f in CORE]
        if not rungs:
            continue
        x0 = x
        for f, lab in rungs:
            xs.append(x)
            fills.append(f)
            labels.append(lab)
            x += 1
        spans.append((name, x0, x - 1))
        x += 0.8
    return xs, fills, labels, spans


def draw_points(ax, d, m, xs, fills, ylim, offset_step=0.22):
    for vi, (vk, vlab, col, mk) in enumerate(VIEWS):
        R = d["views"].get(vk, {}).get("ladder_R", {}).get(m)
        if not R:
            continue
        off = (vi - 0.5) * offset_step
        for x, f in zip(xs, fills):
            e = R.get(f)
            if not e:
                continue
            lo, hi = np.clip(e["ci95"], ylim[0] + 0.02, ylim[1] - 0.02)
            ax.plot([x + off] * 2, [lo, hi], color=col, lw=1.0, zorder=3)
            if not ylim[0] + 0.02 < e["R"] < ylim[1] - 0.02:      # off the scale: marker at the edge with value
                top = e["R"] > 0
                yv = ylim[1] - 0.12 if top else ylim[0] + 0.12
                ax.scatter(x + off, yv, s=18, marker="^" if top else "v", color=col, zorder=4)
                ax.annotate(f"{e['R']:.1f}", (x + off, yv), ha="center", va="bottom" if top else "top",
                            fontsize=5, color=col, xytext=(0, 3 if top else -3), textcoords="offset points")
                continue
            ax.scatter(x + off, e["R"], s=20, marker=mk, color=col, edgecolor=BF.SURFACE, linewidth=0.7, zorder=4,
                       label=vlab if f == fills[0] else None)


def style(ax, xs, labels, ylim):
    ax.set_ylim(*ylim)
    ax.set_xlim(xs[0] - 0.7, xs[-1] + 0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=90, fontsize=5.6, color=BF.MUTED)
    ax.tick_params(axis="y", labelsize=6.5, colors=BF.MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BF.MUTED)
        ax.spines[s].set_linewidth(0.6)


def blocks_figure(d, out, full):
    metrics = [m for m in d["metrics"] + [d["identity_reference"]]
               if any(m in d["views"].get(v, {}).get("ladder_R", {}) for v, *_ in VIEWS)]
    xs, fills, labels, spans = layout(full)
    ncol = 3
    nrow = int(np.ceil((len(metrics) + 2) / ncol))
    fig = plt.figure(figsize=(6.0 * ncol, 3.6 * nrow), facecolor=BF.SURFACE)
    gs = fig.add_gridspec(nrow, ncol)
    for p, m in enumerate(metrics):
        ax = fig.add_subplot(gs[p // ncol, p % ncol])
        ax.set_facecolor(BF.SURFACE)
        for (name, a, b) in spans:
            ax.axvspan(a - 0.45, b + 0.45, color="#f4f3f0", zorder=0)
            ax.text((a + b) / 2, YLIM[0] + 0.03, name, ha="center", va="bottom", fontsize=4.8, color=BF.MUTED,
                    zorder=5)
        for x, f in zip(xs, fills):
            lo, hi = BF.EXPECTED.get(f, (None, None))
            if lo is not None:
                ax.fill_between([x - 0.42, x + 0.42], lo, YLIM[1], color=BF.BAND, zorder=1)
            if hi is not None:
                ax.fill_between([x - 0.42, x + 0.42], YLIM[0], hi, color=BF.BAND, zorder=1)
        for h in (0, 1):
            ax.axhline(h, color=BF.MUTED, lw=0.7, ls=(0, (2, 2)), zorder=2)
        draw_points(ax, d, m, xs, fills, YLIM)
        tt = []
        for vk, vlab, *_ in VIEWS:
            ex = d["views"].get(vk, {}).get("expectations", {}).get(m, {})
            passed = [v["pass"] for v in ex.values() if isinstance(v, dict)]
            if passed:
                tt.append(f"{vlab} {sum(passed)}/{len(passed)}")
        ax.set_title(f"{LABEL.get(m, m)}: expectations met " + ", ".join(tt), fontsize=8, color=BF.INK, loc="left")
        style(ax, xs, labels, YLIM)
    k = len(metrics)
    gax = fig.add_subplot(gs[k // ncol, k % ncol:])
    grid(gax, d, d["metrics"])
    h, lab = fig.axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", fontsize=8, frameon=False, title="faces, 16 px windows", title_fontsize=8)
    fig.text(0.5, 0.003, "R = (d - d_self) / (d_stranger - d_self): 0 = the target itself, 1 = unrelated person (mean of 16; "
             "Amendment F1). Bars: 95% interval, bootstrap over families and window positions. Blue bands: "
             "reality-B expectations (mirror, sibling, averages below 1; wrong detail at or above 0.9; noise above 1)."
             "\n" + d.get("status", ""), ha="center", fontsize=6.5, color=BF.MUTED)
    fig.tight_layout(rect=(0, 0.025, 0.93, 1))
    save(fig, out)


def grid(ax, d, metrics):
    names = list(BF.SHORT)
    cols = [(vk, vlab, n) for vk, vlab, *_ in VIEWS for n in names]
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_ylim(len(metrics) - 0.5, -0.5)
    for i, m in enumerate(metrics):
        for j, (vk, vlab, n) in enumerate(cols):
            v = d["views"].get(vk, {}).get("expectations", {}).get(m, {}).get(n)
            if not isinstance(v, dict):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=5, color=BF.MUTED)
                continue
            ax.add_patch(plt.Rectangle((j - 0.45, i - 0.4), 0.9, 0.8, color=BF.GOOD if v["pass"] else BF.BAD,
                                       alpha=0.18, lw=0))
            val = v["value"]
            shown = "/".join(f"{x:.1f}" for x in val) if isinstance(val, list) else f"{val:.2f}"
            ax.text(j, i, ("pass" if v["pass"] else "fail") + f"\n{shown}", ha="center", va="center", fontsize=4.4,
                    color=BF.INK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{vlab}: {BF.SHORT[n]}" for _, vlab, n in cols], rotation=90, fontsize=5, color=BF.MUTED)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([LABEL.get(m, m) for m in metrics], fontsize=6, color=BF.INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Reality-B expectations (value = R)", fontsize=8, color=BF.INK, loc="left")


PARTLY = list(EF.PARTLY) + ["retest", "aug_shift", "aug_light"]
SHOWN = [("mse", "MSE"), ("ssim", "SSIM"), ("lddt", "lDDT-like"), ("acc_cohort", "ACC vs sex-age mean"),
         ("arcface_cos", "ArcFace identity (reference)")]
ORDER = ["aug_shift", "aug_light", "retest", "lookalike_oracle", "lookalike", "mirror", "sibling", "dataset_mean", "cohort_mean", "mean_k2",
         "stranger", "wrong_detail", "noise"]


def expected_figure(d, out):
    xs_all, fills_all, labels_all, _ = layout(full=True)
    lab_of = dict(zip(fills_all, labels_all))
    block_of = {f: name for name, rungs in BLOCKS for f, _ in rungs}
    block_col = dict(zip([n for n, _ in BLOCKS], ["#7f8b99", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                                                  "#6250d6"]))
    fills = ORDER
    xs = list(range(len(fills)))
    ylim = (-1.6, 2.3)
    fig, axes = plt.subplots(len(VIEWS), len(SHOWN), figsize=(4.6 * len(SHOWN), 5.2 * len(VIEWS)),
                             facecolor=BF.SURFACE, sharey=True, squeeze=False)
    for r, (vk, vlab, col, mk) in enumerate(VIEWS):
        view = d["views"].get(vk)
        for c, (m, mlab) in enumerate(SHOWN):
            ax = axes[r, c]
            ax.set_facecolor(BF.SURFACE)
            for h in (0, 1):
                ax.axhline(h, color=BF.MUTED, lw=0.7, ls=(0, (2, 2)), zorder=1)
            for x, f in zip(xs, fills):
                ax.add_patch(plt.Rectangle((x - 0.42, ylim[0]), 0.84, 0.04 * (ylim[1] - ylim[0]),
                                           color=block_col[block_of[f]], lw=0, zorder=2))
            vx = [(x, EF.AXIOM[f]) for x, f in zip(xs, fills) if f in EF.AXIOM]
            ax.scatter([x for x, _ in vx], [v for _, v in vx], s=46, marker="D", facecolor="white",
                       edgecolor=EF.THEORY, linewidth=1.1, zorder=4, label="target value (axioms of reality B)")
            px = [x for x, f in zip(xs, fills) if f in PARTLY]
            ax.scatter(px, [1.0] * len(px), s=60, marker="v", facecolor="white", edgecolor=EF.THEORY, linewidth=1.0,
                       zorder=4, label="target: below 1, no value")
            xn = xs[fills.index("noise")]
            ax.annotate("", (xn, 1.55), (xn, 1.02), arrowprops=dict(arrowstyle="->", color=EF.THEORY, lw=1.4))
            ax.text(xn - 0.1, 1.58, "> 1", ha="center", fontsize=7, color=BF.INK)
            R = (view or {}).get("ladder_R", {}).get(m, {})
            pts = [(x, R[f]) for x, f in zip(xs, fills) if R.get(f)]
            clip = lambda v: float(np.clip(v, ylim[0] + 0.05, ylim[1] - 0.05))  # noqa: E731
            if pts:
                ax.plot([x + 0.18 for x, _ in pts], [clip(e["R"]) for _, e in pts], color=EF.MEAS, lw=0.9,
                        alpha=0.7, zorder=3)
                for x, e in pts:
                    ax.plot([x + 0.18] * 2, [clip(e["ci95"][0]), clip(e["ci95"][1])], color=EF.MEAS, lw=1.0,
                            zorder=3)
                    if clip(e["R"]) != e["R"]:
                        ax.annotate(f"{e['R']:.1f}", (x + 0.18, clip(e["R"])), fontsize=5, color=EF.MEAS,
                                    ha="center", xytext=(0, 4 if e["R"] > 0 else -8), textcoords="offset points")
                ax.scatter([x + 0.18 for x, _ in pts], [clip(e["R"]) for _, e in pts], s=20, color=EF.MEAS, zorder=5,
                           label=f"measured {mlab}")
            sc = (view or {}).get("realityB_score", {}).get(m)
            if sc:
                nb, nt = sc["partly_below_1"]
                rms = f"{sc['rms']:.2f}" if sc["rms"] is not None else "n/a"
                noise = {True: "yes", False: "no", None: "n/a"}[sc["noise_above_1"]]
                ax.set_title(f"{vlab}, {mlab}\nvalue fit RMS {rms}; noise > 1: {noise}; partly correct < 1: {nb}/{nt}",
                             fontsize=7.5, color=BF.INK, loc="left")
            else:
                ax.set_title(f"{vlab}, {mlab}", fontsize=7.5, color=BF.INK, loc="left")
            ax.set_ylim(*ylim)
            ax.set_xlim(-0.7, len(xs) - 0.3)
            ax.set_xticks(xs)
            ax.set_xticklabels([lab_of[f] for f in fills], rotation=90, fontsize=5.6, color=BF.MUTED)
            ax.tick_params(axis="y", labelsize=7, colors=BF.MUTED)
            if c == 0:
                ax.set_ylabel("R  (0 = the target itself, 1 = unrelated person)", fontsize=7.5, color=BF.INK)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if r == 0 and c == 0:
                ax.legend(loc="upper left", fontsize=6.5, frameon=False)
    names = [n for n, _ in BLOCKS]
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=block_col[n]) for n in names], names, loc="lower center",
               ncol=len(names), fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.03), title="block (strip under rung)",
               title_fontsize=7)
    fig.text(0.5, 0.005, "Target from the axioms of reality B only (Amendment 14): no information 0.5; mean of k "
             "unrelated people 0.5 + 0.5/k; wrong detail 1; white noise above 1; partly correct rungs below 1. "
             + d.get("status", ""), ha="center", fontsize=6, color=BF.MUTED, wrap=True)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save(fig, out)


PAIRS = [("retest", "shrink_repeat", "A-E4a repeat blend not rewarded", "another photo"),
         ("mirror", "shrink_mirror", "A-E4b mirror blend not rewarded", "own mirror half")]


def shrink_table(d, out):
    header = ["dataset", "metric"]
    for _, _, _, lab in PAIRS:
        header += [f"{lab}\nR donor", "R blend", "blend - donor\n[95% CI]", "verdict"]
    rows, colors = [], []
    for vk, vlab, *_ in VIEWS:
        v = d["views"].get(vk, {})
        for m in d["metrics"]:
            R = v.get("ladder_R", {}).get(m, {})
            ex = v.get("expectations_realityA", {}).get(m, {})
            row, col = [vlab, LABEL.get(m, m)], ["white", "white"]
            for donor, blend, key, _ in PAIRS:
                e = ex.get(key)
                if not e or donor not in R or blend not in R:
                    row += ["n/a"] * 4
                    col += ["white"] * 4
                    continue
                txt, c = SH.verdict(e["lower95"], e["upper95"])
                row += [f"{R[donor]['R']:.2f}", f"{R[blend]['R']:.2f}",
                        f"{e['value']:+.2f} [{e['lower95']:+.2f}, {e['upper95']:+.2f}]", txt]
                col += ["white", "white", c, c]
            rows.append(row)
            colors.append(col)
            print(" | ".join(x.replace("\n", " ") for x in row))
    fig, ax = plt.subplots(figsize=(16, 0.36 * len(rows) + 1.6), facecolor=BF.SURFACE)
    ax.axis("off")
    tab = ax.table(cellText=rows, colLabels=header, cellColours=colors, loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(7)
    tab.scale(1, 1.3)
    for (i, j), cell in tab.get_celld().items():
        cell.set_edgecolor("#d9d8d4")
        if i == 0:
            cell.set_text_props(weight="bold", fontsize=7)
            cell.set_height(cell.get_height() * 1.6)
    lam = "; ".join(f"{vlab}: " + ", ".join(f"{k} {x:.2f}" for k, x in d["lambda_shrink_by_source"][vk].items())
                    for vk, vlab, *_ in VIEWS)
    ax.set_title("Shrinkage test on faces: a real donor faded towards the dataset mean, blend = mean + lambda (donor - "
                 f"mean), lambda = r of the rung ({lam}). R: 0 = the target itself, 1 = unrelated person.", fontsize=8,
                 loc="left")
    fig.text(0.01, 0.01, "Intervals: bootstrap over families and window positions. " + d.get("status", ""),
             fontsize=6.5, color=BF.MUTED)
    fig.tight_layout()
    save(fig, out)


def save(fig, out):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, f"{out}.{ext}"), dpi=200, facecolor=BF.SURFACE)
    plt.close(fig)
    print(os.path.abspath(os.path.join(ROOT, f"{out}.png")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/face_ladder_metrics.json")
    ap.add_argument("--prefix", default="results/figures/fig_face_metrics")
    a = ap.parse_args()
    d = json.load(open(os.path.join(ROOT, a.json) if not os.path.isabs(a.json) else a.json))
    blocks_figure(d, a.prefix + "_blocks", full=False)
    blocks_figure(d, a.prefix + "_blocks_full", full=True)
    expected_figure(d, a.prefix + "_expected")
    shrink_table(d, a.prefix + "_shrink_table")


if __name__ == "__main__":
    main()
