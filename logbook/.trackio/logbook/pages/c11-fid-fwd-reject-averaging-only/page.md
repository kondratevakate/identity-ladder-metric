# C11 FID/FWD reject averaging only


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_b031400b38f3", "created_at": "2026-11-10T10:06:57+00:00", "title": "FID/FWD reject averaging only"}
-->
**PROVISIONAL (PRE-QC).** Set-level FID and FWD reject averaging strongly but cannot tell the person from a stranger.

Scale and n: Fig. 2 panel B, ratio to the stranger set; 7 image sets (brain excluded: repeat set 138 of 1326 windows).

| Printed | File value | Match | In text | Source (key) | Commit |
| --- | --- | --- | --- | --- | --- |
| FID dataset mean: 16 | 16.36 | yes | 4_results.tex | `results/cross_modality/fig_metric_summary_R.json` (fid/dataset_mean) | e8be6ff 2026-11-10 |
| FWD dataset mean: 186 | 185.6 | yes | 4_results.tex | `results/cross_modality/fig_metric_summary_R.json` (fwd/dataset_mean) | e8be6ff 2026-11-10 |
| FID true person: 0.93 | 0.929 | yes | 4_results.tex | `results/cross_modality/fig_metric_summary_R.json` (fid/ref) | e8be6ff 2026-11-10 |
| FWD true person: 0.57 | 0.572 | yes | 4_results.tex | `results/cross_modality/fig_metric_summary_R.json` (fwd/ref) | e8be6ff 2026-11-10 |
| brain repeat set: 138 | 138 | yes | 4_results.tex | `results/cross_modality/values_generative.json` (values/fid|QTIM/rungs/repeat/n) | 40ff9bd 2026-11-10 |
| brain stranger set: 1326 | 1326 | yes | 4_results.tex | `results/cross_modality/values_generative.json` (values/fid|QTIM/rungs/stranger/n) | 40ff9bd 2026-11-10 |
| brain repeat FID ratio: 8.3 | 8.322 | yes | 4_results.tex | `results/cross_modality/values_generative.json` (values/fid|QTIM/rungs/repeat/ratio_to_stranger) | 40ff9bd 2026-11-10 |

Check: the printed number equals the file value within half a unit of its last printed digit, and the printed
snippet is present in the cited .tex section (`logbook/claims.py`, test `tests/test_logbook_claims.py`).

Text issues:
- The set-size rule and the brain exclusion were fixed after seeing the brain value; the text says so (post hoc), which is correct and should stay.


---
<!-- trackio-cell
{"type": "code", "id": "cell_b58969181c82", "created_at": "2026-11-10T10:06:57+00:00", "title": "C11 machine-readable", "language": "python"}
-->
````python title=claims.py
"""Claims the CVPR 2027 paper makes, each tied to the result file and key its numbers come from.

Every check compares a number exactly as printed in the paper text with the value in the committed results file,
within the precision the text prints (half a unit of the last printed digit). A check also confirms that the
printed snippet still appears in the cited .tex file, so a later text edit cannot silently detach a claim from its
evidence. Aggregates only: no participant IDs, filenames or image-level rows are read.

Status vocabulary: "verified" (number equals the file, data QC approved), "provisional (pre-QC)" (number equals the
file, but the data precede the registration / mask QC gate), "not run" (no result, reason given).
Known text issues are listed per claim in `flags`; they are reported, never fixed here.
"""
import json
import os

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
# fundus results live in the sibling bilateral worktree (identity-ladder-bilateral), read only
BILATERAL = os.environ.get("IDENTITY_LADDER_BILATERAL",
                           os.path.normpath(os.path.join(ROOT, "..", "identity-ladder-bilateral")))
SEC = "paper/cvpr2027/sec/"
ABS, RES, SYN = SEC + "0_abstract.tex", SEC + "4_results.tex", SEC + "5_synthesis.tex"


def path(rel):
    """Results files: repo-relative, or 'bilateral:' prefix for the fundus worktree."""
    if rel.startswith("bilateral:"):
        return os.path.join(BILATERAL, rel.split(":", 1)[1])
    return os.path.join(ROOT, rel)


_CACHE = {}


def load(rel):
    if rel not in _CACHE:
        with open(path(rel), encoding="utf-8") as f:
            _CACHE[rel] = json.load(f)
    return _CACHE[rel]


def get(rel, *keys):
    o = load(rel)
    for k in keys:
        o = o[k]
    return o


BRAIN = "results/donor_ladder_brain_metrics.json"
COMMON = "results/cross_modality/common_all.json"
L16 = "results/donor_ladder_brain_lookalike16.json"
HACA = "results/donor_ladder_brain_haca3_stage1.json"
FUNDUS = "bilateral:results/fundus_ladder_metrics.json"
IRIS = "results/iris_ladder_metrics.json"
FACE = "results/face_ladder_metrics.json"
FIG = "results/cross_modality/fig_metric_summary_R.json"
AGREE = "results/cross_modality/agreement.json"
SIZE_B = "results/donor_ladder_brain_metrics_size.json"
SIZE_F = "results/face_ladder_sizes.json"
SIZE_I = "results/iris_ladder_metrics_w10.json"
GEN = "results/cross_modality/values_generative.json"


def rq(m, r, i=0):
    """QTIM random patches, both sides, repeat-anchored R (index 0 point, 1/2 interval)."""
    e = get(BRAIN, "views", "random_patches/LR", "ladder_R", m, r)
    return e["R"] if i == 0 else e["ci95"][i - 1]


def cs(d, r, m="mse", i=0):
    """Common frame, self-anchored R_self: [point, lo, hi, n]."""
    return get(COMMON, "datasets", d, "R_self", r, m)[i]


def agree(m1, m2):
    a = load(AGREE)
    return a["metric_matrix"][a["metrics"].index(m1)][a["metrics"].index(m2)]


def window_rhos():
    b = [v for m in get(SIZE_B, "rank_stability", "metrics").values() for k, v in m.items() if "_vs_" in k]
    f = [x for s in get(SIZE_F, "rank_stability").values() for m in s.values() for x in m["spearman"].values()]
    i = [v["spearman_rung_order"] for v in get(SIZE_I, "window_comparison", "per_metric").values()]
    return b + f + i


def both_corner():
    """Number of metrics inside Fig. 2 panel A's 'both' corner (x >= 0.4 and y >= 0.25)."""
    a = get(FIG, "panel_A")
    return sum(1 for v in a.values() if v["x_stranger_gap"] >= 0.4 and v["y_shrunk_gap"] >= 0.25)


def fid_set(rung):
    return get(GEN, "values", "fid|QTIM", "rungs", rung, "n")


# Each check: (label, text as printed, tex file, function returning the file value, source file, source key)
CLAIMS = [
    dict(id="C1", label="Brain MSE: mean and mirror", status="provisional (pre-QC)",
         claim="In the brain panel, MSE places the dataset mean at 0.413 and the mirror at 0.717 on the repeat-zero, "
               "stranger-one scale.",
         scale="R_repeat (repeat 0, mean of 16 strangers 1); QTIM random patches, both sides; 221 targets",
         checks=[("dataset mean", "0.413", ABS, lambda: rq("mse", "dataset_mean"), BRAIN, "views/random_patches/LR/ladder_R/mse/dataset_mean/R"),
                 ("mirror", "0.717", ABS, lambda: rq("mse", "mirror"), BRAIN, "views/random_patches/LR/ladder_R/mse/mirror/R")]),
    dict(id="C2", label="Brain T2: shrunk repeat", status="provisional (pre-QC)",
         claim="The repeat shrunk toward the mean scores closer than the repeat itself under MSE and repeat-tolerance lDDT.",
         scale="R_repeat; QTIM random patches, both sides; 23 repeat targets; two-way bootstrap intervals",
         checks=[("MSE shrunk repeat", "-0.014", RES, lambda: rq("mse", "shrink_repeat"), BRAIN, ".../mse/shrink_repeat/R"),
                 ("MSE interval low", "-0.028", RES, lambda: rq("mse", "shrink_repeat", 1), BRAIN, ".../mse/shrink_repeat/ci95[0]"),
                 ("MSE interval high", "-0.004", RES, lambda: rq("mse", "shrink_repeat", 2), BRAIN, ".../mse/shrink_repeat/ci95[1]"),
                 ("lDDT-rep shrunk repeat", "-0.084", RES, lambda: rq("lddt_rep", "shrink_repeat"), BRAIN, ".../lddt_rep/shrink_repeat/R"),
                 ("lDDT-rep low", "-0.108", RES, lambda: rq("lddt_rep", "shrink_repeat", 1), BRAIN, ".../lddt_rep/shrink_repeat/ci95[0]"),
                 ("lDDT-rep high", "-0.065", RES, lambda: rq("lddt_rep", "shrink_repeat", 2), BRAIN, ".../lddt_rep/shrink_repeat/ci95[1]")]),
    dict(id="C3", label="Mean of 16 look-alikes", status="provisional (pre-QC)",
         claim="Averaging 16 context-selected strangers comes close to the repeat in brain MRI, in QTIM and IXI.",
         scale="common self scale (target 0, 16 strangers 1); QTIM 221 targets, IXI 100; deep box on the repeat scale",
         checks=[("QTIM mean of 16", "0.375", RES, lambda: cs("QTIM", "lookalike16"), COMMON, "datasets/QTIM/R_self/lookalike16/mse"),
                 ("QTIM mean of 16 low", "0.370", RES, lambda: cs("QTIM", "lookalike16", "mse", 1), COMMON, "same, [1]"),
                 ("QTIM mean of 16 high", "0.380", RES, lambda: cs("QTIM", "lookalike16", "mse", 2), COMMON, "same, [2]"),
                 ("QTIM dataset mean", "0.503", RES, lambda: cs("QTIM", "dataset_mean"), COMMON, "datasets/QTIM/R_self/dataset_mean/mse"),
                 ("QTIM single look-alike", "0.594", RES, lambda: cs("QTIM", "lookalike"), COMMON, "datasets/QTIM/R_self/lookalike/mse"),
                 ("deep box mean of 16", "0.033", RES, lambda: get(L16, "datasets", "qtim", "views", "deep/LR", "ladder_R", "mse", "lookalike16", "R"), L16, "datasets/qtim/views/deep/LR/ladder_R/mse/lookalike16/R"),
                 ("deep box low", "-0.136", RES, lambda: get(L16, "datasets", "qtim", "views", "deep/LR", "ladder_R", "mse", "lookalike16", "ci95")[0], L16, "same, ci95[0]"),
                 ("deep box high", "0.134", RES, lambda: get(L16, "datasets", "qtim", "views", "deep/LR", "ladder_R", "mse", "lookalike16", "ci95")[1], L16, "same, ci95[1]"),
                 ("IXI mean of 16", "0.363", RES, lambda: cs("IXI", "lookalike16"), COMMON, "datasets/IXI/R_self/lookalike16/mse"),
                 ("IXI dataset mean", "0.505", RES, lambda: cs("IXI", "dataset_mean"), COMMON, "datasets/IXI/R_self/dataset_mean/mse"),
                 ("IXI opposite side", "0.662", RES, lambda: cs("IXI", "other_side"), COMMON, "datasets/IXI/R_self/other_side/mse")]),
    dict(id="C4", label="IXI scanner site", status="provisional (pre-QC)",
         claim="In IXI a stranger from the same scanner site is closer than one from another site, so part of the "
               "unrelated reference is acquisition.",
         scale="common self scale; IXI 100 targets, random patches, both sides",
         checks=[("same site", "0.874", RES, lambda: cs("IXI", "stranger_same_centre"), COMMON, "datasets/IXI/R_self/stranger_same_centre/mse"),
                 ("same site low", "0.844", RES, lambda: cs("IXI", "stranger_same_centre", "mse", 1), COMMON, "same, [1]"),
                 ("same site high", "0.899", RES, lambda: cs("IXI", "stranger_same_centre", "mse", 2), COMMON, "same, [2]"),
                 ("other site", "1.059", RES, lambda: cs("IXI", "stranger_other_centre"), COMMON, "datasets/IXI/R_self/stranger_other_centre/mse"),
                 ("other site low", "1.049", RES, lambda: cs("IXI", "stranger_other_centre", "mse", 1), COMMON, "same, [1]"),
                 ("other site high", "1.072", RES, lambda: cs("IXI", "stranger_other_centre", "mse", 2), COMMON, "same, [2]")]),
    dict(id="C5", label="HACA3+ session shift", status="provisional (pre-QC)",
         claim="HACA3+ stage 1 reduces the deep-region and parietal session shift across 23 paired scans; a contrast "
               "sensitivity, not an identity endpoint (visual verdict pending in the source).",
         scale="mean absolute session shift, standardized intensity units; 23 pairs",
         checks=[("deep before", "0.121", RES, lambda: get(HACA, "session_shift_z", "deep_before", "mean_abs"), HACA, "session_shift_z/deep_before/mean_abs"),
                 ("deep after", "0.016", RES, lambda: get(HACA, "session_shift_z", "deep_after", "mean_abs"), HACA, "session_shift_z/deep_after/mean_abs"),
                 ("parietal before", "0.059", RES, lambda: get(HACA, "session_shift_z", "parietal_before", "mean_abs"), HACA, "session_shift_z/parietal_before/mean_abs"),
                 ("parietal after", "0.033", RES, lambda: get(HACA, "session_shift_z", "parietal_after", "mean_abs"), HACA, "session_shift_z/parietal_after/mean_abs"),
                 ("pairs", "23", RES, lambda: get(HACA, "session_shift_z", "deep_after", "n"), HACA, "session_shift_z/deep_after/n")]),
    dict(id="C6", label="Fundus and iris cohorts", status="provisional (pre-QC)",
         claim="The fundus and iris panels contain the stated eyes and people, with endpoint-specific coverage.",
         scale="counts from the panel summaries",
         checks=[("GRAPE eyes", "239", RES, lambda: get(FUNDUS, "panel", "GRAPE", "counts", "units"), FUNDUS, "panel/GRAPE/counts/units"),
                 ("GRAPE people", "134", RES, lambda: get(FUNDUS, "panel", "GRAPE", "counts", "persons"), FUNDUS, "panel/GRAPE/counts/persons"),
                 ("GRAPE repeat eyes", "171", RES, lambda: get(FUNDUS, "panel", "GRAPE", "counts", "targets_per_fill", "repeat"), FUNDUS, "panel/GRAPE/counts/targets_per_fill/repeat"),
                 ("GRAPE repeat people", "97", RES, lambda: get(FUNDUS, "panel", "GRAPE", "R_self", "mse", "repeat", "n_persons"), FUNDUS, "panel/GRAPE/R_self/mse/repeat/n_persons"),
                 ("PAPILA eyes", "488", RES, lambda: get(FUNDUS, "panel", "PAPILA", "counts", "units"), FUNDUS, "panel/PAPILA/counts/units"),
                 ("PAPILA people", "244", RES, lambda: get(FUNDUS, "panel", "PAPILA", "counts", "persons"), FUNDUS, "panel/PAPILA/counts/persons"),
                 ("UTIRIS NIR eyes", "156", RES, lambda: get(IRIS, "panel", "UTIRIS_NIR", "counts", "units"), IRIS, "panel/UTIRIS_NIR/counts/units"),
                 ("UTIRIS NIR people", "78", RES, lambda: get(IRIS, "panel", "UTIRIS_NIR", "counts", "persons"), IRIS, "panel/UTIRIS_NIR/counts/persons"),
                 ("UTIRIS VIS eyes", "158", RES, lambda: get(IRIS, "panel", "UTIRIS_VIS", "counts", "units"), IRIS, "panel/UTIRIS_VIS/counts/units"),
                 ("UTIRIS VIS people", "79", RES, lambda: get(IRIS, "panel", "UTIRIS_VIS", "counts", "persons"), IRIS, "panel/UTIRIS_VIS/counts/persons"),
                 ("UPOL eyes", "128", RES, lambda: get(IRIS, "panel", "UPOL", "counts", "units"), IRIS, "panel/UPOL/counts/units"),
                 ("UPOL people", "64", RES, lambda: get(IRIS, "panel", "UPOL", "counts", "persons"), IRIS, "panel/UPOL/counts/persons")]),
    dict(id="C7", label="Face donor: pixel vs identity", status="verified",
         claim="A locally selected face donor is close under pixel error but near the unrelated reference under a "
               "whole-image ArcFace reference.",
         scale="self-anchored R (target 0, 16 strangers 1); different spatial scopes; CelebA 921 targets, IMDb-WIKI 678",
         checks=[("CelebA donor MSE", "0.399", ABS, lambda: get(FACE, "views", "celeba", "ladder_R", "mse", "lookalike", "R"), FACE, "views/celeba/ladder_R/mse/lookalike/R"),
                 ("CelebA donor ArcFace", "0.983", ABS, lambda: get(FACE, "views", "celeba", "ladder_R", "arcface_cos", "lookalike", "R"), FACE, "views/celeba/ladder_R/arcface_cos/lookalike/R"),
                 ("IMDb-WIKI donor MSE", "0.397", RES, lambda: get(FACE, "views", "imdb", "ladder_R", "mse", "lookalike", "R"), FACE, "views/imdb/ladder_R/mse/lookalike/R"),
                 ("IMDb-WIKI donor ArcFace", "0.984", RES, lambda: get(FACE, "views", "imdb", "ladder_R", "arcface_cos", "lookalike", "R"), FACE, "views/imdb/ladder_R/arcface_cos/lookalike/R")]),
    dict(id="C8", label="Per-window errors reward shrinkage", status="provisional (pre-QC)",
         claim="Per-window error and similarity metrics see the person but place a mean-shrunk copy of the target closer "
               "than the true person.",
         scale="Fig. 2 panel B, mean over 9 image sets of self-anchored R; brain, fundus, iris pre-QC",
         checks=[("MSE true person", "0.40", RES, lambda: get(FIG, "mse", "ref"), FIG, "mse/ref"),
                 ("MSE dataset mean", "0.50", RES, lambda: get(FIG, "mse", "dataset_mean"), FIG, "mse/dataset_mean"),
                 ("MSE shrunk true person", "0.29", RES, lambda: get(FIG, "mse", "shrink_ref"), FIG, "mse/shrink_ref"),
                 ("lDDT white noise", "1.76", RES, lambda: get(FIG, "lddt", "noise"), FIG, "lddt/noise"),
                 ("metrics in the 'both' corner", "0", None, both_corner, FIG, "panel_A (x >= 0.4 and y >= 0.25)")],
         flags=["4_results.tex calls lDDT 'the only per-window score that places white noise far beyond a stranger "
                "(1.76)'. That holds among the error / similarity scores (MSE 0.86, SSIM 1.03, TM-like 0.95), but LPIPS "
                "AlexNet (1.84), LPIPS VGG (1.52), Betti matching (1.66) and persistence (3.90) are also per-window and "
                "also place noise beyond a stranger. Suggest 'the only error or similarity score'."]),
    dict(id="C9", label="Protein-style scores track MSE", status="provisional (pre-QC)",
         claim="TM-like, GDT-like and superposed RMSE order the rungs almost as MSE does.",
         scale="Spearman rho of rung order, mean over datasets (agreement.json metric_matrix)",
         checks=[("TM-like vs MSE (low end)", "0.96", RES, lambda: agree("mse", "tm_like"), AGREE, "metric_matrix[mse][tm_like]"),
                 ("RMSE/rep vs MSE (high end)", "0.99", RES, lambda: agree("mse", "rmse_rep"), AGREE, "metric_matrix[mse][rmse_rep]"),
                 ("GDT-like vs MSE (inside range)", "0.97", None, lambda: agree("mse", "gdt_like"), AGREE, "metric_matrix[mse][gdt_like]")]),
    dict(id="C10", label="LPIPS rejects the mean only", status="provisional (pre-QC)",
         claim="LPIPS rejects the crude dataset mean but keeps the shrunk true person beside the true person.",
         scale="Fig. 2 panel B; 7 image sets (brain on the repeat scale, IXI not run)",
         checks=[("AlexNet dataset mean", "0.90", RES, lambda: get(FIG, "lpips_alex", "dataset_mean"), FIG, "lpips_alex/dataset_mean"),
                 ("VGG dataset mean", "1.09", RES, lambda: get(FIG, "lpips_vgg", "dataset_mean"), FIG, "lpips_vgg/dataset_mean"),
                 ("AlexNet shrunk", "0.47", RES, lambda: get(FIG, "lpips_alex", "shrink_ref"), FIG, "lpips_alex/shrink_ref"),
                 ("AlexNet true person", "0.43", RES, lambda: get(FIG, "lpips_alex", "ref"), FIG, "lpips_alex/ref"),
                 ("VGG shrunk", "0.57", RES, lambda: get(FIG, "lpips_vgg", "shrink_ref"), FIG, "lpips_vgg/shrink_ref"),
                 ("VGG true person", "0.56", RES, lambda: get(FIG, "lpips_vgg", "ref"), FIG, "lpips_vgg/ref")]),
    dict(id="C11", label="FID/FWD reject averaging only", status="provisional (pre-QC)",
         claim="Set-level FID and FWD reject averaging strongly but cannot tell the person from a stranger.",
         scale="Fig. 2 panel B, ratio to the stranger set; 7 image sets (brain excluded: repeat set 138 of 1326 windows)",
         checks=[("FID dataset mean", "16", RES, lambda: get(FIG, "fid", "dataset_mean"), FIG, "fid/dataset_mean"),
                 ("FWD dataset mean", "186", RES, lambda: get(FIG, "fwd", "dataset_mean"), FIG, "fwd/dataset_mean"),
                 ("FID true person", "0.93", RES, lambda: get(FIG, "fid", "ref"), FIG, "fid/ref"),
                 ("FWD true person", "0.57", RES, lambda: get(FIG, "fwd", "ref"), FIG, "fwd/ref"),
                 ("brain repeat set", "138", RES, lambda: fid_set("repeat"), GEN, "values/fid|QTIM/rungs/repeat/n"),
                 ("brain stranger set", "1326", RES, lambda: fid_set("stranger"), GEN, "values/fid|QTIM/rungs/stranger/n"),
                 ("brain repeat FID ratio", "8.3", RES, lambda: get(GEN, "values", "fid|QTIM", "rungs", "repeat", "ratio_to_stranger"), GEN, "values/fid|QTIM/rungs/repeat/ratio_to_stranger")],
         flags=["The set-size rule and the brain exclusion were fixed after seeing the brain value; the text says so "
                "(post hoc), which is correct and should stay."]),
    dict(id="C12", label="Window size keeps the rung order", status="provisional (pre-QC)",
         claim="Window sizes set as a fraction of the object keep the rung order; fundus has no window-size sweep.",
         scale="Spearman rho of rung order between window sizes: brain 8/16/32 mm, faces 5-20 px, iris 10 vs 16 px",
         checks=[("lowest rho", "0.72", SYN, lambda: min(window_rhos()), SIZE_B, "rank_stability (+ face_ladder_sizes, iris w10)"),
                 ("highest rho", "1.00", SYN, lambda: max(window_rhos()), SIZE_F, "rank_stability (+ brain sizes, iris w10)")]),
]


def decimals(text):
    return len(text.split(".")[1]) if "." in text else 0


def run_check(chk):
    """Returns dict(label, text, value, match, in_text) for one check; never raises on a missing file."""
    label, text, tex, fn, src, key = chk
    try:
        value = float(fn())
        err = None
    except (OSError, KeyError, IndexError, ValueError) as e:
        value, err = None, f"{type(e).__name__}: {e}"
    tol = 0.5 * 10 ** -decimals(text) + 1e-9
    match = value is not None and abs(value - float(text)) <= tol
    in_text = None
    if tex:
        with open(os.path.join(ROOT, tex), encoding="utf-8") as f:
            body = f.read().replace("$", "").replace("{,}", "").replace("\\,", "")
        in_text = text in body
    return dict(label=label, text=text, value=value, match=match, in_text=in_text, tex=tex, source=src, key=key,
                error=err)


def run_all():
    return {c["id"]: [run_check(k) for k in c["checks"]] for c in CLAIMS}

````


````output
[
 {
  "label": "FID dataset mean",
  "text": "16",
  "value": 16.362,
  "match": true,
  "in_text": true,
  "tex": "paper/cvpr2027/sec/4_results.tex",
  "source": "results/cross_modality/fig_metric_summary_R.json",
  "key": "fid/dataset_mean",
  "error": null
 },
 {
  "label": "FWD dataset mean",
  "text": "186",
  "value": 185.55,
  "match": true,
  "in_text": true,
  "tex": "paper/cvpr2027/sec/4_results.tex",
  "source": "results/cross_modality/fig_metric_summary_R.json",
  "key": "fwd/dataset_mean",
  "error": null
 },
 {
  "label": "FID true person",
  "text": "0.93",
  "value": 0.929,
  "match": true,
  "in_text": true,
  "tex": "paper/cvpr2027/sec/4_results.tex",
  "source": "results/cross_modality/fig_metric_summary_R.json",
  "key": "fid/ref",
  "error": null
 },
 {
  "label": "FWD true person",
  "text": "0.57",
  "value": 0.572,
  "match": true,
  "in_text": true,
  "tex": "paper/cvpr2027/sec/4_results.tex",
  "source": "results/cross_modality/fig_metric_summary_R.json",
  "key": "fwd/ref",
  "error": null
 },
 {
  "label": "brain repeat set",
  "text": "138",
  "value": 138.0,
  "match": true,
  "in_text": true,
  "tex": "paper/cvpr2027/sec/4_results.tex",
  "source": "results/cross_modality/values_generative.json",
  "key": "values/fid|QTIM/rungs/repeat/n",
  "error": null
 },
 {
  "label": "brain stranger set",
  "text": "1326",
  "value": 1326.0,
  "match": true,
  "in_text": true,
  "tex": "paper/cvpr2027/sec/4_results.tex",
  "source": "results/cross_modality/values_generative.json",
  "key": "values/fid|QTIM/rungs/stranger/n",
  "error": null
 },
 {
  "label": "brain repeat FID ratio",
  "text": "8.3",
  "value": 8.321595868237033,
  "match": true,
  "in_text": true,
  "tex": "paper/cvpr2027/sec/4_results.tex",
  "source": "results/cross_modality/values_generative.json",
  "key": "values/fid|QTIM/rungs/repeat/ratio_to_stranger",
  "error": null
 }
]
````
