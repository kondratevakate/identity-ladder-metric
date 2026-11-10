"""Prepare a manuscript from saved aggregate results; no experiments are run."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import re
import shutil
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TEMPLATE = "https://media.eventhosts.cc/Conferences/CVPR2027/author-kit-CVPR2027-v1-latex.zip"
KEYS = ["pepe2005", "janes2009", "daugman2004", "wang2004", "blau2018",
        "zhang2018lpips", "maierhein2024", "huang2023grape", "hosseini2010"]


def get(url):
    request = urllib.request.Request(url, headers={"User-Agent": "IdentityLadder-manuscript/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(fetch_template=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    import numpy as np

    figures = HERE / "figures"
    figures.mkdir(exist_ok=True)
    outputs = HERE / "generated"
    outputs.mkdir(exist_ok=True)
    names = ["population_null_synthetic", "pupil_ladder", "label_robustness", "grape_calibration", "iris_codes", "iris_calibration"]
    data = {name: json.loads((ROOT / "results" / (name + ".json")).read_text()) for name in names}
    provenance = {
        "date": "2026-11-08", "scope": "Saved aggregate outputs only; no participant images read",
        "sources": {f"results/{name}.json": digest(ROOT / "results" / (name + ".json")) for name in names},
        "source_isbi_tex_sha256": digest(ROOT / "paper/isbi2027/overleaf/main.tex"),
        "source_isbi_bib_sha256": digest(ROOT / "paper/isbi2027/overleaf/refs.bib"),
        "preparation_code_sha256": digest(Path(__file__)),
    }
    previous = HERE / "evidence.json"
    if previous.exists():
        old = json.loads(previous.read_text())
        provenance.update({key: value for key, value in old.items() if key.startswith("template_")})

    # Keep the old bibliography intact. Select entries without editing their metadata.
    original = (ROOT / "paper/isbi2027/overleaf/refs.bib").read_text()
    entries = re.split(r"(?=^@)", original, flags=re.MULTILINE)
    selected = []
    for key in KEYS:
        matches = [entry.strip() for entry in entries if re.match(r"@\w+\{" + key + r",", entry)]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one bibliography entry for {key}")
        selected.extend(matches)
    # New reference metadata comes directly from Crossref, cached for repeatability.
    cache = HERE / "retfound.crossref.json"
    if not cache.exists():
        cache.write_bytes(get("https://api.crossref.org/works/10.1038/s41586-023-06555-x"))
    record = json.loads(cache.read_text())["message"]
    authors = " and ".join((a["family"] + ", " + a.get("given", ""))
                           if "family" in a else "{" + a["name"] + "}"
                           for a in record["author"])
    authors = authors.replace("&", r"\&").replace("\u2019", "'")
    selected.append("@article{zhou2023retfound,\n"
                    f"  author = {{{authors}}},\n"
                    f"  title = {{{record['title'][0]}}},\n"
                    "  journal = {Nature},\n  year = {2023},\n"
                    f"  volume = {{{record['volume']}}},\n  pages = {{{record['page'].replace('-', '--')}}},\n"
                    "  doi = {10.1038/s41586-023-06555-x}\n}")
    (HERE / "refs.bib").write_text("% Selected original records and Crossref metadata; see citation_report.md.\n\n" + "\n\n".join(selected) + "\n")

    if fetch_template:
        package = get(TEMPLATE)
        provenance["template_url"] = TEMPLATE
        provenance["template_zip_sha256"] = hashlib.sha256(package).hexdigest()
        with zipfile.ZipFile(io.BytesIO(package)) as archive:
            names_in_zip = archive.namelist()
            copied = []
            for name in names_in_zip:
                base = Path(name).name
                if base in {"cvpr.sty", "ieeenat_fullname.bst"}:
                    (HERE / base).write_bytes(archive.read(name))
                    copied.append(base)
            # Preserve upstream documentation and license without taking their sample paper.
            for name in names_in_zip:
                base = Path(name).name
                if base.lower() in {"readme.md", "license", "license.txt"}:
                    target = HERE / "template-notices" / base
                    target.parent.mkdir(exist_ok=True)
                    target.write_bytes(archive.read(name))
            provenance["template_files"] = sorted(copied)
        if not (HERE / "cvpr.sty").exists():
            raise ValueError("The official template archive did not contain cvpr.sty")

    # Table 1: all numeric cells are rendered from saved aggregates.
    rows = []
    subset = [("repeat", "Repeat capture"), ("shrink_0.5", "50\\% residual"),
              ("shrink_0", "Population mean"), ("wrong_residual", "Wrong detail")]
    provenance["population_null"] = {}
    for modality, title in [("tissue", "Tissue"), ("eyes", "Eyes")]:
        aggregate = data["population_null_synthetic"]["modalities"][modality]["aggregate"]
        provenance["population_null"][modality] = {key: aggregate[key] for key, _ in subset}
        for key, label in subset:
            item = aggregate[key]
            values = [item["c"][metric]["mean"] for metric in ["rmse", "ssim"]]
            values += [item["mean_control"][metric]["mean"] for metric in ["residual_ratio", "target_projection"]]
            assert all(np.isfinite(values))
            rows.append(f"{title} & {label} & " + " & ".join(f"{v:.3f}" for v in values) + r" \\")
        if modality == "tissue":
            rows.append(r"\midrule")
    (outputs / "null_rows.tex").write_text("\n".join(rows) + "\n" + r"\bottomrule" + "\n")

    # Table 2: unit counts and original cluster-bootstrap pilot intervals.
    grape = data["grape_calibration"]["feature_spaces"]["retfound_released"]
    grape_conditions = [
        ("same_eye_other_visit | same camera", "Same eye, later visit, same camera"),
        ("same_eye_other_visit | different camera", "Same eye, later visit, other camera"),
        ("fellow_eye_same_visit | same camera", "Fellow eye, same visit, same camera"),
    ]
    ocular = []
    provenance["ocular_pilots"] = {}
    for key, label in grape_conditions:
        item = grape[key]
        lo, hi = item["mean_c_ci95_patient_bootstrap"]
        ocular.append(f"GRAPE & {label} & {item['n_patients']} & {item['n_pairs']} & {item['mean_c_auc']:.3f} [{lo:.3f}, {hi:.3f}]" + r" \\")
        provenance["ocular_pilots"][key] = item
    iris = data["iris_codes"]["UTIRIS_NIR"]["calibrated_fellow_eye_plain"]
    ocular.append(r"\midrule")
    for key, label in [("same_eye_same_sensor", "Same eye, same sensor"), ("fellow_eye_same_sensor", "Fellow eye, same sensor")]:
        item = iris[key]
        lo, hi = item["mean_c_ci95_person_bootstrap"]
        ocular.append(f"UTIRIS & {label} & {item['n_people']} & {item['n_pairs']} & {item['mean_c_auc']:.3f} [{lo:.3f}, {hi:.3f}]" + r" \\")
        provenance["ocular_pilots"]["UTIRIS_" + key] = item
    (outputs / "ocular_rows.tex").write_text("\n".join(ocular) + "\n" + r"\bottomrule" + "\n")

    # Representation comparisons use identical recorded pairs within each dataset.
    # Iris-code QC changes the eligible population, so its values are not inserted here.
    representation_rows = []
    provenance["representation_comparisons"] = {}
    specifications = [
        ("GRAPE", data["grape_calibration"]["feature_spaces"],
         [("pixels_16x16", "Pixels"), ("pixels_16x16_fellow_eye_flipped", "Pixels, fellow eye reflected"),
          ("retfound_released", "RETFound")],
         "same_eye_other_visit | same camera", "fellow_eye_same_visit | same camera", "n_patients"),
        ("UPOL", data["iris_calibration"]["UPOL"]["feature_spaces"],
         [("exposure_only_negative_control", "Image mean and SD"), ("pixels_16x16", "Pixels"),
          ("pixels_16x16_zscored", "Standardized pixels")],
         "same_eye_same_sensor", "fellow_eye_same_sensor", "n_people"),
        ("UTIRIS NIR", data["iris_calibration"]["UTIRIS_NIR"]["feature_spaces"],
         [("exposure_only_negative_control", "Image mean and SD"), ("pixels_16x16", "Pixels"),
          ("pixels_16x16_zscored", "Standardized pixels")],
         "same_eye_same_sensor", "fellow_eye_same_sensor", "n_people"),
    ]
    for dataset, spaces, features, repeat_key, bilateral_key, unit_key in specifications:
        expected_counts = None
        for feature, label in features:
            repeat, bilateral = spaces[feature][repeat_key], spaces[feature][bilateral_key]
            counts = (repeat[unit_key], bilateral[unit_key], repeat["n_pairs"], bilateral["n_pairs"])
            if expected_counts is None:
                expected_counts = counts
            assert counts == expected_counts, "Representation comparison has changed eligible pair counts"
            scores = [repeat["mean_c_auc"], bilateral["mean_c_auc"]]
            assert all(np.isfinite(scores)) and all(0 <= value <= 1 for value in scores)
            representation_rows.append(
                f"{dataset} & {label} & {counts[0]}/{counts[1]} & {counts[2]}/{counts[3]} & "
                + " & ".join(f"{value:.3f}" for value in scores) + r" \\")
            provenance["representation_comparisons"][dataset + "/" + feature] = {
                "same_unit": repeat, "fellow_eye": bilateral,
            }
        representation_rows.append(r"\midrule")
    representation_rows[-1] = r"\bottomrule"
    (outputs / "representation_rows.tex").write_text("\n".join(representation_rows) + "\n")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.set(xlim=(0, 10), ylim=(0, 2.5)); ax.axis("off")
    boxes = [(.1, "DEFINE", "Identity unit\n+ nuisance condition"),
             (2.65, "COMPARE", "Repeat / altered capture\n+ identity-negative controls"),
             (5.2, "CALIBRATE", "Matched unrelated pairs\n+ explicit reference coverage"),
             (7.75, "AUDIT", "Discrimination + attenuation\n+ target alignment")]
    for x, title, body in boxes:
        ax.add_patch(FancyBboxPatch((x, .8), 2.12, 1.25, boxstyle="round,pad=0.06", facecolor="#edf3f7", edgecolor="#315670"))
        ax.text(x + 1.06, 1.77, title, ha="center", va="center", weight="bold", color="#1e4058")
        ax.text(x + 1.06, 1.25, body, ha="center", va="center", fontsize=8.2)
        if x < 7:
            ax.annotate("", xy=(x + 2.49, 1.43), xytext=(x + 2.21, 1.43), arrowprops={"arrowstyle": "->", "color": "#315670"})
    ax.text(5, .32, "A high rank against unrelated pairs is evidence about that comparison, not proof of preserved identity.", ha="center", fontsize=9)
    fig.savefig(figures / "audit_overview.pdf", bbox_inches="tight", pad_inches=.05)
    fig.savefig(figures / "audit_overview.png", dpi=180, bbox_inches="tight", pad_inches=.05)
    plt.close(fig)

    # A compact plot of the two failure directions, from the saved pupil experiment.
    pupil = data["pupil_ladder"]
    plot_rows = [("same_eye_repeat", "Repeat"), ("same_eye_dilated", "Same eye, dilated"),
                 ("matched", "Dilated, matched ref."), ("population_mean", "Population mean")]
    fig, ax = plt.subplots(figsize=(4.5, 2.7))
    positions = np.arange(len(plot_rows))
    for index, (metric, label, color) in enumerate([("rmse", "RMSE", "#236a96"), ("ssim", "SSIM", "#bb572b")]):
        values = [pupil["dilation_matched_summary"][metric] if key == "matched" else pupil["summary"][key][metric] for key, _ in plot_rows]
        mean = np.array([v["mean"] for v in values])
        errors = np.array([[v["mean"] - v["min"] for v in values], [v["max"] - v["mean"] for v in values]])
        ax.errorbar(mean, positions + (index - .5) * .15, xerr=errors, fmt="o", capsize=2, label=label, color=color, markersize=4)
    ax.set_yticks(positions, [label for _, label in plot_rows], fontsize=9)
    ax.set_xlim(-.04, 1.04); ax.invert_yaxis()
    ax.axvline(.5, color=".55", ls=":", lw=1)
    ax.set_xlabel("Unrelated-reference rank c (seed ranges)", fontsize=9)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(.3, 1.01), ncol=2, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures / "pupil_scores.pdf", bbox_inches="tight")
    fig.savefig(figures / "pupil_scores.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    for stem in ["visual_tissue_rebuilt", "pupil_fixed_example", "label_fixed_example_folding"]:
        source = ROOT / "results/figures" / (stem + ".pdf")
        shutil.copy2(source, figures / source.name)
        provenance.setdefault("synthetic_figures", {})[source.name] = digest(source)
    (HERE / "evidence.json").write_text(json.dumps(provenance, indent=2) + "\n")
    manuscript = (HERE / "main.tex").read_text()
    cited = {key for group in re.findall(r"\\cite\{([^}]+)\}", manuscript) for key in group.split(",")}
    assert cited == set(KEYS + ["zhou2023retfound"]), "Bibliography and manuscript citations differ"
    assert all(digest(ROOT / key) == value for key, value in provenance["sources"].items())
    print("Prepared three aggregate tables, five figures, references and evidence.json; no experiments run.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch-template", action="store_true", help="Fetch the official CVPR 2027 style files")
    prepare(parser.parse_args().fetch_template)
