# Artifact and evidence inventory

Checked 8 November 2026. Counts describe files present in this checkout, not independent experiments or successful reruns. Current source/result hashes are in `artifact_inventory.json`. Historical test counts are not current verification.

## Present on disk

There are **24 result JSON files**, **7 CSV tables**, **3 result reports**, **20 PNG figures**, **17 PDF figures**, **29 source Python files**, and **17 Python test files**. PNG/PDF pairs usually represent the same figure and are not separate findings. The combined `all_ladder_metrics.csv` has 1116 data rows; this is not a sample size.

## Evidence suitable for expanding the current manuscript

| Block | Saved artifacts, relative to project root | Evidence and limits |
|---|---|---|
| Controlled mean collapse | `results/population_null_synthetic.json`, `population_null_summary.csv`, `population_null_report.md`; `src/population_null.py` | 24 synthetic tissue families and 24 synthetic eye identities; shrinkage, repeats and incorrect-detail controls; three seeds. The path uses the true target and is not an inpainting model. |
| Pupil dilation | `results/pupil_ladder.json`, `pupil_ladder_summary.csv`, `pupil_report.md`; `src/pupil_ladder.py` | 96 synthetic targets across three seeds; same latent eye under dilation, matched references, pupil-mask measurements. No measured human pupil physiology. |
| Labels and anatomy prior | `results/label_robustness.json`, `label_robustness_summary.csv`, `label_paired_twin_summary.csv`; `src/label_robustness.py` | Five seeds, 300 simulated families per condition, two variance settings and three priors; 3,600 dependent target-condition evaluations. Threshold segmentation; conditional Dice findings, not a general theorem. |
| Fundus representations | `results/grape_calibration.json`; `src/grape_calibration.py` | 631 GRAPE images; RETFound, 16-by-16 pixels and reflected fellow-eye pixels; exact camera-pair matching. Visit/time/camera/progression remain intertwined. |
| Iris capture controls | `results/iris_calibration.json`; `src/iris_calibration.py` | UPOL and UTIRIS; pixels, intensity-standardized pixels and image mean/SD controls. Strong scores can also arise in simple photometric summaries. |
| Iris texture code | `results/iris_codes.json`; `src/iris_codes.py` | UTIRIS NIR is primary; same-eye and fellow-eye scores with QC. UPOL and VIS are secondary due to segmentation coverage. Different QC subsets prevent an unpaired head-to-head ranking with pixel baselines. |
| Official-package metric checks | `results/brats_metrics_table.csv`, `brats_panel_checks_tm1.9.0.json`; `src/brats_panel_checks.py` | Nine package outputs, rank equivalences and implementation behavior. Two-dimensional diagnostic adaptation, not official challenge evaluation. |

## MRI and prepared visual artifacts

`results/image_ladder.json` records a processed pilot with 95 SIMON, 89 IXI and 34 QTIM scans. SIMON has one participant. The saved file records 12 registration exclusions and two extraction failures. `results/audit_2026-11-01.json` documents why protocol and centre effects, QC and dependent pairs limit interpretation. These records establish prior experimental work, not a clean cross-anatomy identity ordering. Old audit prose includes values from before fixes; current aggregate files take precedence.

`results/real_atlas_mri.json` records **6 MRI targets from 6 families**; `results/real_atlas_iris.json` records **6 iris targets**. These files retain aggregate atlas results and selection/limitation records. Their participant photographs, scans, per-case ledgers and rendered real-image atlases were stored outside this repository. Current availability and visual quality of that private material were not verified here.

`results/real_detail_methods.json` documents one fixed MRI case and one fixed iris case, 18 prepared MRI volumes, a 32-by-32 MRI patch, disjoint reference/template pools and the limitations of affine correspondence. This is demonstration-level support, not cohort validation. `src/real_case_atlas.py`, `real_mri_detail.py`, `rebuild_visuals.py` and `detail_report.py` contain the preparation/reporting code.

`results/synthetic_atlas_eyes.json` records 12 synthetic eye examples; `synthetic_atlas_tissue.json` records six synthetic tissue conditions from two families. The public `results/figures/` directory contains synthetic example cards, ladders, all-metric plots, pupil illustrations, and label/topology comparisons. `population_mean_four_modalities.png`, referenced by historical reporting code, is a private montage of synthetic and real tissue/iris, not a brain/fundus/face/pupil atlas.

## Manuscripts and planning artifacts

| Location | Role |
|---|---|
| `paper/isbi2027/overleaf/main.tex`, `refs.bib`; `paper/isbi2027/main.pdf` | ISBI source, bibliography and existing PDF; preserved during this revision |
| `paper/cvpr2027/main.tex`, `main.pdf` | English CVPR development manuscript and rebuilt reading copy |
| `paper/cvpr2027/prepare.py`, `generated/`, `evidence.json` | Aggregate-to-manuscript table/figure generation and numeric provenance |
| `paper/cvpr2027/research_brief.md` | Latest cross-anatomy scope and requested difference-atlas specification |
| `paper/cvpr2027/outline.md`, `development.md` | Earlier outline, expanded scope, open experiments and overlap analysis |
| `docs/DATASETS.md`, `DECISION_TREES.md`, `PROTOCOL_real_cases.md` | Dataset catalogue, methodological choices and historical frozen real-case protocol |
| `paper/cvpr2027/citation_report.md`, `verification.md` | Bibliography-check record and manuscript validation history |

## Missing evidence for the expanded paper

There is no completed brain/eye/fundus/face/pupil spatial difference atlas in this checkout. Face datasets are catalogue leads, without saved face-comparison results. Real pupil physiology remains unmeasured. LPIPS, DreamSim and DINO comparisons are planned, not saved benchmark outcomes. No held-out comparison of actual modern inpainting models is established by these artifacts. The full-data human-image pipeline still contains historical Windows paths, so the existence of source files alone does not establish reproducibility in WSL.

The next empirical contribution is the measured atlas and a common held-out pair/metric protocol, followed by reconstruction-output evaluation. Existing ISBI analyses remain shared evidence and should not be relabeled as new CVPR experiments.
