# Cross-modality analysis (metrics explorer, 7-8 Nov 2026)

Exploratory, post hoc analysis across brain (QTIM, IXI), faces (CelebA, IMDb-WIKI), fundus (GRAPE, PAPILA) and iris
(UTIRIS NIR, UTIRIS VIS, UPOL). Recorded in `docs/ANALYSIS_cross_modality.md`. Inputs are the private per-target rows
under `C:\Projects\data\projects\identity-ladder-metric\` and the per-modality result JSONs; every output in
`results/cross_modality/` is an aggregate (no participant IDs, filenames or image-level rows).

Run order (`OUT=results/cross_modality`; Python with `-I`):

| Step | Script | Output |
|---|---|---|
| 1 | `explore_corr.py OUT/corr_all.json` | per-target Spearman between metrics (reproduces the brain correlation file) |
| 2 | `common_frame.py OUT/common_all.json` | one R_self function and raw per-rung means for every dataset, bootstrap over units |
| 3 | `tests_uniform.py OUT/tests_all.json` | T1-T3b with one rule for every dataset, Holm within dataset |
| 4 | `cal_uniform.py OUT/cal_faces.json CelebA IMDb-WIKI` and `... OUT/cal_brain.json QTIM` | z, EVT, posterior, c (fundus calibrate() ported; `GRAPE` as a check) |
| 5 | `drawn_rungs.py OUT` | `drawn_face_thumbs.npz`: rung pictures on a drawn face (no real person), uses `src/ladder_faces.py` |
| 6 | `rank_tiles.py OUT` and `rank_tiles.py OUT dataset` | rank tiles by metric and by dataset, `ranks_all.json` |
| 7 | `cal_tiles.py OUT` | calibrated score c tiles, `cal_c_all.json` |
| 8 | `agreement.py OUT` | dataset x dataset and metric x metric agreement of rung order |
| 9 | `dataset_card.py OUT <bilateral_worktree>` | `dataset_card.json` / `.png`: persons, images, rungs, V, ladder ratios |
| 10 | `generative_block.py OUT <bilateral_worktree>` | LPIPS, Betti matching, persistence, iris code, FID, FWD ranks, values (`values_generative.json`) and tiles (fundus 9.15, iris P3, faces F2, brain Amendment 20; missing parts "not run") |
| 11 | `agreement.py OUT` again | agreement including the generative block |
| 11b | `summary_figure.py OUT OUT/fig_metric_summary` | two-panel summary figure (rank gaps; R on the distance scale, FID/FWD log) and its values `_R.json` |
| 12 | `build_data.py OUT`, then `build_explorer.py OUT page.html` | explorer page (published privately as a claude.ai artifact) |

Validation recorded when written: R_self equals the fundus and iris files exactly and the face file within 0.01;
T1-T3b p values equal the brain, GRAPE, PAPILA and iris files exactly; calibrations equal GRAPE (z, EVT, posterior
to machine precision, c within 5e-5) and the brain file (z, EVT, posterior). Brain, fundus and iris are pre-QC.
