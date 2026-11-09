# Materials coverage map

This inventory maps available tracked materials to text roles. It is an editorial coverage record, not a claim that every historical artifact is a current result. PNG/PDF pairs are duplicate renderings, and JSON/CSV summaries may describe the same run.

| Material family | Files | Text role |
|---|---:|---|
| Brain donor ladder | 15 | Main brain results and supplementary maps/sensitivity; status per source |
| CARE reproduction | 16 | Related work and supplementary aggregation audit; not an image-level reconstruction rerun |
| Cross-modality exploration | 18 | Supplementary exploratory context; no universal fused identity score |
| Earlier pilots | 4 | Historical context; acquisition/protocol confounding prevents promotion to current primary results |
| Face donor ladder | 54 | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| Fundus and earlier 2D | 20 | Main P8/P9 frozen aggregate; P7 and unregistered pilots retained as historical evidence |
| Iris | 23 | Manuscript frozen P2 evidence; newer P3/window exports are available for separate integration review |
| Mechanism controls | 57 | Supplementary synthetic controls; toy evidence, no biological validation |
| Metric board | 3 | Operational generated view; not an additional experiment or independent result |
| Metric comparison | 9 | Main profiles and supplementary dependence; do not count monotone variants as independent evidence |
| Official evaluator probes | 2 | Main implementation audit and supplement; synthetic code behavior, no model ranking or organizer-server validation |
| Real-case demonstrations | 11 | Supplementary preparation history; fixed examples are not cohort validation |
| Registration | 24 | Methods, source-version interpretation and limitations; unresolved gate retained |
| Related ISBI | 8 | Related-work/overlap review; source preserved, not copied as a new contribution |
| Supporting material | 62 | Methods/provenance/reference context; review separately before making an empirical claim |
| Withdrawn MRI results | 2 | Excluded: source explicitly invalidates c/rho/label results after the intensity-scaling failure; not provisional support |

## Frozen other-branch inputs

The bilateral snapshot 82ed7fe supplies results/donor_ladder_2d_facefmt.json, results/fundus_ladder_metrics.json, the published aggregate atlas and its generator. Original JSON bytes are retained in inputs/ and source hashes/commits in evidence_manifest.json and atlas_provenance.json. These are frozen Git artifacts, not the currently running fundus extension.

## Material not available in this host snapshot

The base face pipeline/protocol has now been recovered, and full iris P3 plus newer brain/face/IXI exports are present after the fork update. The manuscript still uses its explicitly pinned a48523d evidence, pending scientific integration review. Private animation and the latest fundus extension remain outside this draft. An available export is not automatically reviewed evidence.

## Artifact-by-artifact placement

| Artifact | Family | Placement / exclusion reason |
|---|---|---|
| `docs/ANALYSIS_cross_modality.md` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `docs/DATASETS.md` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `docs/DECISION_TREES.md` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `docs/Experiment_Plan_Available_Data.md` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `docs/FACE_ID_LESSONS.md` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `docs/HANDOFF.md` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `docs/HANDOFF_face_metrics.md` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `docs/HANDOFF_iris_metrics.md` | Iris | Manuscript frozen P2 evidence; newer P3/window exports are available for separate integration review |
| `docs/LITERATURE_closest_work_2026-11-02.md` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `docs/OPEN_REPOSITORY_RELEASE_PLAN.md` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `docs/OPEN_REPOSITORY_REVIEW.md` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `docs/PROTOCOL_2d_ladder.md` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `docs/PROTOCOL_care_reproduction.md` | CARE reproduction | Related work and supplementary aggregation audit; not an image-level reconstruction rerun |
| `docs/PROTOCOL_care_reproduction_amendment1.md` | CARE reproduction | Related work and supplementary aggregation audit; not an image-level reconstruction rerun |
| `docs/PROTOCOL_care_reproduction_amendment2.md` | CARE reproduction | Related work and supplementary aggregation audit; not an image-level reconstruction rerun |
| `docs/PROTOCOL_donor_ladder_brain.md` | Brain donor ladder | Main brain results and supplementary maps/sensitivity; status per source |
| `docs/PROTOCOL_donor_ladder_face.md` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `docs/PROTOCOL_donor_ladder_fundus.md` | Fundus and earlier 2D | Main P8/P9 frozen aggregate; P7 and unregistered pilots retained as historical evidence |
| `docs/PROTOCOL_donor_ladder_general.md` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `docs/PROTOCOL_donor_ladder_robustness.md` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `docs/PROTOCOL_face_metrics.md` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `docs/PROTOCOL_iris_metrics.md` | Iris | Manuscript frozen P2 evidence; newer P3/window exports are available for separate integration review |
| `docs/PROTOCOL_real_cases.md` | Real-case demonstrations | Supplementary preparation history; fixed examples are not cohort validation |
| `docs/PROTOCOL_registration.md` | Registration | Methods, source-version interpretation and limitations; unresolved gate retained |
| `docs/REGISTRATION_QC_HANDOFF.md` | Registration | Methods, source-version interpretation and limitations; unresolved gate retained |
| `docs/REPRODUCTION_BACKLOG.md` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `docs/REPRODUCTION_WORKFLOW.md` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `docs/REVIEW_care_reproduction.md` | CARE reproduction | Related work and supplementary aggregation audit; not an image-level reconstruction rerun |
| `docs/Research_Plan_Identity_Ladder_Metric.md` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `docs/audits/open_repository_claims_20261109.json` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `docs/audits/open_repository_findings_20261109.csv` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `docs/audits/open_repository_inventory_20261109.json` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `docs/audits/release_readiness_20261109.json` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `docs/figures_draft.md` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `paper/isbi2027/README.md` | Related ISBI | Related-work/overlap review; source preserved, not copied as a new contribution |
| `paper/isbi2027/overleaf/fig_cases.pdf` | Related ISBI | Related-work/overlap review; source preserved, not copied as a new contribution |
| `paper/isbi2027/overleaf/fig_ladder.pdf` | Related ISBI | Related-work/overlap review; source preserved, not copied as a new contribution |
| `paper/isbi2027/overleaf/fig_ladder.png` | Related ISBI | Related-work/overlap review; source preserved, not copied as a new contribution |
| `paper/isbi2027/overleaf/fig_real_case.pdf` | Related ISBI | Related-work/overlap review; source preserved, not copied as a new contribution |
| `paper/isbi2027/overleaf/fig_real_theory.pdf` | Related ISBI | Related-work/overlap review; source preserved, not copied as a new contribution |
| `paper/isbi2027/overleaf/main.tex` | Related ISBI | Related-work/overlap review; source preserved, not copied as a new contribution |
| `paper/isbi2027/overleaf/refs.bib` | Related ISBI | Related-work/overlap review; source preserved, not copied as a new contribution |
| `results/METRICS_STATUS.csv` | Metric board | Operational generated view; not an additional experiment or independent result |
| `results/METRICS_STATUS.json` | Metric board | Operational generated view; not an additional experiment or independent result |
| `results/METRICS_STATUS.md` | Metric board | Operational generated view; not an additional experiment or independent result |
| `results/all_ladder_metrics.csv` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/audit_2026-10-23.json` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/audit_2026-11-01.json` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/baseline_v0.json` | Earlier pilots | Historical context; acquisition/protocol confounding prevents promotion to current primary results |
| `results/brats_metrics_table.csv` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/brats_panel_checks_tm1.1.2.json` | Official evaluator probes | Main implementation audit and supplement; synthetic code behavior, no model ranking or organizer-server validation |
| `results/brats_panel_checks_tm1.9.0.json` | Official evaluator probes | Main implementation audit and supplement; synthetic code behavior, no model ranking or organizer-server validation |
| `results/care_iteration2.json` | CARE reproduction | Related work and supplementary aggregation audit; not an image-level reconstruction rerun |
| `results/care_iteration2_inputs.json` | CARE reproduction | Related work and supplementary aggregation audit; not an image-level reconstruction rerun |
| `results/care_iteration2_report.md` | CARE reproduction | Related work and supplementary aggregation audit; not an image-level reconstruction rerun |
| `results/care_iteration2_state.json` | CARE reproduction | Related work and supplementary aggregation audit; not an image-level reconstruction rerun |
| `results/care_iteration2_table.csv` | CARE reproduction | Related work and supplementary aggregation audit; not an image-level reconstruction rerun |
| `results/care_reproduction.json` | CARE reproduction | Related work and supplementary aggregation audit; not an image-level reconstruction rerun |
| `results/care_reproduction_all_phases_posthoc.json` | CARE reproduction | Related work and supplementary aggregation audit; not an image-level reconstruction rerun |
| `results/care_reproduction_all_phases_posthoc_report.md` | CARE reproduction | Related work and supplementary aggregation audit; not an image-level reconstruction rerun |
| `results/care_reproduction_amendment1_inputs.json` | CARE reproduction | Related work and supplementary aggregation audit; not an image-level reconstruction rerun |
| `results/care_reproduction_inputs.json` | CARE reproduction | Related work and supplementary aggregation audit; not an image-level reconstruction rerun |
| `results/care_reproduction_report.md` | CARE reproduction | Related work and supplementary aggregation audit; not an image-level reconstruction rerun |
| `results/care_reproduction_table.csv` | CARE reproduction | Related work and supplementary aggregation audit; not an image-level reconstruction rerun |
| `results/code_provenance.json` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/cross_modality/agreement.json` | Cross-modality exploration | Supplementary exploratory context; no universal fused identity score |
| `results/cross_modality/agreement.png` | Cross-modality exploration | Supplementary exploratory context; no universal fused identity score |
| `results/cross_modality/cal_brain.json` | Cross-modality exploration | Supplementary exploratory context; no universal fused identity score |
| `results/cross_modality/cal_c_all.json` | Cross-modality exploration | Supplementary exploratory context; no universal fused identity score |
| `results/cross_modality/cal_faces.json` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/cross_modality/cal_tiles.png` | Cross-modality exploration | Supplementary exploratory context; no universal fused identity score |
| `results/cross_modality/common_all.json` | Cross-modality exploration | Supplementary exploratory context; no universal fused identity score |
| `results/cross_modality/corr_all.json` | Cross-modality exploration | Supplementary exploratory context; no universal fused identity score |
| `results/cross_modality/data.json` | Cross-modality exploration | Supplementary exploratory context; no universal fused identity score |
| `results/cross_modality/dataset_card.json` | Cross-modality exploration | Supplementary exploratory context; no universal fused identity score |
| `results/cross_modality/dataset_card.png` | Cross-modality exploration | Supplementary exploratory context; no universal fused identity score |
| `results/cross_modality/generative_status.json` | Cross-modality exploration | Supplementary exploratory context; no universal fused identity score |
| `results/cross_modality/generative_tiles.png` | Cross-modality exploration | Supplementary exploratory context; no universal fused identity score |
| `results/cross_modality/rank_tiles.png` | Cross-modality exploration | Supplementary exploratory context; no universal fused identity score |
| `results/cross_modality/rank_tiles_by_dataset.png` | Cross-modality exploration | Supplementary exploratory context; no universal fused identity score |
| `results/cross_modality/ranks_all.json` | Cross-modality exploration | Supplementary exploratory context; no universal fused identity score |
| `results/cross_modality/ranks_generative.json` | Cross-modality exploration | Supplementary exploratory context; no universal fused identity score |
| `results/cross_modality/tests_all.json` | Cross-modality exploration | Supplementary exploratory context; no universal fused identity score |
| `results/cross_modality/values_generative.json` | Cross-modality exploration | Supplementary exploratory context; no universal fused identity score |
| `results/donor_ladder_2d.json` | Fundus and earlier 2D | Main P8/P9 frozen aggregate; P7 and unregistered pilots retained as historical evidence |
| `results/donor_ladder_2d_p7.json` | Fundus and earlier 2D | Main P8/P9 frozen aggregate; P7 and unregistered pilots retained as historical evidence |
| `results/donor_ladder_brain.json` | Brain donor ladder | Main brain results and supplementary maps/sensitivity; status per source |
| `results/donor_ladder_brain_bilateral.json` | Brain donor ladder | Main brain results and supplementary maps/sensitivity; status per source |
| `results/donor_ladder_brain_haca3_stage1.json` | Brain donor ladder | Main brain results and supplementary maps/sensitivity; status per source |
| `results/donor_ladder_brain_ixi.json` | Brain donor ladder | Main brain results and supplementary maps/sensitivity; status per source |
| `results/donor_ladder_brain_map_grid.json` | Brain donor ladder | Main brain results and supplementary maps/sensitivity; status per source |
| `results/donor_ladder_brain_maps.json` | Brain donor ladder | Main brain results and supplementary maps/sensitivity; status per source |
| `results/donor_ladder_brain_metrics.json` | Brain donor ladder | Main brain results and supplementary maps/sensitivity; status per source |
| `results/donor_ladder_brain_metrics_size.json` | Brain donor ladder | Main brain results and supplementary maps/sensitivity; status per source |
| `results/donor_ladder_brain_metrics_t23.json` | Brain donor ladder | Main brain results and supplementary maps/sensitivity; status per source |
| `results/donor_ladder_brain_metrics_t23_v5.json` | Brain donor ladder | Main brain results and supplementary maps/sensitivity; status per source |
| `results/donor_ladder_brain_seg_gcp_run.json` | Brain donor ladder | Main brain results and supplementary maps/sensitivity; status per source |
| `results/donor_ladder_brain_size.json` | Brain donor ladder | Main brain results and supplementary maps/sensitivity; status per source |
| `results/donor_ladder_face.json` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/donor_ladder_face_v2.json` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/donor_ladder_face_v3.json` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/donor_ladder_face_v4.json` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/donor_ladder_face_v5.json` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/donor_ladder_face_v5_prefix.json` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/donor_ladder_face_v6.json` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/donor_ladder_face_v6f.json` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/donor_ladder_fundus.json` | Fundus and earlier 2D | Main P8/P9 frozen aggregate; P7 and unregistered pilots retained as historical evidence |
| `results/donor_ladder_metrics_correlation.json` | Metric comparison | Main profiles and supplementary dependence; do not count monotone variants as independent evidence |
| `results/face_flow_v5.json` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/face_flow_v6.json` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/face_flow_v6f.json` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/face_ladder_metrics.json` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/face_ladder_sizes.json` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/face_masks_v6.json` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/face_masks_v6f.json` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/face_masks_v7.json` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/figures/all_brats_metrics_base.pdf` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/figures/all_brats_metrics_base.png` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/figures/all_brats_metrics_low_h.pdf` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/figures/all_brats_metrics_low_h.png` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/figures/fig_brain_metrics_t23.pdf` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/figures/fig_brain_metrics_t23.png` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/figures/fig_brain_window_sizes.pdf` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/figures/fig_brain_window_sizes.png` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/figures/fig_donor_ladder_fundus.pdf` | Fundus and earlier 2D | Main P8/P9 frozen aggregate; P7 and unregistered pilots retained as historical evidence |
| `results/figures/fig_donor_ladder_fundus.png` | Fundus and earlier 2D | Main P8/P9 frozen aggregate; P7 and unregistered pilots retained as historical evidence |
| `results/figures/fig_donor_ladder_overlay.pdf` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/figures/fig_donor_ladder_overlay.png` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/figures/fig_face_flow_v5.pdf` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/figures/fig_face_flow_v5.png` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/figures/fig_face_flow_v6.pdf` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/figures/fig_face_flow_v6.png` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/figures/fig_face_flow_v6f.pdf` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/figures/fig_face_flow_v6f.png` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/figures/fig_face_ladder_maps.pdf` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/figures/fig_face_ladder_maps.png` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/figures/fig_face_ladder_maps_v2.pdf` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/figures/fig_face_ladder_maps_v2.png` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/figures/fig_face_ladder_maps_v3.pdf` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/figures/fig_face_ladder_maps_v3.png` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/figures/fig_face_ladder_maps_v4.pdf` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/figures/fig_face_ladder_maps_v4.png` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/figures/fig_face_ladder_maps_v5.pdf` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/figures/fig_face_ladder_maps_v5.png` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/figures/fig_face_ladder_maps_v6.pdf` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/figures/fig_face_ladder_maps_v6.png` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/figures/fig_face_ladder_maps_v6f.pdf` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/figures/fig_face_ladder_maps_v6f.png` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/figures/fig_face_metrics_blocks.pdf` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/figures/fig_face_metrics_blocks.png` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/figures/fig_face_metrics_blocks_full.pdf` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/figures/fig_face_metrics_blocks_full.png` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/figures/fig_face_metrics_expected.pdf` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/figures/fig_face_metrics_expected.png` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/figures/fig_face_metrics_shrink_table.pdf` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/figures/fig_face_metrics_shrink_table.png` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/figures/fig_face_window_sizes.pdf` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/figures/fig_face_window_sizes.png` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/figures/fig_iris_donor_ladder.pdf` | Iris | Manuscript frozen P2 evidence; newer P3/window exports are available for separate integration review |
| `results/figures/fig_iris_donor_ladder.png` | Iris | Manuscript frozen P2 evidence; newer P3/window exports are available for separate integration review |
| `results/figures/fig_iris_maps_common.pdf` | Iris | Manuscript frozen P2 evidence; newer P3/window exports are available for separate integration review |
| `results/figures/fig_iris_maps_common.png` | Iris | Manuscript frozen P2 evidence; newer P3/window exports are available for separate integration review |
| `results/figures/fig_iris_metrics_blocks.pdf` | Iris | Manuscript frozen P2 evidence; newer P3/window exports are available for separate integration review |
| `results/figures/fig_iris_metrics_blocks.png` | Iris | Manuscript frozen P2 evidence; newer P3/window exports are available for separate integration review |
| `results/figures/fig_iris_metrics_blocks_full.pdf` | Iris | Manuscript frozen P2 evidence; newer P3/window exports are available for separate integration review |
| `results/figures/fig_iris_metrics_blocks_full.png` | Iris | Manuscript frozen P2 evidence; newer P3/window exports are available for separate integration review |
| `results/figures/fig_iris_metrics_ratio.pdf` | Iris | Manuscript frozen P2 evidence; newer P3/window exports are available for separate integration review |
| `results/figures/fig_iris_metrics_ratio.png` | Iris | Manuscript frozen P2 evidence; newer P3/window exports are available for separate integration review |
| `results/figures/fig_iris_p3_blocks.pdf` | Iris | Manuscript frozen P2 evidence; newer P3/window exports are available for separate integration review |
| `results/figures/fig_iris_p3_blocks.png` | Iris | Manuscript frozen P2 evidence; newer P3/window exports are available for separate integration review |
| `results/figures/fig_iris_p3_setlevel.pdf` | Iris | Manuscript frozen P2 evidence; newer P3/window exports are available for separate integration review |
| `results/figures/fig_iris_p3_setlevel.png` | Iris | Manuscript frozen P2 evidence; newer P3/window exports are available for separate integration review |
| `results/figures/fig_iris_window_f8.pdf` | Iris | Manuscript frozen P2 evidence; newer P3/window exports are available for separate integration review |
| `results/figures/fig_iris_window_f8.png` | Iris | Manuscript frozen P2 evidence; newer P3/window exports are available for separate integration review |
| `results/figures/fig_ladder.pdf` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/figures/fig_ladder.png` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/figures/fig_ladder2d_maps_p7_common.pdf` | Fundus and earlier 2D | Main P8/P9 frozen aggregate; P7 and unregistered pilots retained as historical evidence |
| `results/figures/fig_ladder2d_maps_p7_common.png` | Fundus and earlier 2D | Main P8/P9 frozen aggregate; P7 and unregistered pilots retained as historical evidence |
| `results/figures/fig_ladder2d_maps_p7_contrast.pdf` | Fundus and earlier 2D | Main P8/P9 frozen aggregate; P7 and unregistered pilots retained as historical evidence |
| `results/figures/fig_ladder2d_maps_p7_contrast.png` | Fundus and earlier 2D | Main P8/P9 frozen aggregate; P7 and unregistered pilots retained as historical evidence |
| `results/figures/fig_ladder2d_sensitivity.pdf` | Fundus and earlier 2D | Main P8/P9 frozen aggregate; P7 and unregistered pilots retained as historical evidence |
| `results/figures/fig_ladder2d_sensitivity.png` | Fundus and earlier 2D | Main P8/P9 frozen aggregate; P7 and unregistered pilots retained as historical evidence |
| `results/figures/fig_ladder_maps_common.pdf` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/figures/fig_ladder_maps_common.png` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/figures/fig_ladder_maps_contrast.pdf` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/figures/fig_ladder_maps_contrast.png` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/figures/fig_ladder_maps_core.pdf` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/figures/fig_ladder_maps_core.png` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/figures/fig_metrics_blocks.pdf` | Metric comparison | Main profiles and supplementary dependence; do not count monotone variants as independent evidence |
| `results/figures/fig_metrics_blocks.png` | Metric comparison | Main profiles and supplementary dependence; do not count monotone variants as independent evidence |
| `results/figures/fig_metrics_blocks_full.pdf` | Metric comparison | Main profiles and supplementary dependence; do not count monotone variants as independent evidence |
| `results/figures/fig_metrics_blocks_full.png` | Metric comparison | Main profiles and supplementary dependence; do not count monotone variants as independent evidence |
| `results/figures/fig_metrics_correlation.pdf` | Metric comparison | Main profiles and supplementary dependence; do not count monotone variants as independent evidence |
| `results/figures/fig_metrics_correlation.png` | Metric comparison | Main profiles and supplementary dependence; do not count monotone variants as independent evidence |
| `results/figures/fig_metrics_expected.pdf` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/figures/fig_metrics_expected.png` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/figures/fig_metrics_expected_full.pdf` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/figures/fig_metrics_expected_full.png` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/figures/fig_metrics_ladder.pdf` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/figures/fig_metrics_ladder.png` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/figures/fig_metrics_polar.pdf` | Metric comparison | Main profiles and supplementary dependence; do not count monotone variants as independent evidence |
| `results/figures/fig_metrics_polar.png` | Metric comparison | Main profiles and supplementary dependence; do not count monotone variants as independent evidence |
| `results/figures/fig_metrics_shrink_table.pdf` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/figures/fig_metrics_shrink_table.png` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/figures/fig_real_case_parietal.png` | Real-case demonstrations | Supplementary preparation history; fixed examples are not cohort validation |
| `results/figures/fig_real_case_parietal_v2.pdf` | Real-case demonstrations | Supplementary preparation history; fixed examples are not cohort validation |
| `results/figures/fig_real_case_parietal_v2.png` | Real-case demonstrations | Supplementary preparation history; fixed examples are not cohort validation |
| `results/figures/fig_real_case_theory.png` | Real-case demonstrations | Supplementary preparation history; fixed examples are not cohort validation |
| `results/figures/fig_real_theory_v2_parietal.pdf` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/figures/fig_real_theory_v2_parietal.png` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/figures/fig_real_theory_v2_random_patches.pdf` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/figures/fig_real_theory_v2_random_patches.png` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/figures/fig_size_shell_dependence.pdf` | Brain donor ladder | Main brain results and supplementary maps/sensitivity; status per source |
| `results/figures/fig_size_shell_dependence.png` | Brain donor ladder | Main brain results and supplementary maps/sensitivity; status per source |
| `results/figures/fig_variability_maps.pdf` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/figures/fig_variability_maps.png` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/figures/identity_ladder.pdf` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/figures/identity_ladder.png` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/figures/identity_ladder_faces.pdf` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/figures/identity_ladder_faces.png` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/figures/identity_ladder_faces_photo.png` | Face donor ladder | Main face panel; historical/synthetic drawings distinguished from measured v6f |
| `results/figures/label_fixed_example.pdf` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/label_fixed_example.png` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/label_fixed_example_folding.pdf` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/label_fixed_example_folding.png` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/label_hemisphere_locations.pdf` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/label_hemisphere_locations.png` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/label_paired_dz_all_metrics.pdf` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/label_paired_dz_all_metrics.png` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/label_paired_mz_all_metrics.pdf` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/label_paired_mz_all_metrics.png` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/label_robustness_all_metrics.pdf` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/label_robustness_all_metrics.png` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/mixing_eyes.pdf` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/mixing_eyes.png` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/mixing_tissue.pdf` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/mixing_tissue.png` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/population_null_all_metrics.png` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/population_null_shape_metrics.png` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/pupil_all_metrics.pdf` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/pupil_all_metrics.png` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/pupil_fixed_example.pdf` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/pupil_fixed_example.png` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/synthetic_atlas/eye_01.png` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/synthetic_atlas/eye_02.png` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/synthetic_atlas/eye_03.png` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/synthetic_atlas/eye_04.png` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/synthetic_atlas/eye_05.png` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/synthetic_atlas/eye_06.png` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/synthetic_atlas/eye_07.png` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/synthetic_atlas/eye_08.png` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/synthetic_atlas/eye_09.png` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/synthetic_atlas/eye_10.png` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/synthetic_atlas/eye_11.png` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/synthetic_atlas/eye_12.png` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/synthetic_atlas/eyes_gallery_1.png` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/synthetic_atlas/eyes_gallery_2.png` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/synthetic_atlas/tissue_01.png` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/synthetic_atlas/tissue_02.png` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/synthetic_atlas/tissue_03.png` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/synthetic_atlas/tissue_04.png` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/synthetic_atlas/tissue_05.png` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/synthetic_atlas/tissue_06.png` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/synthetic_atlas/tissue_gallery.png` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/figures/visual_eyes_rebuilt.pdf` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/figures/visual_eyes_rebuilt.png` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/figures/visual_tissue_rebuilt.pdf` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/figures/visual_tissue_rebuilt.png` | Supporting material | Methods/provenance/reference context; review separately before making an empirical claim |
| `results/grape_calibration.json` | Fundus and earlier 2D | Main P8/P9 frozen aggregate; P7 and unregistered pilots retained as historical evidence |
| `results/grape_mirror_interval.json` | Fundus and earlier 2D | Main P8/P9 frozen aggregate; P7 and unregistered pilots retained as historical evidence |
| `results/grape_repro.json` | Fundus and earlier 2D | Main P8/P9 frozen aggregate; P7 and unregistered pilots retained as historical evidence |
| `results/image_ladder.json` | Earlier pilots | Historical context; acquisition/protocol confounding prevents promotion to current primary results |
| `results/iris_calibration.json` | Iris | Manuscript frozen P2 evidence; newer P3/window exports are available for separate integration review |
| `results/iris_codes.json` | Iris | Manuscript frozen P2 evidence; newer P3/window exports are available for separate integration review |
| `results/iris_ladder_metrics.json` | Iris | Manuscript frozen P2 evidence; newer P3/window exports are available for separate integration review |
| `results/iris_ladder_metrics_w10.json` | Iris | Manuscript frozen P2 evidence; newer P3/window exports are available for separate integration review |
| `results/label_ladder.json` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/label_paired_twin_summary.csv` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/label_robustness.json` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/label_robustness_report.md` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/label_robustness_summary.csv` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/ladder2d.json` | Fundus and earlier 2D | Main P8/P9 frozen aggregate; P7 and unregistered pilots retained as historical evidence |
| `results/ladder2d_brats_baseline.csv` | Fundus and earlier 2D | Main P8/P9 frozen aggregate; P7 and unregistered pilots retained as historical evidence |
| `results/ladder2d_brats_baseline.json` | Fundus and earlier 2D | Main P8/P9 frozen aggregate; P7 and unregistered pilots retained as historical evidence |
| `results/ladder2d_brats_baseline.md` | Fundus and earlier 2D | Main P8/P9 frozen aggregate; P7 and unregistered pilots retained as historical evidence |
| `results/ladder2d_manifest.json` | Fundus and earlier 2D | Main P8/P9 frozen aggregate; P7 and unregistered pilots retained as historical evidence |
| `results/population_null_report.md` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/population_null_summary.csv` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/population_null_synthetic.json` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/pupil_ladder.json` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/pupil_ladder_summary.csv` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/pupil_report.md` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/real_atlas_iris.json` | Iris | Manuscript frozen P2 evidence; newer P3/window exports are available for separate integration review |
| `results/real_atlas_mri.json` | Real-case demonstrations | Supplementary preparation history; fixed examples are not cohort validation |
| `results/real_cases.json` | Withdrawn MRI results | Excluded: source explicitly invalidates c/rho/label results after the intensity-scaling failure; not provisional support |
| `results/real_cases_data_qc_summary.json` | Real-case demonstrations | Supplementary preparation history; fixed examples are not cohort validation |
| `results/real_cases_manifest.json` | Real-case demonstrations | Supplementary preparation history; fixed examples are not cohort validation |
| `results/real_cases_manifest_parietal.json` | Real-case demonstrations | Supplementary preparation history; fixed examples are not cohort validation |
| `results/real_cases_parietal.json` | Withdrawn MRI results | Excluded: source explicitly invalidates c/rho/label results after the intensity-scaling failure; not provisional support |
| `results/real_detail_methods.json` | Real-case demonstrations | Supplementary preparation history; fixed examples are not cohort validation |
| `results/registration/GRAPE_G0.json` | Registration | Methods, source-version interpretation and limitations; unresolved gate retained |
| `results/registration/GRAPE_G1.json` | Registration | Methods, source-version interpretation and limitations; unresolved gate retained |
| `results/registration/GRAPE_G2.json` | Registration | Methods, source-version interpretation and limitations; unresolved gate retained |
| `results/registration/GRAPE_H1.json` | Registration | Methods, source-version interpretation and limitations; unresolved gate retained |
| `results/registration/GRAPE_H2.json` | Registration | Methods, source-version interpretation and limitations; unresolved gate retained |
| `results/registration/UPOL_H1.json` | Registration | Methods, source-version interpretation and limitations; unresolved gate retained |
| `results/registration/UPOL_I0.json` | Registration | Methods, source-version interpretation and limitations; unresolved gate retained |
| `results/registration/UPOL_I1.json` | Registration | Methods, source-version interpretation and limitations; unresolved gate retained |
| `results/registration/UPOL_I2.json` | Registration | Methods, source-version interpretation and limitations; unresolved gate retained |
| `results/registration/UPOL_I3.json` | Registration | Methods, source-version interpretation and limitations; unresolved gate retained |
| `results/registration/UPOL_I4.json` | Registration | Methods, source-version interpretation and limitations; unresolved gate retained |
| `results/registration/UPOL_I5.json` | Registration | Methods, source-version interpretation and limitations; unresolved gate retained |
| `results/registration/UTIRIS_H1.json` | Registration | Methods, source-version interpretation and limitations; unresolved gate retained |
| `results/registration/UTIRIS_I0.json` | Registration | Methods, source-version interpretation and limitations; unresolved gate retained |
| `results/registration/UTIRIS_I1.json` | Registration | Methods, source-version interpretation and limitations; unresolved gate retained |
| `results/registration/UTIRIS_I2.json` | Registration | Methods, source-version interpretation and limitations; unresolved gate retained |
| `results/registration/UTIRIS_I3.json` | Registration | Methods, source-version interpretation and limitations; unresolved gate retained |
| `results/registration/UTIRIS_I4.json` | Registration | Methods, source-version interpretation and limitations; unresolved gate retained |
| `results/registration/UTIRIS_I5.json` | Registration | Methods, source-version interpretation and limitations; unresolved gate retained |
| `results/registration/comparison.csv` | Registration | Methods, source-version interpretation and limitations; unresolved gate retained |
| `results/registration/comparison.md` | Registration | Methods, source-version interpretation and limitations; unresolved gate retained |
| `results/registration/manifest.json` | Registration | Methods, source-version interpretation and limitations; unresolved gate retained |
| `results/simon_e0.json` | Earlier pilots | Historical context; acquisition/protocol confounding prevents promotion to current primary results |
| `results/simon_e0_per_structure.csv` | Earlier pilots | Historical context; acquisition/protocol confounding prevents promotion to current primary results |
| `results/synthetic_atlas_eyes.json` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/synthetic_atlas_tissue.json` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/toy_ladder.json` | Mechanism controls | Supplementary synthetic controls; toy evidence, no biological validation |
| `results/visual_rebuild_manifest.json` | Real-case demonstrations | Supplementary preparation history; fixed examples are not cohort validation |
