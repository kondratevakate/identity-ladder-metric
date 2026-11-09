# Claim-level reproduction evidence review

Reviewer: Codex, using the existing repro-judge icml2026_v2 rubric. The README and exact rubric were read from the installed project. No paid/model judge was called, and no GLM calibration statistic is attributed to this review. No research experiment was rerun; code tests and manuscript preparation are recorded separately.

The claims below were extracted from this first draft and the named source protocols, not supplied as an official challenge claim list. A source's F/P or PASS label is not evidence by itself. Numeric artifacts, setting, counts, and measurement scope determine the verdict. Verified here means that the documented experiment supports the narrowly phrased claim; it does not certify provenance or independent experimental replication.

| ID | Exact or anchored claim | Verdict | Concrete evidence | Inference boundary |
|---|---|---|---|---|
| C1 | The QTIM MSE position of the dataset mean is below the mirror position | verified (2) | donor_ladder_brain_metrics.json, views/random_patches/LR/ladder_R/mse: 0.412716 versus 0.717274, with recorded intervals and 221 targets/168 families | A descriptive metric profile under provisional affine alignment, not anatomical identity validation |
| C2 | Under MSE the face dataset mean is closer than the repeat photograph in both datasets | verified (2) | face_ladder_metrics.json, views/*/ladder_R/mse: CelebA mean 0.499036 versus retest 0.640; IMDb-WIKI mean 0.498467 versus retest 0.690, source records retain full precision | Person membership is shared by the repeat but capture conditions and anatomy appearance differ; no universal optimum claimed |
| C3 | A context-selected CelebA donor can be close locally and near unrelated in the identity reference | verified (2) | face_ladder_metrics.json: local MSE R_self 0.399476; whole-image ArcFace donor-reference R_self 0.982955 | Different spatial scopes; not identity measured on an inpainted composite |
| C4 | The repeated iris is closer than the fellow eye in the three saved pixel summaries | verified (2) | iris_ladder_metrics.json, results: NIR 0.407 versus 1.256; VIS 0.569 versus 1.293; UPOL 0.177 versus 1.118, with person-bootstrap intervals and source eligibility | P2 pixel/support question only; no biometric invariance or full P3 claim |
| C5 | HACA3+ stage 1 reduces the recorded mean absolute paired contrast shift in the two frozen boxes | verified (2) | donor_ladder_brain_haca3_stage1.json, session_shift_z: deep 0.120959 to 0.016368; parietal 0.059131 to 0.033166, 23 pairs | Stage-1 contrast sensitivity, visual verdict pending; not individual-detail preservation |
| C6 | The full iris P3 Betti-related H4 fails and H5/H6 hold | inconclusive (0) | Workstation handoff reports the outcome, but the full numerical outputs are absent from the accessible snapshot | A handoff verdict alone is insufficient under the rubric; the draft records it only as pending integration |
| C7 | The implementation ignores filler-valued missing support | toy (1) | Executed known-geometry/mask tests in the repository test run check this behavior on constructed arrays | Unit-level proxy evidence does not certify every private preprocessing artifact or historical run |
| C8 | Fixed latent eye identity can receive a very low intensity rank after simulated pupil dilation | toy (1) | pupil_ladder.json and pupil_report.md: 96 synthetic targets; RMSE c about 0.004 after dilation and 1.000 on repeats | An executed synthetic mechanism experiment, not measured human pupil physiology |

The unchanged rubric score is 12/16 across these eight screened empirical claims. It is not a probability of truth, a paper-quality score, or a claim of full reproduction. Theoretical identities are checked algebraically in the scientific review rather than forced into an experimental logbook rubric.

## Additional negative and incomplete evidence

The CARE iteration-2 artifact records 36 comparison rows within rounding tolerance, 63 outside tolerance, 8 blocked by coverage and 1 unavailable. This contradicts a blanket statement that every released-score comparison reproduces, but does not refute the reconstruction method's scientific contribution. The draft keeps the narrow aggregation result separate from CARE model performance.

Directional T2 is not established for the recorded QTIM MSE comparison. A failed one-sided superiority test alone is not assigned a falsified label for a reverse scientific claim. Source confidence intervals and selection/multiplicity limitations remain visible.

The latest object-relative sweeps, a Brain-ID embedding endpoint, the complete modern-inpainting comparison and the private animation are not reconstructed from filenames or protocols. Missing evidence remains inconclusive or not run, according to its actual source state.

## 9 November: additional decision-oriented claim audit

These claims were extracted during the author's BraTS-question revision. They are separate from the eight-claim development review and its historical 12/16 total. No new overall score is calculated and no experiment was rerun.

| ID | Claim assessed | Verdict | Evidence inspected | Boundary |
|---|---|---|---|---|
| B1 | The saved official-evaluator input layout computes SSIM over two spatial axes | toy | brats_panel_checks_tm1.1.2.json and tm1.9.0.json, ssim_dimensionality; source helper and inspected official code use the four-dimensional batch/channel layout | Verified behavior on the constructed volume is proxy-scale evidence, not clinical quality or server-parity validation |
| B2 | Plain PSNR is not exactly rank-equivalent to RMSE on the saved clipping probe | toy | per_case_order: eleven constructed fills, Kendall tau 0.890909 for plain PSNR versus 1.0 for the epsilon variant; psnr_clamping records 12.316670 versus 9.091105 dB | Does not establish new anatomical information or measure the amount of duplicated weight on an actual leaderboard |
| B3 | An alternative metric gives a better BraTS-method ranking than the current rule | inconclusive | No valid shared-case comparative model predictions and independent MRI quality reference were found in the accessible evidence | The decision question remains open; donor/control profiles are partial progress |
| B4 | Dice/topology or a downstream segmenter fixes individual-detail evaluation in real MRI | inconclusive | label robustness is synthetic and prior-dependent; real_cases and real_cases_parietal explicitly withdraw their old c/rho/label outputs | Synthetic counterexamples justify testing the assumption but do not validate a real MRI endpoint |

The twelve-question coverage matrix is an authored map from evidence to decisions, not an empirical logbook verdict or a percentage of the problem solved. Its positive contribution is to identify discriminating checks and the missing transition from mechanism tests to validated competition choices.

## Superseding scope: proposed generator profiles across populations

The latest author clarification concerns general inpainting and conditional generation rather than a BraTS-centered decision. The historical claim scores above remain historical. The new population-audit specification and equation are a proposed readout, not an executed result to which a verified label can be assigned.

| Claim | Evidence status | Basis and limit |
|---|---|---|
| Metadata-matched means supply population reference controls | Documentary source support | Face/brain protocols define matched donor cohorts; current aggregate profiles contain comparator results. This is not subgroup model performance. |
| A generator's control-relative behavior differs between target populations | inconclusive | No generated-output subgroup experiment in the accessible snapshot; proposed profiling requires fixed targets/masks/support, sampling budgets and independent-unit uncertainty. |
| Such a group contrast identifies unfairness, causal model bias or mode collapse | inconclusive | Metric/reference sensitivity, task composition and sampling remain alternative explanations. No such empirical conclusion is reported. |
| Main infographic faithfully represents selected source panels | Presentation verification only | Source/crop/count/asset hashes in provenance_infographic.json, vector labels, source annotation assertions and rendered-page inspection. No scientific map is generated by AI. |

The anonymous decorative bust was produced with the built-in image tool and is disclosed in the LLM statement. It is not a participant image, experimental observation, biometric reference or reproduction result. No research-generator outputs were evaluated and no participant analysis or new metric experiment was run.
