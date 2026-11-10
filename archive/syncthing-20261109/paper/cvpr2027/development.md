# Development record, 8 November 2026

The present manuscript is a first evidence-bounded draft. These items describe work needed to develop the CVPR contribution; they are not completed experiments.

## Latest user-directed scope

On 8 November the user specified experiments across human-body image datasets, with symmetry, identity, variability, inpainting and a comparison of distance metrics. The requested main image is an atlas of spatial differences across brain, eye, fundus, face and pupil-related views. `research_brief.md` records the detailed target, cohort gaps, map definitions and experiment sequence. Pupil geometry versus iris texture remains a terminology question. The manuscript now includes these design elements as proposed work, alongside measured representation comparisons from saved ocular aggregates.

## Text expansion and artifact inventory

At the user's request to build on repository artifacts, `ARTIFACTS.md` inventories 24 result JSONs, seven CSVs, three reports, 20 PNG figures, 17 PDF figures, 29 source Python files and 17 test files. File counts are not independent results, and the historical human-image derivatives remain outside this checkout.

The English manuscript gained approximately 1,600 words by a consistent source-token count, including methods for bilateral correspondence, a nine-row representation comparison, interpretation of photometric controls, and a proposed spatial-atlas/inpainting section. `prepare.py` now uses `iris_calibration.json` as an additional source and generates the new table directly from saved values. No paired significance claims or pixel-versus-phase-code leaderboard were added, because the latter uses a different QC population.

The working PDF is 11 pages and needs an editorial main/supplement split for an eight-page submission. This expansion prioritizes substantive text for review. The existing ISBI files and all result JSON/CSV inputs remain unchanged. No new experiments, raw-image inspection, dataset downloads, commits or pushes were performed.

## Changes from the ISBI framing

ISBI focuses on regression to the mean in brain-MRI inpainting evaluation. The CVPR question is whether a similarity representation supports an identity-preservation interpretation across controlled identity and nuisance comparisons. Mean collapse is one failure condition alongside nuisance sensitivity, exposure shortcuts and uninformative topology. Existing mean/shrinkage experiments are shared evidence and must be accounted for when checking publication overlap.

## Claims corrected in the new draft

The optimum for pointwise label accuracy does not imply a general optimum for Dice, whose denominator couples voxels. The label simulations support a conditional observation only. A score of one against a finite stranger panel means the candidate outranks that panel; it does not mean zero reconstruction error, equality with an oracle, or statistical equivalence. Nuisance matching changes the reference distribution, but does not make a representation nuisance invariant. A bilateral organ or relative is not a universal intermediate identity rung. A comparison against an independent population mean is a negative control, not necessarily the best context-conditioned predictor.

## Priority experiments for a stronger submission

1. Freeze identity units, endpoints, cluster weighting, exclusions and matching rules before new scoring. Separate method selection from held-out auditing.
2. Evaluate LPIPS, DINOv2 and DreamSim as established comparators; consider DINOv3 after checking weights, terms and feasibility. These are planned baselines, not results. Primary sources: [LPIPS](https://arxiv.org/abs/1801.03924), [DINOv2](https://arxiv.org/abs/2304.07193), [DreamSim](https://arxiv.org/abs/2306.09344).
3. Add a nuisance-normalized iris representation and held-out repeat-based residual noise floor. Preserve participant-disjoint template/reference/test pools and family grouping.
4. Audit outputs of actual reconstruction or inpainting methods on a cohort with observable repeated identity. Oracle shrinkage alone is insufficient. If the paper claims broad vision relevance, add a suitable nonmedical task with repeated identity and nuisance metadata.
5. Recompute real-data intervals with resampling of participants/families and calibration references, rerunning the calibration inside each resample. Current JSON intervals are inherited pilot intervals, not certified to include reference-pool uncertainty.
6. Report performance and missing-reference coverage per nuisance condition. Keep acquisition date, camera and disease progression confounding explicit in GRAPE.

## Editorial decisions still open

ISBI is submitted and under review, confirmed by the user in this session. Withdrawal has not been requested. The author list and enrollment, the final scope (audit benchmark versus a separately validated distance), and real-data reproducibility on the current machine remain open. This checkout has no `.git` directory, despite historical handoff references to a GitHub repository, so no current commit hash or push is claimed.

## Overlap matrix after the user's status confirmation

| Element in this development draft | In the local ISBI source? | Treatment for an independent CVPR paper |
|---|---|---|
| Mean-versus-donor squared-error argument | Yes | Shared background; do not count as new contribution |
| Tissue shrinkage and wrong-detail results | Yes | Shared evidence; cannot establish CVPR novelty |
| Label-space Dice/prior sensitivity | Yes | Shared evidence; remove from primary CVPR claim set |
| Matched conditional calibration with explicit limitations | Calibration appears in ISBI; expanded here | Expansion alone is insufficient to establish distinctness |
| Pupil nuisance intervention | Not in the local ISBI manuscript | Potential audit evidence; still exploratory and synthetic |
| GRAPE and UTIRIS pilot profile | Not in the local ISBI manuscript | Potential separate evidence; strengthen confound analysis and uncertainty |
| Held-out reconstruction-output audit across modern representations | No; not run | Candidate primary CVPR contribution requiring new experiments |

This comparison is against the local ISBI source, not a fetched copy of the actual submitted PDF. Differences in the submitted version must be checked before any enrollment. Current CVPR policy prohibits overlapping concurrent submissions; the working draft is allowed to develop the argument but is not cleared for submission. A distinct experimental contribution, not a rewritten title, is needed if both review processes continue.

## Provenance and safety of the writing workflow

Use existing aggregate results and synthetic figures only. Do not regenerate human images, download datasets, change the frozen real-case protocol, or treat historical Windows paths as current data availability. Preserve the ISBI manuscript and old results. This session uses Codex/OpenAI assistance; do not silently reuse the ISBI acknowledgment that names only Claude.
