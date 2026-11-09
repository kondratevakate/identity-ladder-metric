# Identity Ladder: cross-anatomy study and main-figure brief

Updated 8 November 2026 from the user's direction. This is the current research target, not a report of completed experiments. The expanded `main.tex` and `main.pdf` now incorporate bilateral-comparison methods, saved representation controls and an explicitly proposed atlas/inpainting extension. The expanded empirical benchmark remains unperformed; `ARTIFACTS.md` records the available evidence.

## Paper question and working title

**Identity Ladder: Measuring Identity, Symmetry, and Variability in Human Images**

Which image distances distinguish individual anatomy from within-person variation, bilateral differences, and acquisition changes, and how does that distinction affect the evaluation of inpainting?

The paper will compare distances across images of the human body, initially brain MRI, external-eye images, fundus photographs, faces, and pupil-related measurements. These are the requested anatomical views, not five confirmed independent datasets. Pupil geometry and iris texture may be derived from the same eye images and must not be counted as independent cohorts. The user has been asked whether the final view means pupil geometry, iris texture, or both; until clarified, retain the distinction.

The intended contribution is a cross-anatomy experimental benchmark with an explicit comparison protocol and a spatial atlas of differences. The ISBI mean-collapse argument provides motivation. The central new evidence must come from the cross-anatomy experiments, the comparison of distance representations, and their application to actual inpainting outputs. We do not assume in advance that one metric will win across all domains.

## Narrative

Human images vary across people and also across repeated measurements of one person. Bilateral structures introduce another comparison: two hemispheres, two eyes, or two sides of a face share a person while retaining different anatomy. Image distances can respond to any combination of these factors. The benchmark asks which interpretation is supported by controlled comparisons rather than imposing a universal ordering on all pair types.

Inpainting provides the practical test. A reconstruction can be close in pixels, retain the target's distinctive structure, or merely resemble a typical member of the population. The experiments should establish when these assessments agree and when their rankings diverge. Deterministic averaging, realistic donor detail, and outputs of reconstruction models serve different experimental roles and must remain identifiable in the analysis.

## Data and comparison matrix

The following availability statements are from the local project records, not a new audit of raw data or access terms.

| Anatomical view | Existing lead or evidence | Intended comparisons | Work still needed |
|---|---|---|---|
| Brain MRI | QTIM prepared-case records; SIMON repeat-acquisition records | Repeat/visit, acquisition change, left-right anatomy, unrelated people; family only with verified labels | Verify current data availability and eligible comparisons; QTIM has one T1 per session and no public MZ/DZ labels |
| External eye / iris | UTIRIS and UPOL leads; UTIRIS aggregate pilot | Same-eye capture, fellow eye, sensor/spectrum, unrelated eye | Separate eye identity from person identity; assess segmentation and normalization coverage |
| Fundus | GRAPE aggregate pilot; PAPILA lead | Same-eye visit, fellow eye, camera, unrelated eye | Separate camera and visit/progression effects; build spatial correspondence before pooling maps |
| Face | FEI and MUCT catalogue leads, without benchmark results | Same-person appearance change, left-right face, unrelated person | Audit metadata and permitted use; choose an eligible repeated-identity cohort |
| Pupil geometry | Existing synthetic dilation experiment | Same-eye dilation, contour/area change, unrelated controls | Confirm requested scope; establish measured human pupil data and annotations if empirical physiology claims are intended |

Unavailable comparisons receive `N/A`, not an inferred ladder position. A flipped image of one eye is an augmentation, not a second eye. Facial reflection within one photograph is an asymmetry measurement, not an independent acquisition. Family comparisons are optional where the metadata support them.

## Figure 1: atlas of differences

The main figure should show the data and spatial differences. A generic workflow schematic can move to a later figure or the supplement.

Use anatomical views as rows: brain, external eye/iris, fundus, face, and pupil geometry if confirmed. For each row, show a representative reference image, then maps for repeat-capture differences, bilateral asymmetry, between-person differences, and the individual residual relative to an independently estimated population template. Add a compact metric-ranking strip beside the atlas, or as panel B, connecting the visible differences to the paper's distance comparison. If this is too dense at two-column width, retain repeat/asymmetry/between-person maps in the main atlas and move residual maps to Figure 2.

Every row needs a clear identity unit, dataset label, number of independent contributors, valid-coverage indication, and a row-specific color bar. Where a comparison is unavailable, leave a labeled empty cell. Do not force identical experimental coverage across modalities. A fifth pupil view from an eye cohort must be labeled as a derived view, not another dataset.

Spatial residuals must be computed from the images after a documented, fixed correspondence and intensity-preprocessing procedure. They are measurements, not generative illustrations or model attention maps. Feature distances without spatial attribution belong in the ranking strip; their scalar values cannot be painted onto anatomy as if they were pixel residuals.

### Map definitions

Let `T` be a prespecified transformation into a common within-modality coordinate system. It must not optimize a separate correspondence to make each tested distance look good. For a valid pixel/voxel `v`, define the channel-averaged squared difference:

`e_ab(v) = mean_c [(T_a x_a)(v,c) - (T_b x_b)(v,c)]^2`.

For grayscale MRI there is one channel; for RGB images use the same fixed channel treatment throughout the row. For each pair condition `q`, compute a mean error map within each participant, then average those participant summaries with equal participant weights. Display `R_q(v) = sqrt(mean_participant mean_pairs e_ab(v))`. Where family structure determines independence, use the prespecified family weighting instead. Report valid coverage at each location and exclude interpolation padding.

The repeat map measures observed within-unit acquisition variation under the specified condition. The bilateral map compares actual opposite-side anatomy after documented reflection/correspondence. The unrelated map uses a prespecified matched donor panel. The individual-residual map compares held-out targets with a template estimated without those targets or their families. Display a representative residual separately from a cohort RMS map and label which is shown.

A difference between unrelated and repeat maps can be an exploratory contrast. It is not automatically a map of biological identity variance: acquisition, registration, progression, and unequal variance can all contribute. Do not silently clip the contrast into an apparently positive variance estimate.

Use one display scale across comparison columns within an anatomical row. Cross-modality raw values have different units and must not share an unlabeled absolute scale. Any normalized version needs an explicit denominator, a repeat-derived resolution rule, and a companion raw-scale result. Fix crops and examples independently of metric rankings. Include alignment sensitivity in the supplement because alignment itself can erase or introduce apparent individuality.

**Draft caption, to complete after computation:** “Spatial differences across human-image domains. Rows show [datasets and identity units]. Columns compare repeat acquisitions, actual bilateral counterparts, unrelated individuals, and residuals from an independent population template. Maps show [specified aggregation] after fixed within-domain preprocessing; scales are shared within each row. The accompanying distance profiles test which representations distinguish these comparison conditions. [Independent sample counts, coverage, and uncertainty to be inserted from measured outputs.]”

The historical `population_mean_four_modalities.png` in `src/detail_report.py::shrinkage_gallery` is related visual material, but contains synthetic tissue, synthetic eye, one real MRI case, and one real iris case. It is not the requested brain/eye/fundus/face/pupil atlas. Existing private examples and their mappings remain in the local data area; the present brief does not move participant images into the paper folder.

## Experiment sequence and endpoints

**E1: Spatial variability atlas.** Freeze eligible cohorts, identity units, pair definitions, registration, intensity handling, exclusions, and display choices. Compute the maps above and summarize the amount and spatial distribution of repeat, bilateral, temporal, and between-person variation wherever those comparisons are supported. Treat time and camera as separate factors where identifiable; report confounded conditions explicitly.

**E2: Distance comparison.** Evaluate the same eligible pairs with pixel, perceptual, representation, and anatomical distances. The primary discrimination endpoint is same-unit repeat versus nuisance-matched unrelated comparison, with participant/family resampling and calibration reconstructed inside each resample. Report raw distances and conditional calibrated scores. Additional bilateral and temporal contrasts answer their own anatomical questions rather than enforcing a complete ladder. Report ranking disagreement and sensitivity to nuisance changes alongside discrimination.

**E3: Controlled stress tests.** Use shrinkage toward an independent mean, blur, incorrect donor detail, acquisition-like intensity changes, registration perturbations, and pupil dilation where defined. Synthetic interventions isolate mechanism; real repeats test whether that mechanism describes measured data. Pure mirror symmetry is not a suitable realism optimum because biological asymmetry may be genuine.

**E4: Inpainting.** Apply selected, reproducible reconstruction methods to held-out regions with observed original images. Include a simple interpolation baseline, population-template fill, contralateral donor where meaningful, matched unrelated donor, and at least one trained reconstruction method feasible on the available data. Score the masked interior and a separate boundary band; test sensitivity to mask size and location. Compare metric rankings with held-out identity evidence and structure-specific measurements. A plausible alternative hidden anatomy may differ from the observed original, so reference error alone is not an identity oracle. Multiple samples per case remain clustered within that case.

Template contributors, calibration donors, training data and held-out targets need declared, identity-disjoint roles; family-disjoint separation is needed where family membership matters. Known-target shrinkage is an oracle diagnostic and cannot stand in for an actual reconstruction method. Pupil measurements from the same eye images remain clustered with that eye/person. Select hyperparameters on development identities, then freeze them for the held-out audit.

## Distance families

| Family | Planned comparators | Question |
|---|---|---|
| Pixel / local structure | MAE, MSE or RMSE, PSNR, SSIM | How strongly do scores reflect photometry, alignment and averaging? |
| Learned perceptual | LPIPS, DreamSim | Do perceptual representations distinguish retained target structure from plausible alternatives? |
| General representation | Cosine and Euclidean distance on fixed DINOv2 features | Which identity and nuisance contrasts survive a shared representation? |
| Domain representation | Iris code, fundus features, face representation, where suitable and permitted | What changes when the representation is specialized to the anatomical task? |
| Anatomy / shape | Dice where segmentation exists, boundary distances, pupil geometry; topology only if informative | Which differences are structural, and what detail do the summaries discard? |

These are proposed comparators, not measured winners or claims of exhaustive current coverage. Fix feature extraction, normalization, spatial pooling, layer/checkpoint, and image conversion before testing. For cosine on unit-normalized vectors, squared Euclidean distance is `2(1-cosine)`; do not count their identical within-stratum ranking as independent evidence. Likewise, monotone variants of MSE do not create additional ranking evidence under fixed-range conditions. Image-dependent PSNR range/clamping can break that equivalence and must be documented.

Primary comparator sources checked on 8 November 2026: [LPIPS](https://arxiv.org/abs/1801.03924), [DreamSim](https://arxiv.org/abs/2306.09344), and [DINOv2](https://arxiv.org/abs/2304.07193). Their published purposes motivate the comparator families; they do not establish anatomical identity validity in our datasets.

## Manuscript outline

The introduction poses the distinction between identity, symmetry, and within-person variability, using Figure 1 as the empirical motivation once the atlas exists. Related work places the study relative to perceptual similarity, biometric identity, anatomical asymmetry, and reconstruction evaluation. Methods define cohorts, comparison conditions, map construction, representations, calibration and uncertainty. Experiments first characterize real variation, then compare distances, then test controlled perturbations and actual inpainting. Discussion explains which interpretations each metric supports, where coverage is missing, and how reconstruction reporting should change.

The main results should include a cohort/coverage table, the difference atlas, a metric-by-condition comparison with uncertainty, and an inpainting comparison that makes ranking disagreements visible. Full per-domain maps, alignment checks and perturbation curves can go to the supplement. The main claim should follow the observed pattern; a universal identity metric or biological ordering must not be assumed in the title, abstract or conclusions.

## Existing evidence versus planned contribution

Saved synthetic mean-collapse, shrinkage, label and pupil experiments and aggregate ocular pilots are starting material. There is no completed five-view atlas, face benchmark, empirical pupil-physiology cohort, or held-out multi-method inpainting comparison established by this update. No new raw data were read and no experiments were run while writing this brief.

The publication-overlap analysis already recorded in `development.md` remains applicable. The new experimental scope is a route to a distinct contribution; this brief does not establish that the contribution has already been completed or cleared for submission.
