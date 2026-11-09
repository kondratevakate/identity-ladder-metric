# Registration QC gate, 2026-11-02

Status: prospective review rules recorded before this chat inspects images. Gate NOT MET.
This dated clarification supersedes scoring fallback and exclusion language in 2D Amendments 2/2a and MRI Amendment 4 where it conflicts with Kate's 2 November pause. Historical protocols and outputs remain intact. No new metric scoring is authorized by this document.

## Ownership and verified inventory

At inspection, HEAD was 91e100f (2D Amendment 2a), newer than the handoff's 87a95a1. Another session is actively changing the shared checkout. Its untracked `src/ladder2d_register.py`, `src/ladder2d_figures.py` and modified `src/ladder2d_brats.py` remain protected. This chat owns this QC protocol only for now.

- Inspected fundus implementation uses ORB/RANSAC similarity and returns the unregistered image on failed fitting. This must be treated as registration failure, never acceptance or a scoring fallback. Amendment 2a specifies OpenCV, but the inspected implementation still uses scikit-image: specification and implementation are not yet synchronized.
- Inspected iris implementation aligns limbus centre and radius. It does not establish rotation or local texture correspondence. Amendment 2a specifies a separate UPOL pupil-centre strategy that was not yet present in the inspected file.
- Figure code contains metric loops. Do not run it as a QC-only entrypoint.
- Existing MRI process PID 40856 runs `src/real_cases_syn.py --limit 1 --wait-for-storage`. Do not duplicate or stop it. The private `syn_amendment4` output directory was absent at inspection; no completed SyN preparation was found there.
- The storage-maintenance lock is held by another session for SpeakingFaces download/registry maintenance. Its owner is also the 2D session. The historical migration-status file says complete but does not describe this current lock. Do not remove the lock or infer that the process has finished.

## Fixed review population

Use the existing hash-frozen GRAPE and iris selections in `PROTOCOL_2d_ladder.md`, without reselection, numerical-QC exclusions, replacement or ranking by image quality. A comparison is the unit; repeated people are not independent participants.

2D gate set: every frozen own-rung pair, plus a deliberately reflected-input control for every same-eye pair. The control horizontally reflects the moving input and requires explicit inverse reflection before estimating a proper similarity transform. It is a coordinate-system test, not a fellow-eye comparison. Fellow-eye comparisons remain a separate anatomical-correspondence stratum, with the full moving image reflected before crop extraction. Natural bilateral anatomy need not have identical fine texture.

Stratify GRAPE by all four original rungs, with reflected-input controls separately for each of the three same-eye rungs; stratify UPOL and UTIRIS separately by same-eye, reflected-input same-eye and fellow-eye. Preserve every intended pair in its denominator. Freeze the private expanded manifest and publish its hash and aggregate counts before image inspection. Until that manifest exists, denominator status is NOT FROZEN, not zero.

MRI gate set: all targets in the original frozen design, separately for the deep and parietal boxes; template alignment, available repeat correspondence, and reflected contralateral correspondence each have their own strata. Freeze exact intended rows from the existing selection before inspection. Retain unavailable repeat rows explicitly; do not confuse absence of a planned repeat with a registration failure. Expanded A1.1 does not become eligible through the original-design gate. Donor/pool images require their own QC before any future scoring, even if the target gate passes.

Development pilot: first pair in saved selection order in each 2D stratum, including its reflected-input control; the first existing completed SyN target for both MRI boxes, once available. Keep all pilot failures. These are development cases, not independent validation. Freeze a remaining-case validation manifest before tuning; never substitute an easier pair for a failed pilot. If a historical problematic mirror example can be identified, add it as a separately labelled regression case without changing gate denominators.

## Required private review sheet

Each sheet shows original full frames with orientation labels; exact anchor crop and inverse-mapped moving crop boundaries; registered full frames; both crops at equal display scale; a 50:50 blend; checkerboard; independently visible landmarks or estimated contours. State coordinate convention, image dimensions, transform direction, reflection order, interpolation and valid-support mask. Show original and registered views together. MRI requires axial, coronal and sagittal views through both boxes, physical coordinates, brain contours, and deformation diagnostics. Never replace the full field with a crop-only figure.

Use one display window for each pair and record it; do not separately tune contrast to hide mismatch. Pupil/limbus contours and automated feature matches are estimates, not ground truth. Review local correspondence independently of the points used to fit registration.

## Decisions recorded before review

PASS requires all of the following:

1. Correct side/orientation, transform direction and reflection; crop boundaries map back to the intended source anatomy with no off-by-one displacement.
2. At least 95% of each intended crop has valid source support; visible occlusion, padding and segmentation error are shown and do not conceal correspondence.
3. Gross anatomical landmarks and crop placement agree. For same-eye repeats, at least three distributed identifiable local features agree to within two pixels on the 224-pixel display grid; failure to identify reliable features is UNREVIEWABLE, not PASS. Iris pupil dilation is recorded, not removed by an unconstrained deformation.
4. Fellow-eye and MRI contralateral views cover homologous anatomical regions without requiring identical individual texture or folding. The reviewer records the landmarks used and any residual anatomical difference. A missing homologous region is FAIL; uncertainty is UNREVIEWABLE.
5. No implausible local warping, folding, severe stretching or loss of individual anatomy. MRI negative/nonpositive Jacobians fail; positive Jacobians alone cannot establish visual acceptance. Global correlation and successful execution cannot satisfy any visual criterion.

FAIL means an observable violation; UNREVIEWABLE means insufficient evidence or an unavailable required output; NOT REVIEWED means no visual decision yet. Store reviewer, timestamp, reasons and sheet hash privately. Keep automated flags, agent review and Kate's acceptance as separate fields. No automated field may set visual PASS.

## Counting and release rule

Report intended, reviewed, pass, fail, unreviewable and not-reviewed counts by dataset, original rung, reflection branch and MRI box. Reviewed = pass + fail + unreviewable. Visual acceptance fraction = pass / intended (raw proportion, higher is better); unreviewable and not-reviewed rows do not disappear. Missing manifest or no reviews gives NOT ESTABLISHED, not an observed 0% success rate. Report computational preparation failures separately from visual failures.

Require at least 90% visual acceptance overall AND in each prespecified stratum, including each mirror stratum, with no unresolved systematic orientation/crop error. Report pilot and remaining validation results separately. Kate's visual acceptance remains explicit. A small successful pilot does not release the metric pause. Any criterion change is dated and all affected cases are re-reviewed under a new version while preserving previous records.

## Next executable actions

### Independent development pilot addendum (2026-11-02, before its images)

Kate authorized independent progress without waiting for Claude. `src/registration_qc_pilot.py` owns a separate private `registration-qc-independent-20261102-v1` directory. It reads immutable source data, leaves registry/path maintenance and the existing SyN process untouched, and does not import concurrent registration or figure code. All frozen 2D pairs and additional reflected-input controls are enumerated before pixels are read; first saved pair per stratum is the development pilot. Remaining rows stay not reviewed. This is operational independence, not a blind reproduction: earlier code was inspected.

This first candidate uses OpenCV SIFT (2,000 features), ratio 0.7, RANSAC proper similarity (3 pixels at 512, 2,000 trials, seed 20261102), minimum 12 inliers, scale 0.8 to 1.25 and rotation at most 20 degrees. It preserves full fields in a 512-pixel letterboxed canvas and permits no nonlinear deformation. Fellow-eye fits are exploratory and may fail because fine features are not homologous. Failures produce black registered panels, never identity/unregistered fallback. Source support is a diagnostic, not visual acceptance.

Pilot crops are explicitly diagnostic: 146-pixel square at (183,183) for fundus and (310,183) for iris on that full-field canvas. They do NOT claim equivalence to the historical centre-square or later pupil/limbus-canonical metric crops. Exact historical and canonical crop-chain validation remains required before the gate. Fitted keypoints and Canny edges aid inspection but are not independent anatomical landmark validation. These limitations prevent this pilot from releasing the metric pause even if its similarity fits succeed.

1. Recheck the current lock, owning implementations and existing SyN completion without launching duplicate jobs.
2. Once ownership and storage are clear, freeze the expanded private QC manifests and hashes; snapshot the registration implementation before using it.
3. Produce the fixed pilot sheets in a new private run directory with input/code/protocol/transform/output hashes. Inspect and fix registration only; never invoke scoring loops.
4. Complete the remaining fixed review set and count every intended comparison. Regenerate `results/METRICS_STATUS.md` after QC milestones. Gate remains NOT MET until the actual visual records support it.
