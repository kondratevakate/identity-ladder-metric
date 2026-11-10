# Protocol: registration and visual QC before identity distances

**Current authority: Amendment 5 below (2 Nov 2026).** Codex owns both 2D and MRI registration after the Claude study stopped and merged at `74c01f8`. This is the single active registration/QC protocol; `results/registration/` is its single public aggregate result tree. The original frozen text and Amendments 1–4 below are retained as historical specifications, with an exact pre-unification snapshot under `docs/archive/registration-pre-unification-20261102/`. Amendment 5 supersedes references to separate ownership, parallel QC protocols, failure exclusion from the gate, and permission to compute new c/downstream results before visual acceptance. Historical results are not relabelled.

Status: FROZEN 2 Nov 2026 on Kate's approval of draft 1 ("Approve, freeze it"), before any registration quality
metric, c or figure was computed under this protocol. Any later change is a dated amendment, and results computed
before an amendment are never relabelled. Git commit and file hashes give provenance, not external preregistration.
Scope: the next paper (the identity ladder). Nothing here goes into the ISBI 2027 paper.

Why a separate study (Kate, 2 Nov): the 2D ladder numbers were distorted by poor registration, and registration,
harmonisation and generative alignment are different methods, sometimes different for each data type. This protocol
fixes which methods are compared, which quality metrics are measured and how a method is chosen, before any of them
is run on the frozen selections. It does not replace the visual QC gate (`docs/REGISTRATION_QC_PROTOCOL.md`, owned by
another session): a method chosen here must still pass that gate before any identity result is computed with it.

## Question

Before an identity distance is computed between two images, how should images of each data type be brought into
correspondence so that every image that enters a comparison (the person's own other image, the fellow eye or other
side, each matched stranger, each member of the population mean) is treated the same way, and no method favours the
person?

Sub-questions, per data type:

1. Accuracy. How close does each method bring homologous anatomy, measured on structures the method did not use to
   fit, and on images moved by a known transform?
2. Failure. On how many images or pairs does each method fail, and what happens to a failure?
3. Fairness. Is accuracy, and is failure, the same for the person's own images as for the fellow eye, the strangers
   and the mean members? A method that aligns the same person better than a stranger raises the calibrated score c of
   the person for reasons that have nothing to do with identity.
4. Downstream effect. How do calibrated c of the own rung, the fellow eye and the population mean move under each
   method, compared with no registration?

The method for each data type is chosen on questions 1 to 3 only. Question 4 is reported but never used to choose a
method, so the choice cannot be tuned to the identity result.

## Data types and inputs

| Data type | Input | SHA-256 | Status |
|---|---|---|---|
| Fundus | GRAPE CFP (registry `grape-fundus`) and `selection_grape.json` (private, data root) | selection `d3c236d60d125f886ea93ed2804a815b0007e12ccc1082c01b36c2d1a7a4708a`; metadata `bc79566f4687f7512628ee26959eca0fdb11dd859d217ebf82be99035fd2e02e`; per-image SHA-256 to the run manifest | available |
| Iris, NIR | UTIRIS (registry `utiris`) and UPOL (registry `upol-iris`), `selection_iris.json` (private) | `c3a41f09ac7d622c8215788b99c2d67ce1ecfc729d05ba33aab1b319240e107c`; per-image to the manifest | available |
| Face | SpeakingFaces (CC BY 4.0) | to be recorded by amendment when downloaded and selected | download pending (storage lock held by the 2D session) |
| Brain MRI | QTIM, through the existing `src/real_cases_syn.py` pipeline (another session) | from that pipeline's own manifest | later; read-only from that pipeline |
| Fundus, external accuracy | FIRE (134 fundus pairs with ground-truth correspondences [1]) | to be recorded if Kate approves the download | optional, not downloaded |

Code entering the study at its start: `src/ladder2d_register.py` `c3b7e396e5954a9fea9b68bd961c6bc61f7ad7c6ffed4567b8c2720a28634dd3`,
`src/iris_codes.py` `1f294f963b5f327e05173649b6af18cc9a1738bdc34d1d8af8567e005000bc92`,
`src/ladder2d.py` `81a9a42973c5351e2988592b7c581193a49a5626cbf07527328dcdf5de20aeee`,
`docs/PROTOCOL_2d_ladder.md` `b19e809950ecfef030e8e74a927024cb86c0cb203118031da410c93429372d72` (at 00f9534).
Every run refuses to start if an input hash differs, and records the SHA-256 of every image it reads.

The pairs, matched strangers and mean pools are those of the frozen 2D selections, without reselection, replacement
or ranking by quality. Roles of an image within a pair: anchor, own candidate (same eye, other capture), fellow eye,
stranger (the pair's reference), mean member.

## Method families

Two kinds of method are distinguished, because they differ in how they can favour the person.

- **Per-image frame.** Each image is mapped on its own to one fixed frame from structures detected in that image
  alone (pupil and limbus circles, disc and fovea, facial landmarks, a brain template). The mapping of an image does
  not depend on the image it is compared with, so the method is symmetric by construction; it can still be biased if
  it fails or is less accurate on some people.
- **Pairwise.** The candidate is moved onto the anchor by optimising agreement between the two (keypoints,
  intensity, deformation). The more alike two images are, the easier the fit, so the person's own image is fitted
  more often and more closely than a stranger (in the GRAPE check of Amendment 2b, ORB registered every same-eye pair,
  0 of 20 fellow-eye pairs and no strangers). A pairwise method is admitted only if it is applied in exactly the same way to every
  candidate slot (own, fellow, every stranger, every mean member), and its fairness is measured below.

Rules common to all methods:

- A failure is never replaced by the unregistered image. The failed image is excluded and counted, by role; a pair
  whose anchor or candidate fails is dropped and counted; strangers and mean members that fail are skipped, and a pair
  keeps at least 30 strangers and 8 mean members or is dropped and counted.
- The fellow eye or other side is reflected left-right before registration, and the reflection is recorded as part of
  the transform, so a reflection is never estimated as a rotation.
- Harmonisation references are fixed population references built from people outside the pair, never the anchor.
- Every parameter below is fixed now. A parameter change after any quality metric is seen is an amendment, and the
  earlier result is kept.

### Fundus (GRAPE)

| Code | Family | Method |
|---|---|---|
| G0 | none | centre square crop, 512 base (as the 2D protocol); reference row for every metric |
| G1 | per-image landmark frame | disc-fovea similarity frame of Amendment 2b (`ladder2d_register.fundus_canonical`) |
| G2 | pairwise keypoint | ORB 800 keypoints on the CLAHE green channel inside the field of view eroded by 15 px, ratio 0.8, RANSAC similarity (OpenCV `estimateAffinePartial2D`, 3 px, 500 iterations, `setRNGSeed(20261101)`) [2, 3]; accepted if at least 12 inliers, scale 0.8 to 1.25, rotation under 20 degrees (Amendment 2a), else failure |
| G3 | pairwise intensity | G1 frame, then ECC maximisation [4] (OpenCV `findTransformECC`, Euclidean motion, grey 512 image inside the joint field of view, 100 iterations, epsilon 1e-5); failure if ECC does not converge or the update exceeds 10 px or 5 degrees |
| G4 | deformable | G3, then dense optical flow (OpenCV DIS, preset MEDIUM) on the green channel, smoothed by a Gaussian of sigma 8 px; failure if any Jacobian determinant of the smoothed field is at most 0 |
| H1 | intensity harmonisation | G1 frame, then per-channel histogram matching inside the field of view to a fixed reference: the per-channel pooled histogram of 50 images from people outside every selected pair, drawn with the seed |
| H2 | device harmonisation | G1 frame, then per-camera standardisation: each channel z-scored inside the field of view with the camera's mean and SD estimated on the same 50-image pool |
| E1 | generative alignment | cross-camera image translation (CycleGAN-type [5]); not run under this draft: it needs training compute and its own amendment. Evaluated, if run, with every metric below plus the hallucination concern of distribution-matching losses [6] |

### Iris, NIR (UTIRIS, UPOL)

| Code | Family | Method |
|---|---|---|
| I0 | none | centre square crop, 224 base |
| I1 | per-image landmark frame | limbus centre to the image centre, limbus radius to 80 px (Amendment 2; UTIRIS); UPOL is run too and its failures counted |
| I2 | per-image landmark frame | pupil centre to the image centre, working width kept (Amendment 2a) |
| I3 | per-image normalisation | rubber-sheet unwrapping between the pupil and limbus circles [7], 64 radial x 256 angular samples, which removes pupil dilation; no rotation search |
| I4 | pairwise rotation | I3, then circular shift of the unwrapped image over plus or minus 16 angular samples to the best correlation with the anchor, as in iris matching [7]; applied to every candidate slot |
| I5 | pairwise intensity | I1 (UTIRIS) or I2 (UPOL), then ECC, Euclidean motion, as G3 |
| H1 | intensity harmonisation | I1 or I2, then histogram matching inside the iris annulus to a fixed population reference of 50 images per dataset |

Pupil dilation is a real within-person change. I3 removes it by construction, so its downstream effect is reported
separately and it is never the only iris method carried forward.

### Face (SpeakingFaces, visible images; by amendment once downloaded)

| Code | Family | Method |
|---|---|---|
| F0 | none | centre crop |
| F1 | per-image landmark frame | five-point similarity alignment (eye centres, nose tip, mouth corners) to a fixed template, the alignment used before face embeddings such as ArcFace [8]; landmark detector fixed by amendment before the data are read |
| F2 | pairwise intensity | F1, then ECC, Euclidean motion |
| F3 | deformable | F1, then thin-plate spline from dense landmarks to the anchor; expected to erase identity, included as a negative example |

The thermal channel and cross-spectral alignment are out of scope.

### Brain MRI (QTIM; later, read-only)

The QTIM pipeline (rigid and affine to a template, then symmetric diffeomorphic SyN [9]) belongs to another session
and is not run or duplicated here. When its outputs exist, this study reads them and computes the same quality
metrics. Families to compare by a later amendment: affine only, SyN, intensity standardisation [10], and, for
repeat scans, a halfway space in which neither time point is privileged, which avoids the bias that asymmetric
processing introduces into longitudinal estimates [11, 12]. Harmonisation of derived features across scanners
(ComBat [13, 14]) is a feature-level method and is evaluated only through the downstream metrics.

## Quality metrics

All distances in pixels of the 224 analysis grid unless stated. Every metric is reported per data type, method and
role, with person-cluster bootstrap 95% intervals (2,000 resamples, seed 20261101) and its denominator.

**Q1 Accuracy.**
- Q1a Known-transform recovery (all methods). Each image is moved by a random similarity transform drawn with the
  seed (rotation up to 10 degrees, scale 0.9 to 1.1, shift up to 10% of the width) and, for G4 and F3, also by a
  smooth random displacement field (Gaussian sigma 16 px, maximum 4 px). The method is run again; target registration
  error = mean distance, over a 16 x 16 grid of points inside the valid region, between where the recovered and the
  true mapping send each point. For a per-image frame this measures the repeatability of the frame; for a pairwise
  method, recovery of the moved image onto its own original. Primary accuracy metric where no manual landmarks exist.
- Q1b Held-out structure error (all methods). Distance between independently detected structures that the method
  did not use to fit, after mapping. Fundus: Dice of the vessel maps (the fixed classical pipeline of the 2D protocol)
  in a ring of 0.3 to 0.6 field-of-view radii around the disc, and, for G2 to G4, disc and fovea position error.
  Iris: pupil-centre distance for I1, limbus-centre distance for I2 (UTIRIS, where the limbus is found). Face:
  landmarks of the dense set not used by F1. Local structure overlap is the surrogate that best separates accurate
  from inaccurate registration [15].
- Q1c Same-eye patch correlation (Pearson, grey, the patch of the 2D protocol). Reported because it was used in the
  checks so far, but labelled a surrogate: image similarity cannot by itself distinguish accurate from inaccurate
  registration [15]. Never used alone to choose a method.
- Q1d Inverse consistency (pairwise methods). Register a to b and b to a; mean distance of the composed mapping from
  the identity over the grid [16].
- Q1e Plausibility (deformable methods). Fraction of the field with Jacobian determinant at most 0, and the 99th
  percentile of the absolute log Jacobian.
- Q1f External accuracy (optional, fundus). If Kate approves the FIRE download: mean landmark error on its 134
  annotated pairs for G1 to G4, under FIRE's own protocol [1].

**Q2 Failure.** Computational failure rate per data type, method and role; pairs dropped per rung. Computational
failures are kept separate from visual QC failures, which are counted under the QC gate.

**Q3 Fairness.**
- Q3a Failure parity: failure rate of strangers and mean members minus that of own candidates; fellow eye minus own.
- Q3b Accuracy parity: median Q1a and Q1b by role; stranger minus own, fellow minus own, paired within person where
  possible.
- Q3c Candidate-slot null: for every pair, one stranger not in its reference is passed through the candidate slot
  with exactly the code path of the own candidate, and scored against the pair's reference. Its mean c must be 0.5;
  a method that treats the candidate slot differently from the reference shows here.
- Q3d Swap symmetry: anchor and candidate exchanged; mean of the paired difference in c, the asymmetry bias that
  longitudinal processing is known to introduce [11, 12].
- Q3e Reflection control: every same-eye candidate also passed reflected and explicitly un-reflected; mean paired
  difference in c from the unreflected case must be 0. A coordinate test, not a fellow-eye comparison.

**Q4 Downstream ladder.** Calibrated c (`grape_calibration.calibrated_score`, as in the 2D protocol) in the pixel
representation P (primary) and the nuisance control N, for: the closest own rung (GRAPE `same_camera_<1y`, iris
`same_eye_same_sitting`), the fellow eye, and the population mean (lambda = 0). Contrast: c_mean minus c_own and c_own
minus c_fellow, per method, and each method minus G0 or I0. Q4 diagnoses what a method does to the ladder; it is not
an identity result while the QC gate is not met.

## Claims under test

**R1 Accuracy.** Per data type, the ranking of methods by Q1b (primary) and Q1a, with intervals.
**R2 Failure.** Failure rate per method and role.
**R3 Fairness.** Q3a to Q3e per method.
**R4 Choice.** One recommended method per data type (one per dataset for iris), chosen by the rule below.
**R5 Downstream.** Q4 per method, descriptive.

## Prespecified interpretation

- A method is **eligible** for a data type only if all of these hold: computational failure is at most 10% in every
  role (the 90% bar of the QC gate); no Q3a difference has a 95% interval entirely above 5 percentage points against
  the stranger or fellow role; no Q3b difference has an interval entirely above 1 pixel in Q1a or entirely above 0.05
  in Q1b Dice in the person's favour; the Q3c interval contains 0.5; the Q3d and Q3e intervals contain 0. For
  deformable methods, also: Jacobian determinant at most 0 on 0% of the field after smoothing.
- A method that fails only the fairness conditions is reported as **biased in the person's favour** and is never
  used for identity results, however accurate.
- Among eligible methods, the **recommended** one has the lowest median Q1b; methods whose Q1b intervals overlap are
  tied, and the tie goes to the method with fewer degrees of freedom (per-image frame, then similarity, then affine,
  then deformable), then to the lower median Q1a. Q4 does not enter the choice.
- If no method is eligible for a data type, the data type is reported as **not registrable under this protocol** and
  no registered ladder is computed for it.
- The recommended method then goes to the visual QC gate; only after that gate passes is the 2D ladder rerun with it,
  under an amendment to `docs/PROTOCOL_2d_ladder.md`.
- No threshold or parameter is tuned after looking at any metric. Number of claims: five, over up to eight methods
  per data type; no multiplicity correction; the number of intervals reported is stated.

## Experiments (a separate network)

Each method is its own experiment with its own script, tests and private run directory, and writes one aggregate
file; a shared evaluator computes Q1 to Q4 identically for every method.

- `src/reg_methods.py`: one function per method code above, each returning (registered image, transform, valid mask,
  failure reason); unit tests on synthetic images with known transforms (recovery, reflection order, failure
  handling, no fallback to the unregistered image).
- `src/reg_eval.py`: Q1 to Q4 from a method's outputs; tests on synthetic data where the answer is known (an
  intentionally biased toy method must fail Q3c, a symmetric one must pass).
- `src/reg_run.py --dataset grape|utiris|upol --method G1`: one experiment; private outputs in
  `C:\Projects\data\projects\identity-ladder-metric\run-<date>-registration\<dataset>_<method>\`.
- Order: GRAPE and UTIRIS first (G0 to G4, H1, H2; I0 to I5, H1), then UPOL, then faces after download, then MRI
  from the QTIM outputs. CPU: at most four threads, one experiment at a time (shared machine).

## Outputs

`results/registration/<dataset>_<method>.json`: aggregates only (per role: n, failure rate, Q1 medians and
intervals; Q3 contrasts; Q4 means and contrasts). `results/registration/comparison.md` and `.csv`: one table per data
type, methods in rows, Q1b, Q1a, failure, fairness verdict, eligibility and Q4 in columns. `results/registration/
manifest.json`: protocol, code, input and image-set SHA-256, git commit, seed, package versions, threads, run time.
Images, image names, person IDs, transforms and per-pair values stay in the private run directory.

## Evidence boundary

- No manual landmarks exist for GRAPE, UPOL or UTIRIS. Accuracy rests on held-out automatic structures (Q1b) and on
  known-transform recovery (Q1a). Q1a measures recovery of a synthetic motion, not of real eye or head motion, and
  automatic structures carry their own detector error. FIRE (Q1f) would give real ground truth for fundus only.
- The fairness tests detect a method that treats roles differently; they cannot prove that real anatomy is aligned
  equally well in all roles, because strangers have different anatomy and no ground truth exists for them.
- Deformable and generative methods can remove individual anatomy; Q1e and Q4 show this only indirectly.
- The selections were already analysed unregistered (2D protocol primary run and Amendment 1), and same-eye patch
  correlations under G1, G2, I1 and I2 were seen in the registration checks of Amendments 2a and 2b. Those methods are
  therefore partly unblinded; the others are first looks.
- Iris sets are small and single-sitting; GRAPE is one hospital's glaucoma cohort. Face and MRI are not covered until
  their amendments.

## Ethics and data

As in the 2D protocol: public de-identified data under their recorded terms; whether Kate's institution requires an
exemption determination for this secondary use: `[to confirm with Kate]`. Real images, image names, person IDs,
transforms, overlays and per-pair values never enter this public repository. Downloads (FIRE, SpeakingFaces) only
under the storage-maintenance lock and with Kate's approval.

## Amendment 1 (2 Nov 2026): implementation details, before any quality metric was computed

Fixed while writing the code, after timing the operations on eight GRAPE images (time only, no metric). Nothing below
changes a method, metric or decision rule; it settles what the frozen text left open.

- Common form of every method: a mapping from the output grid to the coordinates of the raw image (GRAPE: the
  512 base image; iris: the working image of `iris_codes`, width 320; I0 maps the 224 base crop to the working image),
  the output image, a valid mask (GRAPE: red channel above 0.1; iris: non-zero canvas) and a failure reason. Q1a and
  the iris Q1b distances are measured in raw pixels and rescaled to a 224 grid of the raw image (x 224/512 GRAPE,
  x 224/320 iris), one unit for every method of a dataset. GRAPE Q1b landmark errors are measured on the output images
  (512 for every method) and rescaled x 224/512.
- Harmonisation reference (H1, H2): "50 images from people outside every selected pair" cannot be met, because every
  image of the universe enters some pair as a stranger or mean member. The reference is 50 images drawn with the seed
  from the whole dataset, one fixed reference for every image (per dataset for iris); it is never the anchor.
- H1 on iris: histogram matching over the whole valid canonical region, not the annulus, because UPOL has no limbus.
- Q1a: one random transform per image (seed plus image index); 16 x 16 grid inside the valid region of the anchor's
  output; the anchor is the original image and the moving image is its moved copy, for per-image and pairwise methods
  alike. Error per point = distance in raw coordinates between where the moved image's mapping lands after undoing
  the known motion and where the original's mapping lands.
- Q1b, fundus: vessel maps (primary 12% threshold) computed once per raw 512 image and carried through each mapping
  (nearest neighbour); the ring is centred on the disc found on the anchor's output image, radius from its field of
  view. Disc and fovea errors from `fundus_landmarks` on the output images, reported for every method and used as a
  held-out structure for G2 to G4 only. Iris: pupil centre from `pupil_blob`, limbus centre from `segment`, both on
  the raw working image, carried into the anchor's raw coordinates through the mappings. Q1b is not available for I2
  on UPOL (no limbus) or for I3 and I4 (no structure left out of the fit); those cells are ranked on Q1a only.
- I3 and I4: the unwrapped image is 64 x 256; its P representation is the unwrapped image resized to 64 x 64 (BOX),
  z-scored, and its Q1c patch is the whole unwrapped image. I0 patch: the centred square (2D Amendment 1); I1, I2,
  I5, H1: the iris box of 2D Amendment 2.
- Q3c: per pair, one reference stranger drawn with the seed is passed through the candidate code path and scored
  against the reference without that person. Q1d: own candidate, the null stranger and the first five reference
  strangers of each pair. Q3d: the pair's reference stays the same strangers, re-registered and scored against the
  new anchor. Q3e: same-eye pairs only; the candidate array is reflected, the reflection is recorded and undone by
  exact index reversal before the method runs.
- Fellow-eye pairs: the raw arrays of the candidate, every stranger, every mean member and the null stranger are
  reflected before any method runs, as in 2D Amendment 1.
- ECC: grey float images, mask = joint valid region, `gaussFiltSize` 5; failure if OpenCV raises, or the update moves
  more than 10 px or rotates more than 5 degrees (224 grid for iris, 512 for GRAPE). G4 reuses the G3 result of the
  same pair: DIS flow from the G3 output to the anchor output, each component smoothed (sigma 8), failure if any
  Jacobian determinant of x + u(x) is at most 0 inside the joint field of view; Q1e on that field. G2: failure also
  when `estimateAffinePartial2D` returns nothing.
- N is standardised with the mean and SD over the anchors' outputs of the method. Four worker processes, OpenCV one
  thread each.

## Amendment 2 (2 Nov 2026): fairness on identity-free geometry only (Kate), before any quality metric was computed

Found while writing the evaluator: the vessel pattern is individual, so vessel-map Dice between the same eye's images
is high once aligned and between strangers' images is low under any registration, including none. Used as a fairness
criterion, it would mark every method as biased. Kate chose "geometry-only parity":

- Q3b uses only structures that carry no identity: Q1a (known-transform error) and the landmark position errors of Q1b
  (disc and fovea for fundus; pupil and limbus centres for iris; facial landmarks for faces). Where a landmark was used
  to fit the method (disc and fovea for G1, H1, H2; limbus for I1; pupil for I2), its error is zero by construction
  and only Q1a enters Q3b.
- The vessel Dice own-minus-stranger gap is reported descriptively and is not an eligibility criterion.
- Ranking (R4) uses Q1b on own same-eye pairs: vessel ring Dice for fundus (higher is better), pupil or limbus centre
  distance for iris (lower is better), whichever is held out for the method; methods without a held-out structure are
  ranked on Q1a.

## Amendment 3 (2 Nov 2026): crop coverage reported, before any quality metric was computed

Prompted by the parallel QC pilot (`docs/REGISTRATION_QC_PROTOCOL.md`, another session), which reported crops partly
outside the iris and low crop support in UTIRIS; taken as a reason to measure, not as a result. Added, descriptive:
the fraction of the Q1c patch with valid source support, per role, and the share of patches under 95% support (the
QC gate's threshold). On the canonical iris frame the 2D iris box reaches 87.5 px from the centre against a limbus
radius of 80, so about one eighth of its width lies beyond the limbus by construction; eyelid and lash cover inside the
box is not measured by any metric of this protocol and is left to the visual gate. No decision rule changes.

## Amendment 4 (2 Nov 2026): thresholds left open by Amendment 2, before any quality metric was computed

- Q3b on landmark position errors uses the same limit as Q1a: a method is biased if the 95% interval of the error
  difference (stranger or fellow minus own) lies entirely above 1 px on the 224 grid.
- Degrees of freedom for the tie rule: none 0; per-image frames and harmonisation on a frame (G1, H1, H2, I1 to I3)
  1; pairwise similarity or rotation (G2, G3, I4, I5) 2; deformable (G4) 4.
- Q3a also compares mean members with own candidates, under the same 5-point limit.

## References

1. Hernandez-Matas C, Zabulis X, Triantafyllou A, Anyfanti P, Douma S, Argyros AA. FIRE: fundus image registration dataset. Model Artif Intell Ophthalmol. 2017;1(4):16-28. doi:10.35119/maio.v1i4.42
2. Rublee E, Rabaud V, Konolige K, Bradski G. ORB: an efficient alternative to SIFT or SURF. In: 2011 International Conference on Computer Vision. IEEE; 2011. p. 2564-71. doi:10.1109/ICCV.2011.6126544
3. Fischler MA, Bolles RC. Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. Commun ACM. 1981;24(6):381-95. doi:10.1145/358669.358692
4. Evangelidis GD, Psarakis EZ. Parametric image alignment using enhanced correlation coefficient maximization. IEEE Trans Pattern Anal Mach Intell. 2008;30(10):1858-65. doi:10.1109/TPAMI.2008.113
5. Zhu JY, Park T, Isola P, Efros AA. Unpaired image-to-image translation using cycle-consistent adversarial networks. In: 2017 IEEE International Conference on Computer Vision (ICCV). IEEE; 2017. p. 2242-51. doi:10.1109/ICCV.2017.244
6. Cohen JP, Luck M, Honari S. Distribution matching losses can hallucinate features in medical image translation. In: Medical Image Computing and Computer Assisted Intervention (MICCAI 2018). Lecture Notes in Computer Science. Cham: Springer; 2018. p. 529-36. doi:10.1007/978-3-030-00928-1_60
7. Daugman J. How iris recognition works. IEEE Trans Circuits Syst Video Technol. 2004;14(1):21-30. doi:10.1109/TCSVT.2003.818350
8. Deng J, Guo J, Xue N, Zafeiriou S. ArcFace: additive angular margin loss for deep face recognition. In: 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE; 2019. p. 4685-94. doi:10.1109/CVPR.2019.00482
9. Avants BB, Epstein CL, Grossman M, Gee JC. Symmetric diffeomorphic image registration with cross-correlation: evaluating automated labeling of elderly and neurodegenerative brain. Med Image Anal. 2008;12(1):26-41. doi:10.1016/j.media.2007.06.004
10. Nyul LG, Udupa JK, Zhang X. New variants of a method of MRI scale standardization. IEEE Trans Med Imaging. 2000;19(2):143-50. doi:10.1109/42.836373
11. Yushkevich PA, Avants BB, Das SR, Pluta J, Altinay M, Craige C. Bias in estimation of hippocampal atrophy using deformation-based morphometry arises from asymmetric global normalization: an illustration in ADNI 3 T MRI data. Neuroimage. 2010;50(2):434-45. doi:10.1016/j.neuroimage.2009.12.007
12. Reuter M, Fischl B. Avoiding asymmetry-induced bias in longitudinal image processing. Neuroimage. 2011;57(1):19-21. doi:10.1016/j.neuroimage.2011.02.076
13. Johnson WE, Li C, Rabinovic A. Adjusting batch effects in microarray expression data using empirical Bayes methods. Biostatistics. 2007;8(1):118-27. doi:10.1093/biostatistics/kxj037
14. Fortin JP, Cullen N, Sheline YI, Taylor WD, Aselcioglu I, Cook PA, et al. Harmonization of cortical thickness measurements across scanners and sites. Neuroimage. 2018;167:104-20. doi:10.1016/j.neuroimage.2017.11.024
15. Rohlfing T. Image similarity and tissue overlaps as surrogates for image registration accuracy: widely used but unreliable. IEEE Trans Med Imaging. 2012;31(2):153-63. doi:10.1109/TMI.2011.2163944
16. Christensen GE, Johnson HJ. Consistent image registration. IEEE Trans Med Imaging. 2001;20(7):568-82. doi:10.1109/42.932742


## Amendment 5 (2 Nov 2026): one owner, protocol, comparison and visual gate

This is a post-result consolidation, authorized by Kate after merge `74c01f8`, not a fresh prospective comparison or an independent replication. `git pull` reported already up to date at that commit. Codex now owns `reg_eval.py`, `reg_methods.py`, `reg_run.py`, the registration QC utilities, and MRI registration. No previous method experiment is rerun merely to merge its evidence.

### Methods already compared and methods not run

The completed comparison is preserved byte-for-byte in `results/registration/comparison.md`, `.csv`, the 19 method aggregate JSON files, and `manifest.json`. Its recorded protocol hash remains the Amendments 1–4 hash; do not replace it with the Amendment 5 hash.

| Dataset | Completed methods | Intended comparisons per method before recorded drops | Saved selection outcome |
|---|---|---:|---|
| GRAPE | G0 centre crop; G1 disc–fovea frame; G2 ORB/RANSAC; H1 G1 + histogram matching; H2 G1 + camera standardisation | 437 | G1 recommended, tied with H1; neither is visually accepted yet |
| UTIRIS | I0 centre crop; I1 limbus frame; I2 pupil frame; I3 rubber sheet; I4 rubber sheet + rotation; I5/I1 ECC; H1/I1 histogram matching | 234 | No eligible method: not registrable under this protocol |
| UPOL | I0, I1, I2, I3, I4, I5/I2, H1/I2 | 192 | No eligible method: not registrable under this protocol |

GRAPE G3/G4 have no completed aggregate in this release: status NOT RUN/NO COMPLETED RESULT, not zero or failed. E1, face F0–F3, external FIRE evaluation, MRI harmonisation and halfway-space comparisons have not been completed here. An implemented method is not an executed comparison. Iris verdicts apply to these methods/settings only, not to every possible iris registration approach.

Additional QC work is part of this same study, but is **development evidence**, not another completed method-ranking experiment:

| Development method / iteration | Evidence | Status and relation to completed methods |
|---|---|---|
| Full-field SIFT similarity, original 13-row pilot | `qc/pilot_v1.json` | Separate from G2 (ORB); original failures retained |
| Fundus green CLAHE + eroded FOV SIFT v2 | `qc/development_latest.json` | Seven reviewed rows, no visual passes; NOT G1 and not evidence that G1 passed |
| Iris pupil/limbus development v2–v6 | `qc/development_latest.json` plus private snapshots | Six latest rows fail; post hoc pupil candidates/assumed concentric limbus, not a relabelled I1–I5 result |
| Legacy disc–fovea frame review sheets v1 | `qc/fundus_frame_v1.json` | Seven sheets prepared, none reviewed; same family as G1, but renderer uses `ladder2d_register.fundus_canonical`, while `reg_methods.g1` uses its own resampler. Exact mapping, interpolation and reflection agreement must be checked before transporting any review |
| QTIM original rigid/affine/SyN, centroid-initialized rigid, then corrected affine/SyN | `qc/mri.json`, private original/repair snapshots | Original gross displacement invalid; corrected one-scan gross anatomy improved; four current box/branch decisions unreviewable. No comparative population claim |

The legacy GRAPE renderer reflects the canonical fellow image; the comparison runner reflects the raw candidate before the method. These must not be assumed interchangeable without checking detector and sampling behaviour. The next G1 QC renderer must use `reg_methods.g1` and the runner's actual reflection convention, show both source and output mappings, and receive a new run ID. Previous sheets and their hashes stay historical.

### One result set, without merging incompatible denominators

- `manifest.json` / `<dataset>_<method>.json` / `comparison.*`: immutable completed method-comparison evidence for Amendments 1–4. Existing c and downstream fields are historical, provisional diagnostics, not accepted identity findings.
- `qc/`: migrated original QC aggregate files, unchanged bytes. The private source images, row manifests, transforms, review records and run snapshots stay at their existing paths; moving them would break provenance.
- `status.json` and `status.md`: generated current view built by `python src/reg_status.py`, combining the completed comparison, method-specific QC readiness, preserved development history and MRI status. This view never sums review decisions across different methods or promotes computational eligibility to visual acceptance.
- `src/metrics_board.py` reads that same result tree. No new top-level `results/registration_qc_*.json` is written.
- `reg_methods.py` is the common implementation for named G/I/H methods; `reg_eval.py` retains the frozen comparator logic. `registration_qc_*` files are review/rendering and explicitly labelled development stages within this protocol, not competing studies. Historical parameter-search scripts must not be reused as confirmatory methods without a dated amendment and frozen remaining-case evaluation.

The comparison denominators (437/234/192) and expanded visual denominators (755/390/320) differ because same-eye reflected-input controls are additional visual rows. Dropped computational rows remain in the gate's intended denominator. Original quality summaries retain their documented successful-case denominators; their exclusions must not shrink visual denominators. No combined accuracy or visual percentage is computed across these two populations.

### Current execution boundary

Only registration, geometry/segmentation checks and visual review may run until the gate passes. Q3c–Q3e in their calibrated-c form and Q4/downstream scoring remain paused; coordinate/reflection checks without identity scores are allowed. `reg_run.py run` includes scoring and therefore is blocked under this amendment, including smoke runs. `reg_run.py status` is aggregate-only. `prep` is preparation, not acceptance. Recomputing the completed `report` must not overwrite this frozen release; future result-generating amendments require a new versioned output directory.

Do not restart completed QTIM jobs. The repaired one-scan output is ready for native-to-template/crop/composite-transform audit. Its nonpositive nonlinear Jacobian count of zero does not validate the full composed transform. Freeze the MRI cohort and repeat/contralateral strata before expansion; no MRI gate denominator exists yet. All acquisition/selection details in `PROTOCOL_real_cases.md` remain inputs to this registration gate, not an alternative gate.

### Integrated visual QC requirements

The following criteria and review population are carried forward from the pre-existing prospective QC protocol; this consolidation does not relax them.

#### Fixed review population

Use the existing hash-frozen GRAPE and iris selections in `PROTOCOL_2d_ladder.md`, without reselection, numerical-QC exclusions, replacement or ranking by image quality. A comparison is the unit; repeated people are not independent participants.

2D gate set: every frozen own-rung pair, plus a deliberately reflected-input control for every same-eye pair. The control horizontally reflects the moving input and requires explicit inverse reflection before estimating a proper similarity transform. It is a coordinate-system test, not a fellow-eye comparison. Fellow-eye comparisons remain a separate anatomical-correspondence stratum, with the full moving image reflected before crop extraction. Natural bilateral anatomy need not have identical fine texture.

Stratify GRAPE by all four original rungs, with reflected-input controls separately for each of the three same-eye rungs; stratify UPOL and UTIRIS separately by same-eye, reflected-input same-eye and fellow-eye. Preserve every intended pair in its denominator. Freeze the private expanded manifest and publish its hash and aggregate counts before image inspection. Until that manifest exists, denominator status is NOT FROZEN, not zero.

MRI gate set: all targets in the original frozen design, separately for the deep and parietal boxes; template alignment, available repeat correspondence, and reflected contralateral correspondence each have their own strata. Freeze exact intended rows from the existing selection before inspection. Retain unavailable repeat rows explicitly; do not confuse absence of a planned repeat with a registration failure. Expanded A1.1 does not become eligible through the original-design gate. Donor/pool images require their own QC before any future scoring, even if the target gate passes.

Development pilot: first pair in saved selection order in each 2D stratum, including its reflected-input control; the first existing completed SyN target for both MRI boxes, once available. Keep all pilot failures. These are development cases, not independent validation. Freeze a remaining-case validation manifest before tuning; never substitute an easier pair for a failed pilot. If a historical problematic mirror example can be identified, add it as a separately labelled regression case without changing gate denominators.

#### Required private review sheet

Each sheet shows original full frames with orientation labels; exact anchor crop and inverse-mapped moving crop boundaries; registered full frames; both crops at equal display scale; a 50:50 blend; checkerboard; independently visible landmarks or estimated contours. State coordinate convention, image dimensions, transform direction, reflection order, interpolation and valid-support mask. Show original and registered views together. MRI requires axial, coronal and sagittal views through both boxes, physical coordinates, brain contours, and deformation diagnostics. Never replace the full field with a crop-only figure.

Use one display window for each pair and record it; do not separately tune contrast to hide mismatch. Pupil/limbus contours and automated feature matches are estimates, not ground truth. Review local correspondence independently of the points used to fit registration.

#### Decisions recorded before review

PASS requires all of the following:

1. Correct side/orientation, transform direction and reflection; crop boundaries map back to the intended source anatomy with no off-by-one displacement.
2. At least 95% of each intended crop has valid source support; visible occlusion, padding and segmentation error are shown and do not conceal correspondence.
3. Gross anatomical landmarks and crop placement agree. For same-eye repeats, at least three distributed identifiable local features agree to within two pixels on the 224-pixel display grid; failure to identify reliable features is UNREVIEWABLE, not PASS. Iris pupil dilation is recorded, not removed by an unconstrained deformation.
4. Fellow-eye and MRI contralateral views cover homologous anatomical regions without requiring identical individual texture or folding. The reviewer records the landmarks used and any residual anatomical difference. A missing homologous region is FAIL; uncertainty is UNREVIEWABLE.
5. No implausible local warping, folding, severe stretching or loss of individual anatomy. MRI negative/nonpositive Jacobians fail; positive Jacobians alone cannot establish visual acceptance. Global correlation and successful execution cannot satisfy any visual criterion.

FAIL means an observable violation; UNREVIEWABLE means insufficient evidence or an unavailable required output; NOT REVIEWED means no visual decision yet. Store reviewer, timestamp, reasons and sheet hash privately. Keep automated flags, agent review and Kate's acceptance as separate fields. No automated field may set visual PASS.

#### Counting and release rule

Report intended, reviewed, pass, fail, unreviewable and not-reviewed counts by dataset, original rung, reflection branch and MRI box. Reviewed = pass + fail + unreviewable. Visual acceptance fraction = pass / intended (raw proportion, higher is better); unreviewable and not-reviewed rows do not disappear. Missing manifest or no reviews gives NOT ESTABLISHED, not an observed 0% success rate. Report computational preparation failures separately from visual failures.

Require at least 90% visual acceptance overall AND in each prespecified stratum, including each mirror stratum, with no unresolved systematic orientation/crop error. Report pilot and remaining validation results separately. Kate's visual acceptance remains explicit. A small successful pilot does not release the metric pause. Any criterion change is dated and all affected cases are re-reviewed under a new version while preserving previous records.


### Integrated disc–fovea review specification

The following anatomy/crop requirements are retained from the pre-existing frame amendment. Its earlier resampling/reflection implementation is historical; the common-method/mapping check above takes precedence for new sheets. Arcade boxes are exploratory anatomy controls, not a replacement for the existing metric crop or permission to expand into an inpainting experiment.

**Landmark check comes first.** The frame is defined by two detected points, so each image is reviewed
for:

- disc: the marked point lies inside the visible optic disc rim;
- fovea: the marked point lies within 0.5 disc diameters (about 30 pixels at 512) of the visible foveal
  centre.

If the fovea is not visible (for example a tessellated, myopic macula), the row is UNREVIEWABLE, not
PASS. A wrong landmark is FAIL for every comparison that uses that image. Bright peripapillary atrophy
next to the disc is a known risk for the disc detector and is checked explicitly.

**Three boxes per comparison, labelled and never swapped:**

| Box | Canonical 512 frame (x0, y0, x1, y1) | Role |
|---|---|---|
| `metric` | 183, 183, 329, 329 | The existing metric mask. Required for any 2D-ladder scoring. |
| `arcade_sup` | 208, 72, 304, 168 | New. Superior temporal arcade, about 2 disc diameters above the disc-fovea line. |
| `arcade_inf` | 208, 344, 304, 440 | New. Inferior temporal arcade, mirror of `arcade_sup`. |

The arcade boxes are placed from anatomy alone: disc-fovea distance 149 px, about 2.5 disc diameters, so
one disc diameter is about 60 px. They were fixed before any image of this run was generated. They are
candidates for the inpainting paper's mask. They do not replace the metric mask, and passing them does
not release metric scoring on the metric mask.

**Review sheet.**

- Row 1: anchor and moving source images with the detected disc (circle) and fovea (cross), and all three
  boxes mapped back to source coordinates; then both canonical frames with the boxes and target points.
- One row per box: anchor crop, moving crop, vessel overlay, checkerboard.
- Vessel overlay: green-channel black top-hat with identical parameters on both images; anchor red,
  moving green, overlap yellow. This is an aid to inspection, not a landmark.

Same display window for both images of a pair.

**Artifact check.** A small structure at the same sensor position in images of different eyes or people
is treated as an optical artifact, not a retinal landmark, and is not counted among the three local
features.


### Current state and next work

The historical latest 2D development review has 13 reviewed rows: 0 pass, 6 fail, 7 unreviewable, with 1,452 unreviewed in its 1,465-row manifest. These are SIFT/iris-development decisions, not reviews of the newly selected G1. The G1-family frame sheets have seven prepared rows and no visual decisions. QTIM has four unreviewable decisions on one repaired scan, with no frozen full MRI denominator. Gate NOT MET; Kate's acceptance, agent review and automatic flags remain separate.

Next: (1) reconcile G1 mapping/interpolation/reflection with the canonical QC renderer and review its fixed development rows; (2) independent iris pupil, limbus and occlusion geometry on the same failed rows, retaining the I0–I5/H1 comparison as the baseline; (3) full QTIM transform/crop audit and frozen MRI manifest; (4) remaining-case validation per stratum. Do not repeat completed method comparisons without a discriminating implementation or protocol change. The additional-dataset screen in `REGISTRATION_DATASET_SCREEN_2026-11-02.md` remains supporting acquisition evidence, not a new gate.

### Execution checkpoint, 3 Nov 2026 (criteria unchanged)

The shared-G1 development pilot was rendered and inspected: seven rows, zero pass, six fail and one unreviewable. `results/registration/qc/fundus_frame_v2.json` is separate from legacy v1 and the SIFT/iris development history. Same-eye disc/fovea localization failures prevent release; Kate review remains separate.

The full MRI fixed-physical-to-native-physical pull composition was audited on the existing repaired scan, including the affine component. Saved-image replay was exact, full Jacobians were positive in the checked valid brain region, and inverse consistency was measured; none of these checks assigns anatomical acceptance. See `results/registration/qc/mri_composite_v1.json` and its frozen private plan. No registration was optimized again.

The existing original QTIM selection was expanded into a private immutable QC manifest without new image inspection: hash `ffb5a2cf66010e7071c1054ba08690a0d5e3eb01ac57c9ddd7c52b8168682461`, aggregate `results/registration/qc/mri_manifest_v1.json`. Required comparisons total 334: template 72, contralateral 72, repeat 23 **per box**. Ninety-eight unplanned set-S repeat rows remain explicitly unavailable, outside those required denominators. The 72 target people represent 68 families, with four shared families across sets; person/family clustering must preserve that dependency. The exposed pilot target's six required rows are development; the other 328 are remaining-case validation. This fixes the review population, not acceptance or cohort-wide preprocessing readiness. Donor/pool QC is still required separately for all selected source scans.

No acceptance criterion, crop, selected participant, completed method comparison or metric pause was changed. Next: anatomical landmark repair and local MRI anatomy review before validation expansion.

## Amendment 6 — modality-specific pretrained development pilots, 3 Nov 2026

Kate explicitly requested contemporary registration methods appropriate to each modality and continued QC, with no table redesign. The prior G0–G4/I0–I5 comparison is a classical-baseline comparison, not a comprehensive SOTA evaluation. All its outcomes and all observed development failures remain preserved. No identity or synthesis-model scoring is authorized here.

Before new model outputs are inspected, freeze these bounded development candidates on the SAME existing pilot rows:

- **GRAPE / G5:** official SuperRetina ECCV 2022 pretrained retinal keypoints/descriptors, commit `338f041cc2ce86f39623e7da950b14f33bbc25df`, checkpoint SHA-256 `df9d28098d1d95347f96c6a2a425db3f9ab1683778173a22eca96f0e863758cf`. Official 768×768 input, green-channel preprocessing, NMS radius 10 / probability 0.01, descriptor ratio 0.9, LMEDS homography. A separately labelled G5S sensitivity uses the SAME features with the existing proper-similarity RANSAC criteria (12 inliers, 3 px at 512, scale 0.8–1.25, rotation <20 degrees); it is not the original SuperRetina endpoint. Reflect fellow input before feature extraction; reflected same-eye controls are explicitly reflected back. No nonlinear retinal warp in this pilot. Both fits remain candidates requiring independent local landmarks and later candidate-role fairness, not automatic visual acceptance.
- **Iris / I6 segmentation feasibility:** official Open-IRIS UNet++/MobileNetV2 ONNX semantic segmentation, code commit `ba232f566ee0f244c503bd8ce3213200c369927c`, model revision `e6c8fbb8e2e7e024a58818530a6e3ae5c005f47a`, SHA-256 `1228a1c760791749fceeedf9322e24f098aac80234368e1beee156e1e8fa57e7`. Reproduce code preprocessing: grayscale input resized to width 640 / height 480, three channels, /255 and ImageNet channel normalization, no denoise; four independent probability masks thresholded at 0.5 and nearest-neighbour resized to source. Preserve raw probabilities privately. Model is intended for infrared; UPOL visible-light application is an explicit out-of-domain feasibility check, not a validated use. This first step supplies pupil/iris/eyelash geometry only: it does not move the fixed crop, count invisible tissue as support or assign registration PASS.
- **QTIM / M2:** FreeSurfer 8.0.0 bundled SynthMorph joint affine/deformable model on the original completed pilot target and the same symmetric template; CPU, two threads, extent 256, integration steps 7, regularization 0.5. Save both transforms and both warped images, exact container digest, model hashes, input/code/protocol hashes and elapsed time. No duplicate SyN job. Subsequent resampling, physical reflection, full Jacobian/native-anatomy checks and target/repeat manifest remain mandatory.

Additional primary-source candidates inspected for relevance, NOT executed or declared superior here: RetinaRegNet (2025, diffusion features, homography + polynomial deformation), HybridRetina (2024, global/local alignment), BrainMorph (2025, rigid/affine/nonlinear keypoints), EasyReg, uniGradICON/multiGradICON; iris IrisParseNet (TIFS 2020) and CVRL's 2025/2026 IREX segmentation/circle models. Version/weight/resource/training-overlap checks must precede their runs. SuperRetina is a reproducible specialized starting baseline, not a claim to be the latest or best retinal method.

For this GRAPE adapter, both inputs first use the exact shared G1 frames so the existing anchor metric crop remains fixed; SuperRetina estimates residual alignment onto that anchor. Record G1 failure, never fall back silently. This is a SuperRetina-feature development adapter, not reproduction of the FIRE benchmark pipeline. Full source/G1 context and inverse-mapped candidate crop remain visible.

Training overlap is not established for every candidate; do not label the runs independent benchmark validation. External FIRE landmarks are useful but cannot replace the failed GRAPE cases or silently exclude the FIRE pair noted in upstream documentation. Learned alignment can erase individual anatomy: appearance gain or published leaderboard rank is not a release criterion. Per-image geometry, same-eye fitting, contralateral homology and cross-person candidate-role fairness remain distinct.

The gate, fixed metric crop, all stratum denominators, development/validation distinction and failure/unreviewable handling are unchanged. Run feasibility on the exposed pilots first, inspect outputs, then specify the validation adapter before expanding. Every run uses a new immutable private folder, code/input/model hashes and aggregate-only public output. No manuscript or harmonization-scorecard work.

Primary sources: https://github.com/ruc-aimc-lab/SuperRetina ; https://github.com/mirthAI/RetinaRegNet ; https://arxiv.org/abs/2409.01068 ; https://github.com/worldcoin/open-iris ; https://huggingface.co/Worldcoin/iris-semantic-segmentation ; https://github.com/xiamenwcy/IrisParseNet ; https://github.com/CVRL/OpenSourceIrisRecognition ; https://w3id.org/synthmorph ; https://github.com/alanqrwang/brainmorph ; https://github.com/uncbiag/uniGradICON .

### M2 runtime diagnostic, before a second output is inspected

The first bundled SynthMorph joint call failed in affine spatial sampling (invalid TensorFlow gather index), with no registered image. Inputs contain finite values, but the saved N4 image has a maximum of 14,000,000. The cause is not isolated; do not call this an anatomical failure or a SynthMorph accuracy result. Preserve v1. A separate v2 uses the SAME original scan before N4, consistent with SynthMorph's documented no-preprocessing input, with identical model/registration settings. Verify its raw scan hash from the original scan manifest. This is an input/preprocessing sensitivity, not a replacement of the failed attempt. Also repair the auxiliary model-hash inventory invocation to use the container's explicit bundled Python path; this does not change registration.

V2 raw-input joint registration exited 137 after 170 seconds with no registered output under the 10 GB container memory limit; the process was killed, and memory pressure is suspected but not independently established as the cause. A bounded v3 uses the documented extent 192 option with all other settings unchanged, raw input and the same 10 GB/two-thread limits. It is a resource/registration-space sensitivity; require explicit coverage checks because reduced extent may crop anatomy. Preserve both previous failures. No anatomical verdict follows from either runtime failure.

## Amendment 7 — paper-derived methods before further variants, 3 Nov 2026

Kate requested recent venue evidence rather than improvised registration/normalization. Pause further custom adapters and parameter variants until the exact published preprocessing, modality/contrast, comparison role, release/checkpoint version, evaluation independence and compute requirements are mapped to this study. `REGISTRATION_2026_PRACTICES.md` records the inspected 2026 methods, caveats and study-specific inferences; `REGISTRATION_2026_SOURCES.json` preserves PDF URLs/hashes. Unavailable code or unmatched modality must remain explicit. None of the inspected papers validates our existing numerical acceptance thresholds by itself. This amendment changes the method-selection process; it does not alter the frozen gate, crops, population, previous decisions or denominators. Existing pilots remain development evidence. The already-running SynthMorph v3 completed with outputs, visual acceptance NOT ESTABLISHED; no new variant was launched in the literature pass. Identity/synthesis/denoising metric experiments remain paused.

## Amendment 8 — authorized sequential fixed-method execution, 3 Nov 2026

Kate authorized fixed methods one after another following the literature screen. Scope remains registration/QC. Each run saves its plan before execution; this amendment records that authorization and links those prospective run plans, rather than retrospectively claiming they were preregistered here. Use the same exposed raw T1w/template pair; do not expand the validation cohort or tune after inspecting outputs.

1. Official installed FreeSurfer SynthMorph **rigid then affine**, CPU two threads/10 GB, default extent 256, same model inventory/container/input hashes; save both LTA transforms and resampled directions. Private `registration-qc-fixed-mri-20261103-v1/plan.json` was saved before either run. Reuse completed joint extent-192 output without rerunning it. Extent differs, so this is a bounded feasibility comparison, not a controlled model ranking. SynthMorph is a published 2024 baseline, not a 2026 venue method.
2. Official **BrainMorph 256-keypoint S, affine**, chosen from the official README's example and default affine CLI, commit `de606375c032a19d109f9bfe944dbd6ae321e707`; checkpoint SHA-256 `1d4b0e39aba9de26a21312353f9cb397cb077a4e5935799ac3944f71df32e8cd`. Published preprocessing is unmodified canonical orientation, 1mm resampling, 256-cube crop/padding and min-max intensity scaling. CPU two threads, 1800-second timeout; empty metric list, saved images/geometry only. BrainMorph is a MELBA 2025 baseline, not a 2026 venue reproduction; training overlap with QTIM is unestablished. Preserve failed starts separately from registration-accuracy evidence.

BrainMorph release compatibility: PyPI KeyMorph 2.0.1 lacks an imported symbol; official source at `dcb799622b2b60877dad27e9705ae6408cdb491c` provides it. The legacy checkpoint also contains `argparse.Namespace` and Linux `PosixPath` metadata. A separate third attempt keeps weights-only loading, allows Namespace and maps only metadata PosixPath to PurePosixPath, without changing weights or upstream registration code. All three run plans, environments, logs and snapshots remain separate. These repairs are software compatibility changes, not alternative registration configurations.

Inspect full-field anatomy and both fixed crop boxes for each successful method, distinguish fit evidence from independent anatomy, and audit physical direction/inverse/FOV where available. New method/box observations are not extra participants or replacements for prior failures. No numerical release threshold, reflection handling, original cohort, gate denominator or scoring pause changes. Methods lacking code/weights remain unavailable instead of being recreated ad hoc.

## Amendment 9 — fixed-cohort MRI QC preparation, 3 Nov 2026

After discussing the completed pilots, Kate authorized progressing on the fixed cases rather than adding models. First full-cohort candidate is **SynthMorph affine**, with exactly the pilot's raw input, official bundled affine weights/container, extent 256, two threads and 10 GB memory. This is a per-image canonical alignment to the fixed symmetric template, with no local deformation or per-case candidate switching. Selection reflects a bounded canonical baseline appropriate for evaluating retained individual anatomy; no pilot score or anatomical superiority is claimed. Other completed methods remain comparators, not automatic fallbacks. Do not tune on validation failures.

Freeze all 95 required source scans (72 original targets at session 1 and 23 planned repeats at session 2), in original selection scan order, linked to the existing 334-row required comparison manifest. The 98 unplanned repeat rows remain unavailable; no subject, failed case or family is substituted. Donor/pool QC remains separate and is not released by this target-cohort preparation. Reuse the exact completed affine pilot output only after verifying hashes. Save an immutable private plan/code snapshot before any new registration; keep computational failures distinct from visual decisions and retain every required row.

Run one registration at a time, save both images and RAS transforms, and prepare native/full registered context, inverse-mapped crop outlines, both crops, blend and checkerboard for each box/branch. Repeat images use the target's display window; template uses its own fixed P99. Template brain contours are display context, not independent native anatomy labels. Contralateral comparison is exact x reversal on the symmetric 1mm template grid after individual alignment. No mirrored image is refitted to erase asymmetry. Check both target and comparison source support, positive affine determinants, grid agreement and inverse consistency; none sets automatic visual PASS. Prepared sheets remain NOT REVIEWED until actual inspection with recorded reasons and hashes. The original per-stratum 90% gate and Kate's explicit acceptance are unchanged.
