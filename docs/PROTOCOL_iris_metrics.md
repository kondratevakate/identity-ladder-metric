# Protocol: metric panel on the iris identity ladder (7 Nov 2026; frozen before any iris metric result)

Iris replication of the fundus panel (general protocol Amendment 9.14, `fundus_ladder_metrics.py` in the
identity-ladder-bilateral worktree) and of the face panel (`PROTOCOL_face_metrics.md`, Amendment F1 scale), as the
fourth modality of the CVPR 2027 failure-mode audit. Kate's choices of 7 Nov 2026 ("Да, по умолчанию"): strip
frame, UTIRIS NIR primary, UTIRIS visible light and UPOL secondary after the segmentation fix, cohort metrics
unavailable, near-duplicate repeats reported and not excluded, per-image masks without fixed sectors.
Status: pre-QC. No metric is read before Kate's verdict on the frame QC sheets (visual-qc-gate).
Earlier iris work (`iris_codes.py`, `iris_calibration.py`, results of 1 to 3 Nov) is not changed and not reused as
evidence; only its table readers and the integro-differential operator are imported.

## Data

- UTIRIS V.1 (Hosseini et al. 2010, doi:10.1109/TIM.2009.2037996): 79 people, NIR (ISG Lightwise, BMP) and visible
  light (Canon EOS 10D, JPEG), about 5 images per eye per sensor, both sensors in one sitting. NIR files whose
  file-name eye differs from their folder eye are excluded and counted (`iris_calibration.utiris_table`).
- UPOL (Dobes et al. 2004, doi:10.1078/0030-4026-00388): 64 people, 3 images per eye, one optical device, visible
  light, black circular frame around the eye.
- No sex, age or kinship metadata in either set. Redistribution is not granted: images, strips, rows and QC sheets
  stay in `C:\Projects\data\projects\identity-ladder-metric\iris-ladder-metrics-20261107-v1\`; `results/` holds
  aggregates only, with no file name, subject number or image.
- Sets analysed and reported separately: `UTIRIS_NIR` (primary), `UTIRIS_VIS`, `UPOL` (secondary).

## Frame, preprocessing P1 (`src/iris_ladder_metrics_frame.py`)

1. Segmentation input: `iris_codes.load_grey` unchanged (red channel of colour images, grey otherwise; width 320 px;
   5 px median). Specular spots are replaced by the local 25 x 25 px grey minimum for segmentation only: a spot is
   a connected component of the white top-hat (25 px disk) above 50 grey levels whose peak reaches 120, closed
   (3 px), holes filled, grown by 5 px.
2. Circles. NIR and UPOL: pupil first. Candidates are disk-like dark components (fill >= 0.6 of the bounding circle,
   holes filled, area 0.15% to 25% of the image, not touching the image border or the UPOL black frame zone) at
   thresholds lo + q (median - lo), q in {0.15, 0.25, 0.35, 0.5}, after an opening with an 8 px disk (cuts lashes),
   plus, at each threshold, the deepest point of the dark mask (maximum of its distance transform, radius = that
   distance), which survives a pupil merged with the lash line.
   Each candidate is refined by the integro-differential operator (centres +-3 px, radii 0.7 to 1.3 of the blob
   radius); the pupil is the refined circle with the largest ratio of the mean grey of its ring (1.1 to 1.5 r) to
   1 + the mean grey of its disk (0 to 0.9 r). UPOL only: candidates within 0.2 of the width from the centroid of
   the field of view (the device centres the iris). Limbus: integro-differential operator on the lateral sectors
   (+-30 deg around 0 and 180 deg), centres +-6 px around the pupil centre, radii 1.5 to 5 pupil radii (UTIRIS) or
   0.25 to 0.48 of the width (UPOL, fixed device geometry). UTIRIS visible light: limbus first (largest disk-like
   dark component at q in {0.5, 0.6, 0.7, 0.8} after a 12 px opening, refined on the lateral sectors at 0.75 to 1.3
   of its radius), then the pupil by the operator inside it (centres within 0.1 of the limbus radius, radii 0.15 to
   0.7 of it).
3. Numeric QC (exclusion, counted per reason): pupil/limbus radius ratio in [0.15, 0.75], centre offset <= 0.25 of
   the limbus radius, valid share of the strip >= 0.40. Then Kate's visual verdict on the QC sheets.
4. Strip: Daugman rubber sheet, 64 radial x 256 angular samples, radial share 0.05 to 0.95 between the two circles,
   bilinear samples of the full-resolution channel after a Gaussian anti-alias blur with SD half the radial sampling
   step (a Gaussian, never a box). Angle counter-clockwise from the image +x axis with y up; column 0 = 0 deg.
5. Masks per image (missing, never zero): samples outside the image or within 3 SD of the UPOL black frame. Robust
   z (median, 1.4826 MAD) against the iris itself: the lateral sectors (within 30 deg of 0 and 180 deg) and the inner
   three quarters of the radius. Reflections: z > 4, grown by 1 sample; occlusion: |z| > 3; eyelid: in the upper
   and lower 130 deg sectors (25 to 155 and 205 to 335 deg) every sample outward of the first run of 4 radial rows
   with |z| > 2.5, plus 3 rows of lid edge inward of that run. No fixed sector is removed (the 1 Nov iris code removed fixed eyelid sectors; that crop is not used).
6. Grey level z-scored inside each strip's own mask (as faces).
7. Fellow eye: the other eye's strip mirrored about the vertical image axis, which on the strip is the exact column
   re-index theta -> 180 deg - theta (tested against unwrapping the flipped image).
8. No pairwise registration and no rotation search between images (head roll and cyclotorsion stay in the data,
   as acquisition, like pose in faces).

Development note: the segmentation and mask rules above were tuned on segmentation pass rates, drawn circles and
masks drawn on strips only (numeric pass on every 4th image: UPOL 22% -> 94%, UTIRIS NIR 94% -> 99%, UTIRIS visible
light 47% -> about 74%; the strip-mask reference moved to the lateral sectors after the QC sheet showed unmasked
upper lids); no strip comparison between images and no metric was computed during tuning. Any later change is a new preprocessing version (P2, ...) and goes back through the QC gate.

## Amendment P2 (7 Nov 2026, before any metric; replaces the P1 segmentation and masks)

Kate rejected P1 on its QC sheets ("стрипы очень плохо и ландмарки плохо"): limbus circles off the iris in UPOL
and pupil circles on the ring reflex. Kate: take a state-of-the-art model. Search (7 Nov): no iris-trained YOLO
weights exist (Ultralytics weights are COCO); two published iris models with open weights were downloaded with
Kate's approval ("давай действуй") to `C:\Projects\data\projects\identity-ladder-metric\iris-models-20261107\`:
- Worldcoin open-iris semantic segmentation (Hugging Face `Worldcoin/iris-semantic-segmentation`, UNet++ with a
  MobileNetV2 encoder, ONNX, 56 MB; classes eyeball, iris, pupil, eyelashes; model card: trained on 10 NIR
  research sets, NIR input; licence files MIT and Apache 2.0 in the repository).
- Notre Dame CVRL HDBIF segmentation (github.com/CVRL/OpenSourceIrisRecognition; NestedSharedAtrousResUNet mask
  and ResNet-18 circle regressor; trained on NIR sets and UBIRIS v2; research use with citation of Trokielewicz,
  Czajka and Maciejewicz 2020, doi:10.1016/j.imavis.2019.103866; Khan, Flynn and Czajka 2026, arXiv:2605.20735).
Trial on 6 evenly spaced images per set (private sheets `trial_models_*.png`): both correct on UTIRIS NIR; on UPOL
the CVRL mask cut out large parts of the iris and its limbus circle overshot on some images; on UTIRIS visible
light the CVRL limbus overshot on 2 of 6. Worldcoin followed the limbus, pupil and eyelids on all 18 images,
although its card states NIR input (the red channel of colour images is used, as for CVRL). Decision: Worldcoin is
the P2 segmenter for all three sets; the CVRL circles are an independent check on UTIRIS NIR (agreement reported).

P2 frame (`src/iris_ladder_metrics_frame.py`, version passed explicitly; weights checked by sha256):
1. Input: red channel of colour images, grey otherwise; padded with grey 127 to 4:3 and resized to 640 x 480
   (CVRL convention); padding cells are missing in the model output.
2. Circles: iris = largest component of predicted iris OR pupil (holes filled); pupil = the predicted pupil
   component inside it whose centroid is nearest to the iris centroid (holes filled; P2 first used the largest
   component, and Kate saw a dark crypt taken as the pupil on one UPOL image); pupil boundary points that touch the
   iris give the pupil circle. Iris boundary points
   that touch the predicted eyeball outside the iris (sclera) are limbus points; points along an eyelid are not
   used. Algebraic least-squares circle with two refits dropping points beyond 3 median absolute residuals.
3. Numeric QC (exclusion, counted): radius ratio in [0.10, 0.80] (UPOL pupils are small), centre offset <= 0.25
   limbus radii, limbus points on >= 20% of 10-degree angle bins, valid strip share >= 0.40.
4. Strip as P1 (64 x 256, radial share 0.05 to 0.95, Gaussian anti-alias, bilinear samples of the full-resolution
   channel). Mask = predicted iris AND NOT pupil AND NOT eyelashes (probability 0.5), minus specular reflections
   (robust z > 4 against the visible-iris samples, grown by 1 sample), minus samples outside the photograph or on
   the UPOL frame, minus a lid edge: strip samples within 3 samples of a predicted non-iris or eyelash sample in
   the outer three quarters of the radius (the QC sheet showed an unmasked dark band of lid margin and lash shadow
   next to the predicted lid), minus lid shadow: samples with robust z < -4 against the visible iris that are
   connected (8-neighbour) to that lid or lash region; isolated dark crypts are kept. Development on 6 NIR people
   (60 images): a threshold of z < -2.5 masked most of the upper iris, which is evenly shadowed by the lid but keeps
   visible texture, so -4 was fixed; the shadowed upper iris stays in the data as an acquisition effect (it is the
   same in the repeat image). No intensity-based eyelid rule (the model's iris class stops at the lids).
5. z-score inside the mask, fellow mirroring, units and everything below unchanged.
P1 code stays in the module for the record and refuses to run the panel (`REJECTED`).

## Units, rungs and fills

- Unit = one eye of one person in one set. Target = its first image in path order that passes segmentation QC;
  repeat = its second such image. Not chosen by similarity or quality (STUDIES-025).
- Pool for a target: the target images of the same-side eyes of all other people (leave-person-out, asserted).
- Coverage `valid`: strip samples valid in >= 90% of the units. Window positions: 16 x 16 on a 16 px grid with
  >= 95% of the window and >= 50% of its 8 px ring in `valid`; no window or ring crosses the 0 deg seam. A target
  enters a position if >= 50% of the window is in its own mask and `valid`. Scored pixels E = window AND target
  mask AND fill validity AND dataset-mean validity AND `valid`; a fill is scored only if E covers >= 50% of the
  target's window pixels.

| Block | Rung | Definition |
|---|---|---|
| same eye | repeat | the eye's second image (same sensor, same sitting in UTIRIS) |
| same person | fellow | the other eye's target image, mirrored |
| selected by context (achievable) | look-alike, mean of 16 look-alikes | pool eyes nearest on the 8 px ring (masked mean squared difference, >= 50% shared), as fundus |
| no selection | stranger, mean of 2, dataset mean | seeded permutation of the pool (seed 20261102 + target), masked means |
| oracle (unreachable) | oracle look-alike | pool eye nearest on the window itself |
| controls | wrong detail, white noise | as fundus (`BM.wrong_detail`, `BM.noise_fill`) |
| controls | repeat / fellow pulled to the mean | mean + lambda (donor - mean), lambda = 1 - ratio / 2, ratio = whole-strip squared error of that rung over the dataset mean's, summed over the set's targets on pixels valid in target, donor and mean (computed in the same run before any window score) |
| not available | relatives, sex-age cohort | no metadata |

Strangers for the anchor and calibration: the first 16 of the permutation (str00 to str15); stranger = str00.

## Metrics

The fundus registry of 26 confirmatory metrics, scored with the fundus and brain functions imported unchanged by
explicit path (sha256 of both sources stored in the JSON). Computed (10): MSE, RMSE, PSNR, MAE, SSIM, MS-SSIM (3
scales, context clipped to the strip), ACC vs dataset mean, amplitude deviation vs dataset mean, lDDT-like,
intensity persistence (H0 sublevel + superlevel Wasserstein). Unavailable (2): ACC vs matched mean and amplitude
vs matched mean (no cohort metadata). Blocked (14): LPIPS (no ImageNet weights), Betti matching x2 (library),
identity x4 (no deep iris encoder weights; the classical iris code is not an embedding), latent x3 (no generative
iris model), reference-free x4 (set-level, no per-target value). Supplementary as fundus: PSNR fixed, Pearson,
TM-like, GDT-like, lDDT_rep, RMSE_rep (sigma_rep per position from repeat pairs of other people, >= 5 pairs).
Identity reference (ArcFace for faces): not run in v1; the 1 Nov iris-code medians stay descriptive.

## Scales, tests and statistics (as fundus)

- Spec block (`ladder-result-spec/1`): per rung, summed squared error over the dataset mean's on the same pixels
  (1 = mean, 2 = stranger in theory), person bootstrap; fellow reported in both directions (R eye and L eye as
  target). Hard checks raise; soft flags (stranger outside 1.85 to 2.15, directions differing by more than 15%) are
  printed and stored.
- Panel: R_self = 0 for the target itself, 1 for the 16 strangers; R_repeat also reported (repeat = 0). Person
  bootstrap, 2000 resamples, seed 20261102.
- Tests per metric, reference = repeat: T1 repeat better than dataset mean; T2 repeat better than repeat pulled to
  the mean; T3a repeat better than wrong detail; T3b repeat better than white noise. One-sided Wilcoxon on person
  means (>= 10 people), Holm over 26 x 4 hypotheses with blocked or unavailable ones kept as p = 1. Also with the
  fellow eye as reference (descriptive). Calibrated scores (z, EVT, P(same), c) as fundus.
- Expectations (fundus set): E1 fellow closer than stranger; E3B dataset mean and mean of 2 better than stranger,
  averages ordered by k; E5B wrong detail not better than stranger (tolerance 0.1); E6B white noise worse than
  stranger. Equivalence (TOST, margin 0.1 on R): wrong detail = stranger.

## Hypotheses and precision (fixed now, STUDIES-030)

- H1 (identity is measurable here): for every computed fidelity metric, the repeat has R_self < 1 with the upper
  95% bound below 1 in UTIRIS_NIR. If not, the frame does not carry identity and the set is reported as such.
- H2 (shrinkage reward, the paper's failure mode): MSE, RMSE, MAE and PSNR prefer the dataset mean to the stranger
  (R_mean about 0.5) and fail T2 (the repeat pulled to the mean scores better than the repeat).
- H3 (fellow eye, prior work): the fellow eye is far from the repeat. Daugman and Downing (2001,
  doi:10.1098/rspb.2001.1696) found left and right eyes of 324 people as different as unrelated eyes in iris-code
  phase; Hollingsworth et al. (2011, doi:10.1016/j.cviu.2011.06.010) report texture similarity of genetically
  identical irises that iris biometrics do not detect. Prediction: R_fellow between 0.8 and 1.0 for the fidelity
  metrics; a value below 0.8 is reported as fellow-eye similarity in pixel space, not as an identity claim.
- Precision: with about 75 people (UTIRIS NIR) the person-bootstrap 95% interval of R is expected to be about +-0.05;
  differences between metrics smaller than 0.1 on R are not interpreted (the fundus tolerance). Secondary sets are
  read only where their interval excludes the comparison value.

## Known weaknesses (written before results, STUDIES-029)

1. Few people (UTIRIS 79, UPOL 64), and exclusion by segmentation failure keeps easier eyes; counts per reason
   are reported per set.
2. The UTIRIS repeat is from the same sitting (and UPOL sessions are undocumented): the acquisition floor is
   optimistic. Near-duplicate repeats (Pearson r > 0.9 on shared valid samples) are counted and reported, not
   excluded.
3. UTIRIS eye labels: a 2016 user report of inconsistent L/R labels was never corrected; mislabelled eyes make the
   fellow rung look like a repeat and the reverse. Not corrected by similarity (that would be pre-selection);
   the two fellow directions are compared.
4. No rotation search: head roll between images misaligns repeats along the angle and inflates repeat error.
5. Visible light on dark irises has weak texture and weak pupil edges; UTIRIS_VIS is secondary for this reason.
6. The rubber sheet assumes circular, concentric boundaries and linear radial stretch; nonlinear pupil dilation
   remains.
7. 16 px windows on a 64-row strip cover a quarter of the iris radius: windows are local texture patches, not
   whole-iris identity.

## Gates and outputs

- Tests: `tests/test_iris_ladder_metrics.py` (pupil through a ring reflex; mirror = strip of the flipped image;
  eyelid and reflection masks; positions within `valid`; a filler outside the mask that would win the oracle
  look-alike if counted; identical fill scores perfectly; 26 metrics registry) and `tests/test_no_box_windows.py`.
- QC sheets, private: `qc_frame_<set>_P1.png` (24 evenly spaced images per set: circles, strip with mask, repeat
  strip, mirrored fellow strip) and, after a smoke run, `qc_fills_<set>_P1.png` (fills per evenly spaced target).
  Kate's verdict is required before the full run.
- Public: `results/iris_ladder_metrics.json` (spec `ladder-result-spec/1`, `status` field, fundus structure:
  `metrics_confirmatory`, `results`, `panel`) and figures `results/figures/fig_iris_*` in the face/fundus format
  with plain labels.
