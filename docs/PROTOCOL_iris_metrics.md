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

## Amendment P2e (8 Nov 2026): margin from the mask edge. REJECTED by Kate on 8 Nov 2026 ("Keep the shadow"); not used for any result

Why. The P3 smoke QC sheets showed dark valid samples along the mask edge (NIR, VIS). Kate (8 Nov 2026) asked for a
margin from that edge, as for the brain. Written after the P2 panel, donor curve and maps and after the P3 smoke
run, so it is not blind to those numbers. It was chosen only from frame statistics, never from a metric value.

Code. `src/iris_ladder_metrics_frame.py`, version `P2e`, `EDGE_MARGIN_P2E = 3`:
- A P2 valid sample is kept only if it lies more than 3 strip samples (Euclidean, wrapped in angle) from every missing
  sample and from the pupil end.
- The grey level is z-scored again inside the kept samples.
- Images, QC, units and eligibility are those of P2.

Measured on all NIR target strips. Mean z against the angular distance from the lid gap (in limbus radii):

| Distance (limbus radii) | 0-0.05 | 0.05-0.10 | 0.10-0.15 | 0.15-0.20 | 0.20-0.25 | 0.25-0.30 | 0.30-0.35 | beyond |
|---|---|---|---|---|---|---|---|---|
| P2 | -2.13 | -1.79 | -1.40 | -1.03 | -0.74 | -0.48 | -0.26 | +0.24 |
| P2e | -1.97 | -1.61 | -1.20 | -0.84 | -0.58 | -0.33 | -0.15 | +0.24 |

The dark band is a smooth lid shadow about 0.35 limbus radii wide, not a thin rim of mis-segmented lid. One angular
sample is about 2.2 radial samples in NIR and VIS (1.5 in UPOL), so 3 isotropic samples are about 0.03 limbus radii.

Margins in physical units, after z-scoring again (dark = z < -2.5):

| Margin (limbus radii) | NIR valid kept | NIR eyes over 2% dark | VIS eyes over 2% dark |
|---|---|---|---|
| 0 | 100% | 110/156 | 15/158 |
| 0.03 | 91% | 107 | 22 |
| 0.05 | 84% | 94 | 22 |
| 0.08 | 74% | 68 | 20 |
| 0.12 | 62% | 32 | 16 |

A first count without z-scoring again gave 110 → 1 eyes for 3 samples. That count was wrong: after z-scoring again,
the same relative threshold finds new dark samples.

Status. No margin removes the shadow without cutting about 40% of the visible iris. The choice is Kate's:
- keep the shadow as an acquisition effect (as P2 decided; it recurs in the repeat image);
- use a wide margin;
- correct the slow shading inside the mask.

Decision (Kate, 8 Nov 2026): keep the shadow. P2 stays the frame for every iris result. P2e stays in the code for
the record and refuses to run (`REJECTED`).

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

## Amendment P3 (8 Nov 2026, before any result of these metrics): metrics unblocked by new weights and libraries

Approval: Kate, 7 Nov 2026 in the metrics-explorer session ("скачивай веса конечно считай все что запланировали",
"библиотеку ставь"), and 8 Nov 2026 in this session ("Да, пиши поправку"). Shared weights and libraries are listed in
the session notes file `metric-panel-resources-20261107.md` (TORCH_HOME =
`C:\Projects\data\projects\identity-ladder-metric\models-20261107\torch`). No iris number of these metrics exists
when this is written. Frame, units, windows, positions, fills, pools, seeds and the 10 computed metrics are unchanged
(P2). New code: `src/iris_ladder_metrics_extra.py` (owner: this session); rows stay private.

Common rules for the new metrics:
- Same targets, positions and fills as the panel; a metric is scored only where it is defined below, and every
  exclusion is counted per set, rung and reason. Missing is never zero: no filler enters a score. Where a network
  needs a full rectangle (LPIPS), pixels outside the target mask are set to the target's own masked Gaussian local
  mean (sd 4 px, wrapped in angle) identically in the reference and the composite, and only pixels in E are scored;
  the share of such pixels in the network input is stored per window, and the result is also reported on windows
  with at most 5% of them (sensitivity).
- Grey strips enter networks as three equal channels. Intensity mapping is fixed, not adaptive: z is clipped to
  [-3, 3] and mapped linearly to [-1, 1] (LPIPS) or [0, 1] (Betti matching, FID input before ImageNet scaling).
- Orientation: lower is closer for all new metrics; the target itself scores 0 (d_self = 0). R_self, tests T1 to T3b
  with Holm over 26 x 4, expectations and calibration follow the panel rules for per-window metrics.

Metrics (status after this amendment):
1. **LPIPS** (computed): `lpips` 0.1.4, AlexNet backbone with the package's v0.1 linear layers (primary); VGG
   (sensitivity, reported in a separate block). Input: the 64 x 64 context centred on the window (rows clipped to
   the 64-row strip, columns wrapped in angle): reference = target, composite = target with the fill inside E.
   Score = mean of the spatial LPIPS map (`spatial=True`, upsampled to input size) over E.
2. **Betti matching, matched and unmatched parts** (computed, 2 metrics): Betti-matching 2D
   (github.com/nstucki/Betti-matching at commit 3e7eca26; Stucki et al., arXiv:2211.15272), sublevel filtration,
   construction V, comparison union, relative = False, dimensions 0 and 1, on the 16 x 16 window of target and fill
   mapped to [0, 1]. Matched part = the sum over matched pairs of 2((a0 - a1)^2 + (b0 - b1)^2); unmatched part = the
   sum over unmatched intervals of (a - b)^2 (the two terms of the library's `loss`). Only windows where E covers
   the whole window (no missing pixel in target or fill) are scored; the eligible share is reported.
3. **FID** (computed per rung, set level): pytorch-fid 0.3.0 InceptionV3 pool features (weights
   pt_inception-2015-12-05), on the same full-window set as Betti matching. Reference set = the target windows;
   compared set = the fill windows of one rung on the same windows. 16 x 16 windows are upsampled bilinearly to
   299 x 299. No per-target value: no R_self per person and no Wilcoxon tests. Reported per rung: FID, its value
   over the stranger rung's (stranger = 1), and a person-bootstrap interval with 200 resamples (seed as the panel).
   With about 4,000 windows the 2048-dimensional covariance is barely determined; this is a known weakness.
4. **FWD** (computed per rung, set level, same sets and reporting as FID): Fréchet Wavelet Distance (Veeramacheneni
   et al., arXiv:2312.15289, ICLR 2025) from the authors' package `pytorchfwd` (to be installed; Kate approved
   libraries), Haar wavelet. The published setting is level 4 on 256 x 256 images; on 16 x 16 windows level 2 is
   used (packets of 4 x 4), fixed now. Windows enter as one grey channel in [0, 1]. The package pins
   scipy==1.11.1, which would downgrade the shared environment; the shared copy installed without dependencies in
   `models-20261107/pytorchfwd-lib` (pytorchfwd 1.0.1, ptwt 0.1.8, PyWavelets 1.9.0; the same folder the fundus
   panel uses) is put on the path, and its `forward_wavelet_packet_transform` is called on the window tensors; the
   per-packet Fréchet distances are averaged as the package does.
   FID and FWD are computed for the named rungs only (not for the 15 extra stranger anchors str01 to str15).

Topology subset S (added 8 Nov 2026 before any result, because of CPU cost on this shared machine without a GPU;
the same rule as the fundus panel, general protocol Amendment 9.15): per target eye, 2 positions drawn with the
seeded generator [SEED, unit, 11] among the positions where the target window lies wholly inside the target's mask
and `valid` (256 of 256 pixels); eyes with no such position are counted. Betti matching, FID and LPIPS-VGG use S;
LPIPS-AlexNet and the iris-code Hamming distance use every scored window; FWD uses every scored window whose E
covers the whole window. Measured cost on the loaded machine (8 Nov): AlexNet 27 ms, VGG 216 ms per window pair,
Betti matching 0.2 s per call. FID point estimates and bootstrap use the exact nuclear-norm form of the Fréchet
distance (Tr sqrt(C1 C2) = nuclear norm of A1 A2^T, the formula of the fundus bootstrap; identical at full rank,
checked 4.19973 vs 4.19973): the package's sqrtm of the rank-deficient 2048 x 2048 covariance took 54 min for the
12 UPOL smoke rungs (UPOL smoke values were computed with the package form and are kept as such).
5. **Topology Distance**: stays blocked. No definition is fixed in the project and none is invented here.
6. **Plausibility**: stays blocked (needs a trained density model of iris strips).
7. **Identity x4** (embedding angle, norm, PFE, PLDA): stay blocked; no deep iris encoder with open weights is on
   this machine. Supplementary, not a replacement: **iris-code Hamming distance** (`src/iris_codes.log_gabor_code`
   on the P2 strip, unchanged parameters): fractional Hamming distance between the codes of the target strip and of
   the composite strip, over code bits valid in both whose rows lie in the window and whose columns lie within the
   window +- 8 columns, no rotation shift. The filter runs along the whole angle, so pixels outside the target mask
   are set as for LPIPS (identically in both strips), and a bit counts only where the target mask holds at that
   sample (inside the window: where E holds).
8. **Latent x3**: stay blocked (no generative iris model, CPU only).

Hypotheses for the new metrics, fixed now:
- H4: LPIPS and both Betti-matching parts do not reward shrinkage: the repeat scores better than the repeat pulled
  to the mean (T2 passes in UTIRIS_NIR), unlike MSE (H2).
- H5: FID and FWD rank the dataset mean and the mean of 2 as worse than or equal to one stranger (ratio to the
  stranger >= 0.9), since averaged windows lose texture.
- H6: LPIPS keeps the panel's order repeat < lookalike < stranger with R_repeat upper bound below 1 in UTIRIS_NIR.
Precision: as the panel (differences below 0.1 on R not interpreted); set-level ratios are read only where the
bootstrap interval excludes the comparison value.

Additional weaknesses: networks trained on natural photos see a 16 px texture patch upsampled (FID) or a small
context (LPIPS); Betti matching was designed for segmentation likelihoods, here applied to intensities; the
full-window restriction keeps cleaner windows (eligible shares are reported per rung). The lid shadow (Amendment
P2e) stays in the valid samples: within about 0.35 limbus radii of a lid gap the NIR grey level is lower (mean z
-2.1 next to the gap, +0.24 beyond 0.35). It recurs in the repeat image of the same eye; for other people it
depends on their lid position, so window-level metrics may partly measure lid geometry. This is not corrected.

Gate: a smoke run (every 8th eye per side, all three sets) writes `qc_extra_<set>_P2.png` (for evenly spaced
targets: reference and composite contexts per rung with LPIPS, the window pair with Betti-matching parts, and the
eligible shares). Kate's verdict on that sheet is required before the full run. Status label until then: pre-QC.
Verdict (Kate, 8 Nov 2026): run on P2 with the lid shadow kept (Amendment P2e rejected). The full run is sharded
over targets (`--shard i/n`, each target in exactly one shard, same seeds per target); set-level FID and FWD are
computed once on the merged windows (`merge`).

## Amendment P4 (8 Nov 2026 night, window in object units; recorded before any result at this size)

Rule: cross-modality note `window-in-object-units-20261108.md` (Kate, 8 Nov 2026, "да переделай честно"); minimal
form per Kate's night plan of 8-9 Nov ("one size f = 1/8 next to the current one", one CPU process).
- Object extent L = iris outer diameter in strip columns: D = W / pi = 256 / pi = 81.5 angular samples. The current
  16 px window is f = 16 / 81.5 = 0.196 (about 1/5); it stays the reference result and is not rerun.
- New size: f = 1/8, w = round(81.5 / 8) = 10 px (both axes of the strip, as now), context ring ceil(w / 2) = 5 px.
  Positions on a 10 px grid with the validity thresholds unchanged (95% of the window, 50% of the ring, 50% in the
  target mask); the number of positions is reported, never filled. Everything else as the panel (frame P2, fills,
  pools, seeds, 10 computed metrics, R_self, T1-T3b with Holm 26 x 4).
- Scope (CPU budget of one process): UTIRIS_NIR only (primary set). Not run tonight, written as limitations: f = 1/16
  (5 px) and f = 1/4 (20 px), VIS and UPOL at 10 px, and the Amendment P3 metrics (LPIPS, Betti matching, FID, FWD,
  iris code) at 10 px.
- Report: `results/iris_ladder_metrics_w10.json` (same structure as the panel JSON; 16 px results untouched) and a
  figure of R_self per rung at 16 px against 10 px with the Spearman correlation of rung order per metric. A rung-
  order claim made at 16 px is stated with its scale unless it also holds at 10 px.
- Gate: smoke run (every 8th eye per side) with the fills QC sheet at 10 px, checked by eye by the agent before the
  full run (Kate's night rule); the window is a required argument (`--window`, no default).
- Weakness: one angular sample is about 2.2 radial samples in NIR, so a square 10 x 10 px window is about 10 x 4.5
  in isotropic units, as the 16 px window was (16 x 7); the anisotropy is unchanged, not corrected.

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
