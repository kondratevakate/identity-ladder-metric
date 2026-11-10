# Protocol: tier-1 metric audit on the face ladder (6 Nov 2026; recorded before any face metric result)

Face replication of the brain metric audit (`PROTOCOL_donor_ladder_brain.md`, Amendments 10 to 14), as the second
modality. Amendment to the face protocol `PROTOCOL_donor_ladder_face.md` (Amendments 1 to 8, kept untracked in the
main checkout), written here because that file is not in this branch. Status: pre-QC. The face frame has not passed
Kate's visual gate; landmark alignment to the population template is accepted for current runs (face protocol,
Limitations 3). Kate's direction (6 Nov 2026): replicate on faces before any further brain work, keep it simple.

## Data and frame (read only, owned by the face pipeline)

- Frame version: v6, named by the face-pipeline session (6 Nov 2026) as the current safe version while v7 is being
  written. Read-only snapshot files: `frame_v6.npz`, `images_v6.json`, `landmarks_v6.npz`, `identity.npz` (ArcFace,
  rows of `images_v6.json`), `results/donor_ladder_face_v6.json`. The live `images.json`, `landmarks.npz` and
  `identity_v7.npz` are not read. v6 masks still contain the colour and lip rules later withdrawn (face Amendments
  7b to 7d), so they are slightly over-masked relative to the final rule; a v7 rerun follows once Kate approves v7. Grey level z-scored inside each image's own mask, 128 x 128 frame, 478-landmark affine to the symmetric
  template, pixel masks = face skin in the oval hull and the photograph, plus glasses (face Amendment 7d).
- Pose: |yaw| and |pitch| < 10 deg for targets, repeats and pool (primary of face Amendment 8.4).
- People, targets and pool exactly as `donor_ladder_face.maps` builds them for half "all": one image per person (the
  most frontal); CelebA targets from the test partition, pool from train; IMDb-WIKI targets in kinship pairs, pool
  people in none. Families: CelebA one per person; IMDb connected components over full-sibling and parent-child
  pairs. Coverage `valid` as in face Amendment 7 (anatomical coverage >= 90% of the pool and >= 50% of the pool
  unmasked). The two datasets are analysed and reported separately.
- Reproduction gate (before any metric is read): the whole-face squared-error ratio to the dataset mean of the
  deterministic rungs (repeat, mirror, sibling, sex-age mean) computed by this script must equal the face results
  JSON of the same version to its 3 printed decimals.

## Windows (fixed before results)

- 16 x 16 px windows (0.4 inter-ocular distance, the face map window) on a 16 px grid over the frame; a window is a
  position if >= 95% of its pixels are in `valid` and >= 50% of its 8 px context ring is in `valid`. The window
  count is reported. Positions play the role of the brain's 32 random cubes; there are no boxes.
- A target enters a position if >= 50% of the window is inside its own mask and `valid`.
- Scored pixels E = window AND target mask AND fill validity AND `valid` (missing is never zero; stored zeros
  outside masks never enter). A mean fill averages, per pixel, only pool members valid there.

## Fills (rungs)

| Block | Rung | Definition |
|---|---|---|
| same person | repeat | the person's other image nearest in pose and expression, not a near-duplicate (r <= 0.9); as the face pipeline |
| same person | own mirror | the target photo flipped about the midline, pixels with abs(x - 63.5) < 2.5 removed (face Amendment 5.4) |
| relatives | sibling | IMDb only: each full sibling's image nearest in pose and expression; first sibling by id |
| selected by context (achievable) | look-alike, k = 1 | the single pool person nearest on the 8 px ring around the window (mean squared difference over ring pixels inside both masks, >= 50% of the target's ring pixels shared; the candidate's own mask covers >= 50% of the window), as the brain scaffold rule |
| no selection | stranger | first person of a seeded permutation of the pool |
| no selection | mean of 2 | first two people of that permutation, masked mean |
| no selection | sex-age mean | pool cell (CelebA Male x Young; IMDb sex x 5-year band, >= 10 people), masked mean |
| no selection | dataset mean | masked mean of the whole pool |
| oracle (unreachable) | oracle look-alike, k = 1 | the pool person nearest on the window itself (same masked rule) |
| controls | wrong detail | dataset mean + the stranger's residual made orthogonal to the target's residual and scaled to its norm on E |
| controls | white noise | dataset mean + Gaussian noise with the SD of the target's residual on E (seeded) |
| controls (shrinkage table) | repeat blend, mirror blend | mean + lambda (donor - mean), lambda = r = 1 - ratio / 2 of that rung in the face results JSON (whole face) |

Strangers for the anchor and c: the first 16 people of the same seeded permutation (str00 to str15). The stranger
rung is str00. Pool size is reported with every look-alike and oracle value. Cross-modality sensitivity (brain
Amendment 11, "pool subsampled to the smallest dataset's pool"): look-alike and oracle also from a seeded pool
subsample of 219 people, the QTIM pool size (rungs `lookalike_p219`, `lookalike_oracle_p219`; supplement only).

## Metrics (tier 1, imported unchanged from `src/donor_ladder_brain_metrics.py`)

MSE, RMSE, MAE, PSNR (range of the target on E), PSNR fixed (8 z), SSIM (Gaussian sigma 1.5 px, normalised
convolution with E as weights), Pearson r, ACC vs dataset mean, ACC vs sex-age mean, amplitude ratio (descriptive),
lDDT-like (pixel pairs within 2 px, thresholds 0.25, 0.5, 1, 2 z); protein-style group 1 with sigma_rep per window
position from repeat pairs of the other families (leave-family-out, >= 5 pairs): TM-like, GDT-like, lDDT_rep,
RMSE_rep. The 2D window is passed as a (1, 16, 16) array; offsets and filters along the singleton axis add nothing
(tested).

Identity reference (no model run): ArcFace cosine between the target image and the donor image, from the
embeddings already computed by the face pipeline (face Amendment 5), d = 1 - cosine, for rungs that are one real
image: repeat, sibling, stranger (anchor: the 16 strangers), look-alike and oracle (mean over the target's windows
of the cosine to that window's donor). Not defined for the mirror (the flipped image is not embedded), means, noise
and blends. Reported next to the panel, not scored as a tier-1 metric.

## Scale, target and checks (as brain Amendments 11, 13, 14)

- R(rung) = (mean d(rung) - mean d(repeat)) / (mean d(16 strangers) - mean d(repeat)), d = distance (similarities
  negated); repeat 0, unrelated person 1. Means over (target, window) cells.
- Intervals: two-way bootstrap, persons (CelebA) or families (IMDb) and window positions resampled independently,
  2000 resamples, seed 20261102; family-only interval also reported.
- Reality-B axiomatic target (Amendment 14, no measured value enters): dataset mean 0.5, sex-age mean 0.5,
  mean of 2 0.75, stranger 1, wrong detail 1; white noise above 1 (lower bound > 1); partly correct rungs (own
  mirror, sibling, look-alike, oracle) below 1 (upper bound < 1); oracle look-alike not above the look-alike.
  Score per metric: value fit (RMS of R - target over dataset mean, sex-age mean, mean of 2, wrong detail) and the
  side constraints.
- Expectations E1, E2, E3B, E5B, E6B (Amendment 13) and the reality-A descriptive set (Amendment 11 E3, E4) are
  computed with the brain code. E2 is not available for CelebA (no relatives).
- Shrinkage table (Amendment 13): per metric R of the donor, R of the blend and blend - donor with its interval.
- Expected difference from the brain, written before results: the face repeat is another photograph (pose,
  expression, light, age), so the repeat anchor is far from 0 error. In the face pipeline v6 the whole-face repeat
  error is above the dataset mean's (ratio 1.32 CelebA, 1.40 IMDb). The R scale is defined by the repeat anyway;
  rungs with negative R are reported as such and not clipped.

## Gates and outputs

- Known-geometry tests (`tests/test_face_ladder_metrics.py`), including a filler outside the mask that would win
  the look-alike search if counted.
- QC sheet of the fills on evenly spaced targets (private folder only, never in git), looked at by the agent before
  the full run; Kate's look is pending unless stated.
- Private rows: `C:\Projects\data\projects\identity-ladder-metric\face-ladder-metrics-20261106-v1\`. Public:
  `results/face_ladder_metrics.json` (aggregates only, no image, file name or identity id), figures
  `results/figures/fig_face_metrics_*`.
- Not run: tier 2 and 3 (as the brain), map figures, pose 15 and 20 deg, strata.

## Amendment F1 (6 Nov 2026, zero = the target itself; augmentation rungs; recorded before these results)

Seen: the subset run (24 CelebA + 24 IMDb-WIKI targets, v6). On faces the retest photo is far from zero error (its
whole-face error is above the dataset mean's), so with the retest as zero the dataset mean, the look-alike and the
own mirror get negative R. Kate (6 Nov 2026): the retest cannot be the zero of the face scale. Decision ("Yes, both"):

1. Scale (primary from now on). R_self(rung) = (mean d(rung) - d(self)) / (mean d(16 strangers) - d(self)), where
   d(self) is the metric's value for a fill identical to the target: 0 for MSE, RMSE, MAE and RMSE_rep; 1 for SSIM,
   Pearson, both ACC, lDDT-like, TM-like, GDT-like, lDDT_rep and ArcFace (similarities, oriented as d = -value).
   PSNR and PSNR fixed are infinite on the target itself and get no R_self (they are monotone in MSE). The value
   is known analytically, so no fill is rerun for it; it is entered as a rung "self" in the same (target, window)
   cells where the stranger anchor exists, and the two-way bootstrap is unchanged. The retest becomes a measured
   rung ("retest", the acquisition floor). The axiomatic reality-B targets of Amendment 14 keep their values on
   this scale (under the additive model target = mean + person + acquisition: population mean 0.5, mean of k
   0.5 + 0.5 / k, wrong detail 1); the retest gets "below 1" and no value. The retest-anchored R stays in the JSON
   as `ladder_R_retest` (descriptive). Expectations and the Amendment 14 score are read from R_self.
2. Augmentation rungs (block "same photo, acquisition changed"; a metric that is invariant to acquisition puts them
   near 0). Amplitudes come from the data, not chosen:
   - aug_shift: the target translated (bilinear; mask by the same shift, a pixel valid only if fully inside) by
     s px in a seeded random direction, s = median over targets of the RMS distance between the 478 landmarks of
     the target and of its retest, both mapped into the frame by their least-squares affine to the template
     (`template_shape` of the frame file; `landmarks_v6.npz`). Computed per source and reported.
   - aug_light: the target plus the low-frequency difference between its own retest and itself, G(x_retest -
     x_target) by mask-normalized Gaussian smoothing over pixels valid in both, sigma 16 px (the pipeline's
     illumination sigma); valid where both are valid.
3. Everything else unchanged. Subset rerun (same 24 + 24 targets) and its QC sheet are shown to Kate before the full
   run.

## Note (6 Nov 2026, face version for final numbers)

The face-pipeline owner reports that Kate froze the v6 selection and parked v7. Final face numbers come from v6f
(v6 selection with the final masks: segmenter + glasses; face protocol Amendments 9-10), once Kate approves the v6f
QC: `frame_v6f.npz`, `results/donor_ladder_face_v6f.json`, same image rows as `images_v6.json`, `identity.npz`.
The v6 run is development only. The reproduction gate is rerun against `donor_ladder_face_v6f.json`.
Update (6 Nov 2026): the face pipeline records the v6f visual QC as approved by Kate (face protocol Amendment 10,
`QC_APPROVED` in `donor_ladder_face.py`); Kate asked for the v6f rerun. The v6 aggregates are kept privately as
`face_ladder_metrics_v6_public_copy.json`; `results/face_ladder_metrics.json` now holds v6f.

## Amendment F2 (8 Nov 2026, the rest of the 26-metric panel on v6f; recorded before any of these results)

Kate approved computing every planned metric of the 26-metric panel with weights and libraries (7 Nov 2026,
metrics-explorer session: "скачивай веса конечно считай все что запланировали", "библиотеку ставь"). This amendment
adds to the frozen v6f run of Amendment F1 (same targets, positions, donors, fills, seeds, scored pixels E and
scale R_self) the metrics still missing on faces. Fills are rebuilt by `face_ladder_metrics.window_fills`
(unchanged; an optional hook only reports the donor row). Reproduction gate: the rebuilt MSE of every rebuilt row
must equal `rows_v6f.csv.gz` (|diff| <= 1e-5) before any new value is read. Code: `src/face_ladder_metrics_panel.py`.
New private rows: `C:\Projects\data\projects\identity-ladder-metric\face-ladder-metrics-20261108-v2\`; v1 stays.
2D functions shared with the fundus and iris panels are imported unchanged by explicit path (`FUNDUS_METRICS_SRC`,
no default; sha256 stored in the JSON).

Group A (per window, as fundus and iris):
- MS-SSIM: fundus `ms_ssim` (3 scales; composite = target outside the window, fill on E inside it; 24 px context;
  masked 2 x 2 downsampling; Wang et al. weights renormalised). Higher = closer.
- LPIPS: AlexNet, lpips 0.1.4, linear layers v0.1 (TORCH_HOME = `models-20261107/torch`). The 16 px window,
  z / 4 clipped to [-1, 1], grey repeated to 3 channels, bilinearly upsampled to 64 px (the brain's short side).
  Scored only if every pixel of the window is in E (no filler enters the network), else missing; the share of
  scored cells is reported per rung (the mirror loses the central windows to its midline strip). Lower = closer.
- Amplitude vs matched mean: |1 - ||f - mu_c|| / ||x - mu_c|| | on E, mu_c the sex-age cell mean of Amendment F1
  (missing where the target has no cell); `ampdev_dataset` likewise with the dataset mean, from the existing
  `amplitude`. Lower = closer, d(self) = 0.

Group B structure (per window):
- Intensity persistence: fundus `persistence_distance` (H0 sublevel + superlevel diagrams on E, 4-connectivity,
  1-Wasserstein, L-infinity ground metric). Lower = closer, d(self) = 0.
- Betti matching (Stucki et al., ICML 2023; github.com/nstucki/Betti-matching at 3e7eca26, `BettiMatching`
  unchanged): z mapped to [0, 1] by (z / 4 + 1) / 2 clipped; pixels outside E set to 1 in both images (identical in
  target and fill, so they enter last and never differ); sublevel filtration, relative = False, comparison = union,
  dimensions 0 and 1. Matched = sum over matched pairs of 2((b0 - b1)^2 + (d0 - d1)^2); unmatched = sum over the
  unmatched intervals of both images of (b - d)^2 (the two parts of the library's `loss`). Lower = closer,
  d(self) = 0. Cost (timed 8 Nov: 0.16 s per pair, about 37 CPU hours for all 870k rows), so Betti matching is
  computed on a fixed subset: 100 evenly spaced targets per source (the `--limit` rule), all positions and rungs,
  reported with its own n.

Group B identity (per row, one-image rungs only, as `arcface_cos`: retest, sibling, stranger and str00 to str15,
look-alike and oracle; missing for mirror, means, noise, blends and augmentations):
- id_angle: the existing ArcFace cosine (registry name only).
- id_norm: the raw ArcFace w600k_r50 embedding norm before L2 normalisation (a quality term, FACE_ID_LESSONS R17;
  ArcFace is not MagFace, so this is a proxy). The stored embeddings are normalised (norm 1.0000), so the face
  pipeline's ONNX model is rerun on the cached images with its own alignment (five points, similarity warp to
  112 px); gate: the normalised rerun matches the stored embedding (cosine >= 0.999). d = | ||e_f|| - ||e_x|| |.
- id_plda: two-covariance PLDA log-likelihood ratio fitted on pool people only (disjoint from targets): PCA of the
  pool embeddings to 128 dimensions; W = pooled within-person covariance over pool people with >= 2 analysable
  images in `images_v6.json` (Amendment 5 outliers excluded); B = covariance of person means minus mean(W / n_i);
  eigenvalues floored at 1e-6. LLR(x1, x2) = log N([x1; x2]; 0, [[B + W, B], [B, B + W]]) - log N(x1; 0, B + W)
  - log N(x2; 0, B + W), derived from the model (Sizov et al. 2014, doi:10.1007/978-3-662-44415-3_47; the closed
  form is not quoted from a source). Higher = same person; d(self) = LLR(x, x) per target.
- id_pfe: blocked. The mutual likelihood score (Shi and Jain, ICCV 2019, arXiv:1904.09658) needs a per-image
  variance from a trained uncertainty head; none exists for the InsightFace w600k_r50 backbone (the public PFE
  weights are for other backbones). Training one is a separate project.

Group B latent: blocked, all three. Pullback and Fisher-Rao need a face generator with an encoder (for example
StyleGAN2-FFHQ with an inversion encoder) that maps the 128 px grey z-scored masked frame into its latent space;
LDDMM between two 16 px windows is a pairwise diffeomorphic registration (the fundus stop rule). Needs: generator
and encoder weights (download), RGB re-rendering of the aligned crops, and a GPU (inverting about 7000 images and
the fills of 34 rungs is days on this CPU).

Group C (set-level, per rung and source; reference-free, so no per-target value; reported raw next to the target
set's own value, no R):
- Cells: a seeded sample (seed 20261102) of 1000 (target, position) cells per source; per rung the cells where its
  window is complete (every pixel in E); the reference set is the target windows of the same cells.
- FID: pytorch-fid 0.3.0 InceptionV3 pool3 (2048-D); window (z / 4 + 1) / 2 clipped to [0, 1], grey to 3 channels,
  resized to 299 px by pytorch-fid. Out of domain for Inception (16 px patches), and n is below 2048, so the value is
  biased; all rungs share the cells and are comparable within a source only.
- FWD (Veeramacheneni et al., arXiv:2312.15289; PyTorch-FWD defaults read from its code): Haar wavelet packets,
  Frechet distance per packet, unweighted mean over packets, values in [0, 1] as above. Level 2 for 16 px windows
  (16 packets of 4 x 4; the README lowers the level for smaller images); numpy Haar (pywt is not installed;
  orthonormality tested).
- Topology Distance (Horak et al., AAAI 2021, doi:10.1609/aaai.v35i9.16943): H0 Vietoris-Rips on the Inception
  features (deaths = minimum-spanning-tree edge lengths, Euclidean), sorted longevity vectors, L2 distance; the one
  essential bar on each side gives inf - inf = 0; groups of 500 points, mean over the two groups of the sample.
- Plausibility: Gaussian density of the window per position (Ledoit-Wolf covariance) fitted on pool windows fully
  in mask and `valid`; two seeded pool halves give two models; a pool donor is scored by the model of the half it is
  not in, every other fill and the target by the mean of both. Value = negative log-likelihood per scored pixel of
  the marginal over E (missing pixels marginalised out, never filled). Per row, so it is also reported per rung
  with R_self (d(self) = the target's own value); lower = more plausible, not closer to the target.
- Expectations, written now: LPIPS and MS-SSIM order like SSIM but rank the dataset mean further from the target;
  Betti matching and persistence put the dataset mean and noise far (fewer or extra features); the identity metrics
  separate retest and sibling from strangers; group C cannot see identity: strangers score like the target set, the
  dataset mean and noise worst. Any of these failing is reported as such.

Gates: tests (`tests/test_face_ladder_metrics_panel.py`: an identical fill scores perfectly on every metric; a
filler outside E changes nothing; Haar packets orthonormal; PLDA LLR higher for same-person pairs on a synthetic
model); a smoke run on 4 evenly spaced targets per source with timing, and a QC sheet (private) shown to Kate before
the full run.

## Amendment F3 (8 Nov 2026, window size in object units; recorded before any result at these sizes)

Kate approved redoing the window size in object units for all four modalities (8 Nov 2026, metrics-explorer session:
"да переделай честно"). Cross-modality rule: side-session note `window-in-object-units-20261108.md`. The 16 px window of
F1/F2 covers a different share of a face than 16 px or 16 mm covers of a brain, fundus or iris, so rung positions
between the mean and the stranger can move with the window. The F2 full run at 16 px continues; it is the reference.

Object extent (geometry only, measured 8 Nov from the v6f cache, no metric read): L = sqrt(mean area of the face mask
`M` in the 128 px frame) over the targets and pool rows of each source: CelebA 79.55 px (n = 2765 rows), IMDb-WIKI
79.47 px (n = 2335). For reference, the mask inside `valid` gives 74.17 and 73.32 px. The 16 px window is f = 16 / L =
0.201 (about 1/5) on both sources.

Sizes: w = f x L rounded half up, f in {1/16, 1/8, 1/4}: w = 5, 10, 20 px on both sources. Context ring = ceil(w / 2) =
3, 5, 10 px. Grid, thresholds and entry rule are those of the Windows section with w in place of 16 (grid step w;
>= 95% of the window and >= 50% of the nominal ring in `valid`; a target enters if >= 50% of the window is in its
mask and `valid`). Positions: CelebA 215, 47, 8; IMDb-WIKI 214, 45, 8 (16 px: 18). Fewer positions at larger sizes are
reported, never filled. Donors, seeds, lambdas (whole-face), the midline strip of the mirror, E and R_self are as F1.
The window and ring are passed explicitly to the geometry functions; no module default is used for these runs.

What runs at each size:
1. Tier-1 panel of F1 (the `face_ladder_metrics` rows), all targets, all three sizes.
2. F2 per-window panel (MS-SSIM, LPIPS, amplitude vs matched mean, persistence, Betti matching, plausibility) on the
   fixed subset of 100 evenly spaced targets per source (the F2 Betti rule), all positions, all three sizes. The
   subset keeps the cost bounded (estimate below) and gives paired rank comparisons across sizes.
   - LPIPS is not computed at w = 5: 25 pixels upsampled 12.8x carry no structure at AlexNet's scale. Reported as
     "not defined at this size", not as a value.
   - MS-SSIM keeps the fundus function and its own validity rule (missing when fewer than 10 scored pixels remain at
     a scale); the share of missing cells per size is reported.
   - Plausibility: Gaussian window densities refit per size (w^2 = 25, 100, 400 dimensions, Ledoit-Wolf, the same two
     pool halves); a position with fewer than 100 complete pool windows per half stops the run.
   - id_norm and id_plda depend on the donor image, not the window; F2 values are used, not recomputed.
3. Set-level metrics (FID, FWD, Topology Distance) only at f = 1/8 (w = 10): 1000 seeded cells per source from the
   subset. FWD uses Haar level 1 (10 px is not divisible by 4; 4 packets of 5 x 5).
4. Latent metrics and id_pfe stay blocked (F2).

Reports per source and size: L, f of the 16 px window, positions, the tile ranks and T1 to T3b as F1, and rank
stability. Rank stability = Spearman correlation of the rung order (mean R_self per rung) between each pair of the
sizes 5, 10, 16 and 20 px, per metric. A rung-order claim holds only if it holds at all three object-unit sizes;
otherwise it is stated together with its scale.

Cost (estimate from the v1 run, 870k rows in 29 min at 16 px, scaled by positions): tier-1 panel about 6 h at
w = 5, 1.3 h at w = 10, 0.25 h at w = 20; F2 subset panel about 8 h for all three sizes. These run after the F2 full
run, with at most 3 workers. Rows go to the private folder
`C:\Projects\data\projects\identity-ladder-metric\face-ladder-sizes-20261108-v3\`.

Gates: a test that the geometry at w = 16, ring 8 reproduces the F1 positions exactly, and that every size's positions
satisfy the thresholds. A smoke run at w = 10 on 4 evenly spaced targets per source, with a QC sheet (private) shown
to Kate before the sweep.

### Amendment F3.1 (8 Nov 2026 night, minimal sweep; recorded before any result at these sizes)
Kate's stop rule for the night (docs/HANDOFF.md, "Night plan 8-9 Nov"): a minimal checkable result per item. The F3
sweep therefore runs the tier-1 panel at w = 5, 10 and 20 px on the same 100 evenly spaced targets per source (the
`--limit 100` rule), not on all targets, and only after the w = 10 smoke sheet looks right to the agent. The F2 panel at
the three sizes is not run tonight; it is reported as a limitation (needs per-size densities, about 8 CPU hours).
Rank stability is computed on these 100 targets per source, against the 16 px tier-1 rows of the same targets.

## Amendment F2.1 (9 Nov 2026, LPIPS with the VGG backbone; recorded before any VGG result)

Requested by the cross-modality integration (the brain panel adds LPIPS VGG); Kate: keep the old results and write
this as new ones. Nothing in F2 changes: `panel_f2`, its rows and its parts stay as they are.
- Metric: LPIPS VGG16, lpips 0.1.4, linear layers v0.1 (torchvision VGG16 weights `vgg16-397923af.pth` from the same
  TORCH_HOME). Input rule unchanged from F2: the 16 px window, z / 4 clipped to [-1, 1], grey to 3 channels, bilinear
  to 64 px, scored only if every window pixel is in E. Lower = closer.
- Pairs: the same F2 fills, rebuilt by the same code at v6f; a VGG-only pass (no other metric recomputed). Gate: the
  rebuilt MSE equals v1 within 1e-5 on every row, and the scored (source, target, position, fill) set equals the set
  where F2 AlexNet LPIPS is present; otherwise nothing is written.
- Outputs: private `face-ladder-metrics-20261109-v3-vgg/` (parts, rows with keys and `lpips_vgg` only, meta); public
  `results/face_ladder_metrics.json` gains a sibling block `panel_f2_lpips_vgg` with the F2 view shape
  (`views[src].ladder_R.lpips_vgg`, R with 0 = target itself, 1 = stranger mean, family-bootstrap CI).
- Expectation, written now (descriptive, not a gate): VGG orders the rungs like AlexNet (Spearman of rung R >= 0.9 per
  source); the Spearman is reported next to the result.
- Compute: local CPU, one torch thread per worker, at most 6 workers (the brain pass holds 2 of the 8 cores); smoke on
  4 targets per source with timing and a QC look before the full run.

### F2.1 note (9 Nov 2026, cloud CPU for the IMDb-WIKI jobs; recorded before those jobs were scored)

The 6-worker laptop run took about 20 GB (3.3 GB per worker) and starved the workstation; it was stopped with 226 of
594 jobs saved (all CelebA). Kate (via the ISBI session): run the rest on a VM. IMDb-WIKI goes to the GCP VM
fs-cpu-remainder (n2-standard-32 spot, europe-west1-b, Kate's project); CelebA stays on the laptop (its agreement
allows copies for internal use at a single site; a cloud copy needs Kate's explicit yes), resumed with 2 workers.
- Payload (`src/face_vgg_vm.py payload`): the v6f cache with every non-IMDb row blanked (NaN in X and embeddings, False
  in masks; 3260 IMDb rows kept, 3686 other rows blanked, no overlap), same indices and job names; v1 keys + MSE for
  IMDb only; the face results JSON (aggregates). Copied by scp to the VM disk, no bucket. Rows return as CSV + JSON
  (no pickles leave the VM) and are deleted from the VM with the VM.
- Environment: Python 3.12.3 (laptop 3.11.9); numpy 1.26.4, scipy 1.15.3, pandas 2.3.3, torch 2.10.0+cpu, torchvision
  0.25.0+cpu, lpips 0.1.4, scikit-learn 1.8.0 as on the laptop; VGG16 weights sha256 397923af8e79... equal.
- Reproduction across machines, required before the remaining jobs: two IMDb jobs (imdb_p00_c00, imdb_p09_c07; 2914
  rows, 1958 VGG pairs) run on both. Result: same keys and missing pattern, MSE identical (max |diff| 0; v1 residual 0
  on both), LPIPS VGG max |diff| 1.5e-7 (float32 last bit) against the 1e-5 tolerance: passed. A first comparison
  showed an MSE difference of 8.9e-16 that came from the default CSV float text format (the laptop's own rows through
  the same CSV differ by the same amount, and by 0 from the VM); transport is now lossless (`%.17g`, round-trip read),
  with a test. The laptop values are kept for the two reference jobs.
- Where each job ran: CelebA 226 jobs on the laptop (6 workers, before the stop) and 116 on the laptop (2 workers, after
  the brain A22 run); IMDb-WIKI 2 reference jobs on both machines (laptop values kept) and 250 on the VM (n2-standard-16
  spot after a preemption of the 32-vCPU machine; the global 32-vCPU quota did not allow the restart at 32). The CelebA
  cloud copy was not made (the upload needs Kate's approval in the running session). The VM and its disk were deleted
  on 9 Nov on Kate's direct yes; no payload remains in the cloud.
- Result (aggregates in `results/face_ladder_metrics.json`, block `panel_f2_lpips_vgg`): gates passed on all 594 jobs
  (869972 rows reproduce v1 exactly; VGG scored exactly the 593018 AlexNet rows). Rung order Spearman vs AlexNet
  0.95 (CelebA) and 0.96 (IMDb-WIKI), above the 0.9 written expectation. VGG compresses the scale (white noise R 1.35 /
  1.25 against 2.74 / 2.19) and, unlike AlexNet, places the dataset mean below the stranger (R 0.87 / 0.90).

## Amendment F2.2 (9 Nov 2026, rung "mean of 16 look-alikes"; recorded before any of its results)

Kate ("запускай", 9 Nov, relayed by the ISBI session): add the rung that fundus and iris have, key `lookalike16`.
- Fill: per target and window, the 16 pool donors nearest to the target in the context ring (the look-alike search of
  F1 unchanged: masked ring distance, brain masked_dist rule, eligibility = window at least 50% inside the donor's
  mask; ranking by distance, ties by pool order), averaged pixel-wise over the donors valid there (masked mean, as
  `mean_k2` and iris `lookalike16`). Missing if fewer than 16 donors are eligible. Same frame (v6f), windows (16 px,
  18 positions), pool, E rule and missing rule as F1/F2. The candidates are not stored; they are recomputed by the
  same distance, so the search cost equals that of the existing k = 1 look-alike.
- Metrics: the tier-1 panel (F1), the F2 panel (identity metrics missing: not one person) and LPIPS VGG (F2.1).
- Gates: in each pass the k = 1 look-alike is recomputed next to the new fill and must equal the stored rows (tier-1:
  every metric within 1e-9 of v1; F2: MSE within 1e-5 of v1 and the F2 metrics within 1e-9; LPIPS AlexNet and
  VGG within 1e-5, since the pairs are batched differently than in the stored runs; every stored row found); the rank-1 donor of the 16 must be the k = 1 donor. Otherwise nothing is written.
- Outputs: private `face-ladder-lookalike16-20261109-v1/`; public sibling block `rung_lookalike16` in
  `results/face_ladder_metrics.json`, `views[src].ladder_R[metric].lookalike16` (R with the F1 scale, computed with the
  stored rows of all other rungs), plus raw means; all other blocks unchanged.
- Expectation, written now: like the other averages, the mean of 16 look-alikes is smoother than any single donor, so
  for pixel metrics (MSE, MAE) it lies closer to the target than the k = 1 look-alike and the stranger; for structure
  and perceptual metrics (SSIM, LPIPS) it may rank behind the k = 1 look-alike, as the dataset mean does.
- Compute: laptop CPU, one worker at a time behind the RAM guard; smoke on 4 targets per source with timing and a QC
  sheet (target, k = 1 look-alike, mean of 16) before the full run.
