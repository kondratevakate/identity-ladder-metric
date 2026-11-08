# Cross-modality analysis of the metric panel (exploratory, post hoc; 7-8 Nov 2026)

Status: exploratory. Everything below was decided after the per-modality results existed, during Kate's review in the
metrics-explorer session. It is not a pre-registered analysis and must be reported as exploratory in any paper.
Code: `src/cross_modality/` (run order in its README). Aggregates and figures: `results/cross_modality/`.
Data status: brain (QTIM, affine only, registration gate not met), fundus (GRAPE P8, PAPILA P9) and iris (frame P2;
lid-shadow sheets not yet re-shown) are pre-QC; faces (frame v6f) are QC approved.

## 1. One computation path for every dataset
- Metric code: all modalities score with `src/donor_ladder_brain_metrics.py` (sha256 3ac5878f...), imported unchanged
  by the face, fundus and iris pipelines.
- R_self = (mean d(rung) - d(self)) / (mean d(16 strangers) - d(self)), one function (`common_frame.py`); PSNR is
  reported as dB minus the strangers' dB, amplitude deviation |1 - amplitude| as a raw mean (no closeness direction).
  Matches the fundus and iris files exactly, the face file within 0.01.
- Reference tests, one rule (`tests_uniform.py`): d = distance(control) - distance(reference) per cell, mean per
  target, mean per family or person, one-sided Wilcoxon (at least 10 units), Holm within dataset. Reference = repeat
  image, else the other side (PAPILA; T1 not available). T1 control = dataset mean, T2 = reference shrunk to the
  mean, T3a = wrong detail, T3b = white noise. p values equal the brain, GRAPE, PAPILA and iris files exactly; faces
  had no tests before and are computed here. Before this, brain used Holm and the fundus file was unadjusted.
- Calibrations (brain Amendment 12 group 2): `cal_uniform.py` ports `fundus_ladder_metrics.py:calibrate`; checked
  against GRAPE (z, EVT, posterior to machine precision, c within 5e-5) and the brain file (z, EVT, posterior). Face
  calibrations (all four) and brain c (absent from the brain file) are computed here.
- Remaining non-identical points: fundus divides by a background estimate, faces have no illumination correction;
  brain bootstrap is over targets (no family column in the rows); brain uses random cubes, both sides.

## 2. Displays chosen with Kate (figures in `results/cross_modality/`)
- Rank tiles: rows = metric x dataset, columns = rungs in one fixed ladder order, tile = the place the metric gives
  the rung from its own raw values (1 = closest). White noise always shown. Column headers carry pictures on a drawn
  face (no real person; `src/ladder_faces.py`): the box over one brow and eye is replaced by each rung.
- Kate's choices: test controls marked in the column headers as [T1]-[T3b] (corner pass/fail flags removed as hard
  to read); a "repeat shrunk to mean" column added as a rung (the T2 control); sibling and co-twin pooled into one
  "relative" column (QTIM co-twin = same-age family member, zygosity unknown, 4 pairs); the sex-age mean drawn as an
  illustration (mean of the 16 drawn people closest over the whole face).
- Same tiles grouped by dataset; calibrated score c tiles; agreement matrices.

## 3. Agreement of rung order (`agreement.py`)
Spearman rho of the tile ranks over rungs both sides have (at least 5). Datasets: rho for each metric, mean over
metrics. Metrics: rho for each dataset, mean over datasets. ArcFace left out (5 shared rungs on one dataset).
- Brain, fundus and iris order the rungs alike (rho 0.87-0.97); faces agree with each other (0.98) and less with the
  rest (0.69-0.84), because faces rank the repeat image 6th to 7th.
- Most cross-dataset stable: MS-SSIM 0.95, persistence 0.92; least: amplitude deviation 0.77-0.81, ACC vs matched
  mean 0.78. Pixel and structure metrics agree with each other (0.84-0.99; MSE, RMSE, PSNR by construction);
  amplitude deviation is near 0 or reversed (down to -0.49 with lDDT).

## 4. Observations (descriptive, pre-QC where stated)
- T2 fails broadly: the repeat shrunk toward the mean beats the repeat for MSE on brain and iris (rank 1).
- Under MSE the white-noise control (dataset mean plus noise with the target's residual SD,
  `donor_ladder_brain_metrics.py:noise_fill`) ties with a relative on the brain: relative 0.50, noise 0.51, stranger
  0.55, overlapping intervals. Structural metrics (SSIM, lDDT) put noise last.
- Calibrated c: faces give the repeat 0.70-0.72 under MSE and SSIM, below the dataset mean (0.87); ACC vs matched
  mean drops the mean to 0.50 (stranger level); ArcFace gives the repeat 0.98-0.99; fundus and iris repeats 0.90-0.99.

## 4b. Night 8-9 Nov additions (Kate's paper scope, minimal items)
- Dataset card (`dataset_card.py`, `dataset_card.json/png`): persons, target images, images in frame (n/a where the
  JSON does not record it), windows, rungs, V = MSE of the dataset-mean fill, and repeat / other side / look-alike /
  stranger as multiples of V from one computation (`common_all.json`). Stranger sits at 1.96-2.02 everywhere; the
  repeat at 0.18-0.57 on brain, fundus and iris but 1.28-1.39 on faces; PAPILA has no repeat (n/a).
- Generative, perceptual and topology block (`generative_block.py`, `generative_tiles.png`, `ranks_generative.json`):
  LPIPS AlexNet and VGG, Betti matching matched / unmatched, persistence, iris code distance (R_self per rung) and
  set-level FID and FWD (per-rung set value), from fundus Amendment 9.15 and iris Amendment P3. LPIPS AlexNet also puts
  the repeat shrunk to the mean first (T2 fails); FID and FWD put the dataset mean last and agree with pixel metrics
  near rho 0. Fundus set-level bootstrap intervals do not always contain the point value (small-sample FID bias), so
  set-level ranks are descriptive only.
- 9 Nov update: faces F2 (`face_ladder_metrics.json` `panel_f2`, commit 85d5a02; LPIPS AlexNet only, Betti matching,
  persistence, FID, FWD) and brain Amendment 20 (`donor_ladder_brain_metrics_t23.json`, deep box, both sides; LPIPS
  AlexNet on central slices, persistence W1; pre-QC) added. Face "retest" = repeat, face rung "self" (R = 0 anchor)
  dropped; brain R is repeat-anchored (an affine map of the per-rung mean, so the rank order is unchanged); relative =
  sibling and co-twin pooled, weighted by targets. Still not run: brain Betti 3D (C++ build) and brain FID / FWD (own
  amendment), LPIPS VGG on faces and brain. Descriptive: faces under LPIPS put the mirror first and the repeat 4th,
  the shrunk repeat 5th (T2 holds); under Betti and persistence the shrunk repeat beats the repeat on faces and brain
  persistence (T2 fails); face FID and FWD put the repeat first and the dataset mean near last.

## 5. Window size follows the object (Kate, 8 Nov 2026: "да переделай честно")
One pixel window covers a different share of each object (brain about 1/10 of its length, faces 0.4 inter-ocular
distance, fundus 1/8 of the frame with PAPILA covering 1.6 x more retina than GRAPE, iris 1/4 of the radial band x
1/16 of the circumference), and the PAPILA sensitivity figure shows rung positions move with the window. Rule sent to
all four modality sessions: object extent L from the data, window w = f x L for f in {1/16, 1/8, 1/4}, the same f
everywhere, current window reported as its f, a cross-modality claim holds only if it holds at all three sizes. Old
results stay; the sweep is written as new versions (Kate: "только сохрани старые результаты а это сделай новыми").
Each modality records it in its own protocol (brain Amendment 21, face F3; fundus and iris pending).

## 6. Shared resources for the full 26-metric panel (Kate, 7 Nov 2026)
Approved: "скачивай веса конечно считай все что запланировали", "библиотеку ставь". One copy in
`C:\Projects\data\projects\identity-ladder-metric\models-20261107\`: LPIPS (AlexNet, VGG) and FID Inception weights
(TORCH_HOME), Betti-matching 2D (pure Python, nstucki/Betti-matching at 3e7eca26) and 3D (C++, needs a compiler;
not built), RETFound (downloaded by the fundus session). The machine is CPU-only; latent metrics need a GPU.
