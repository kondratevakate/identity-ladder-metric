# Protocol: pretrained inpainting models on the identity ladder (model fills, v1)

Written 16 to 17 Nov 2026, before any full run was scored. Branch `cvpr2027-model-fills`. Results go to
`results/model_fills_v1/` and to the private folder `C:\Projects\data\projects\identity-ladder-metric\model-fills-20261116-v1\`.
Nothing of the earlier runs is changed; stored rows are only read.

## 1. Question

The ladder calibrates metrics on fills of known relationship to the hidden region. A reviewer asked where the fills
of actual inpainting models sit. This protocol places published, pretrained models on the same scale: same targets,
windows, valid pixels and scoring code as the references (general protocol, Amendment 10: hidden region = 1/8 of
the object).

## 2. Models (weights are used as released; nothing is trained or tuned here)

| Key | Model and weights | Trained on | Licence | Applied to |
|---|---|---|---|---|
| `lama_places` | LaMa, big-lama, TorchScript export `big-lama.pt` from the IOPaint model release (sha256 344c77bb...9ea9) | Places | Apache-2.0 (code) | CelebA, IMDb-WIKI |
| `lama_celebahq` | LaMa `lama-celeba-hq/lama-fourier/best.ckpt` (sha256 3060742e...f281), generator code of `advimman/lama` at 786f593 | CelebA-HQ 256 | Apache-2.0 (code) | CelebA, IMDb-WIKI |
| `sd15_inpaint` | Stable Diffusion 1.5 inpainting, `stable-diffusion-v1-5/stable-diffusion-inpainting`, empty prompt, guidance 1.0, 20 steps, 512 px, seed fixed per (target, window) | LAION subsets | CreativeML OpenRAIL-M | CelebA, IMDb-WIKI (subset, section 5) |
| `brats24_unet` | Rank 1 of the BraTS 2024 inpainting challenge, Docker image `brainles/brats24_inpainting_ying_weng`, weights Zenodo record 14230865 (ensemble of 25 checkpoints, mean of their outputs) | BraTS glioma T1 with healthy-tissue masks | CC BY 4.0 (weights) | QTIM (subset, section 5) |

Leakage note: CelebA-HQ is a subset of CelebA, so `lama_celebahq` may have seen CelebA identities or images in
training. Its clean test is IMDb-WIKI; its CelebA numbers are reported with this note.
Domain note: `brats24_unet` was trained on tumour patients in the BraTS grid with large, irregular masks; QTIM is
healthy young adults, affinely registered to the MNI template, with 16 mm cubes. `lama_places` and `sd15_inpaint`
are generic models that never saw aligned grey faces.

## 3. What the model sees and returns

Faces. The model gets the whole 128 px frame of the target (aligned, z-scored grey level, 0 outside the face mask)
with ONE window hidden, and the window mask. Grey level is mapped to [0, 1] by the fixed affine
u = (clip(z, -4, 4) + 4) / 8, given as three equal channels at the model's resolution (256 px, SD 512 px; bicubic
up, area down, nearest for the mask); hidden pixels are zeroed in the input. The channel mean of the output is
mapped back with the inverse affine. Only the hidden window is taken from the output.

Brain. The model gets the skull-stripped scan (intensity times brain mask) with ONE 16 mm cube zeroed, and the cube
mask, in the 240 x 240 x 155 BraTS array. The 1 mm MNI array goes there by axis flips (RAS+ to LPS array order),
centred zero padding in x and y and a 155-slice z window (the brain mask is up to 162 slices tall; the window keeps
as much mask as possible and is otherwise centred on the cube). No interpolation of the scan or of the fill. The
model code is the challenge image's own (`Dataset_Inference.preprocess`, `model.forward`, the reverse statements of
`get_result_image`), run in memory; three cases are also run through the official entry point (`brats` package) and
must agree. The returned cube is z-scored with the target's own in-mask mean and SD. For the mirrored side (frame 1
of the ladder) the cube at the mirrored x position is hidden in the true scan and the returned cube is flipped.

## 4. Scoring

- Rung name `model_<key>`. Scored by the unchanged ladder functions: `face_ladder_metrics.score_position` (faces) and
  `donor_ladder_brain_metrics.score` on the stored region memmaps (brain). Valid pixels are the target's; a model
  fill is defined everywhere, so no pixel is added or dropped. Missing (cube not run) is missing, never zero.
- Gate before any model row is kept: the stored rows of the references are reproduced exactly by the same call
  (faces: every stored rung on the smoke targets, the dataset average, second image and mean of 16 on all targets;
  brain: the dataset average). The summary recomputes every reference value of `common_f8_v1.json` and must match.
- Reported: R on the random-person scale (Eq. 2 of the paper) with the 95 % bootstrap interval over families
  (faces) or targets (brain), for MSE, SSIM, TM-like, GDT-like, lDDT-like; paired differences model minus second
  image, dataset average, mean of the 16 nearest people, softened same person.
- LPIPS, FID and FWD of the references exist only at the 16 px panel window (f = 0.2) in faces, so the face models
  are also run at that window and scored with the panel's own functions (`model_fills_face_panel.py`); gates: stored
  LPIPS of the dataset average and stored FID and FWD of the random-person set are reproduced. Brain FID and FWD are
  not in the paper's set-level summary (small sets) and are not computed for the model.

## 5. Subsets (fixed before the runs they apply to, on compute cost alone)

- `sd15_inpaint`, f = 1/8: every 16th target of each face set (about 100 targets, the size of the F3 size-sweep
  subset), all windows. Panel window: only the stored set-level target cells (778 CelebA, 744 IMDb-WIKI).
  Reason: about 1.5 s per window on an L4.
- `brats24_unet`: one cube costs 25 forward passes of a 3D U-Net (about 40 s when the GPU is shared; measured in the
  smoke). Cubes: positions 0, 4, ..., 28 of the 32 random cubes, both sides (16 cubes per target). Targets in a
  fixed order: the 23 with a second scan first, then the others in bit-reversal order of their table index, so that
  any prefix is spread evenly (`job/order.json`, written 17 Nov before the run). The run stops on its GPU time
  budget (6 h), never on a result. References are recomputed on the same targets and cubes for the comparison.

## 6. Gates passed before the full runs (17 Nov)

- Faces smoke (`lama_places`, 4 targets per set, CPU): 11,952 of 11,952 stored rq4 rows reproduced; 368 model rows.
  QC sheet looked at: fills continuous with the context, no seam or intensity offset; errors at eyes, nostrils and
  mouth corners, where the model paints smooth skin.
- Brain smoke (`brats24_unet`, 3 targets x 3 cubes x 2 sides): 192 of 192 stored dataset-average rows reproduced; QC
  sheet looked at: fill inside the cube, intensity continuous with the shell, smooth, no sulcal detail.
- Brain, official entry point against the in-memory loop on three cases: worst difference 4.1e-07 of the cube
  maximum (float rounding); passed.
- Brain, array orientation: the BraTS file order (flips of x and y) was fixed in advance from the BraTS header
  convention; no BraTS scan was available to confirm it. On the 18 smoke cubes the other orders tried gave a slightly
  higher error (R under MSE 0.281 in the BraTS order, 0.304 without flips, 0.291 with z also flipped; SSIM 0.583,
  0.612, 0.607). The model is close to orientation-insensitive on these cubes, so this is weak support only.
- Faces, full run of both LaMa models at f = 1/8: 137,049 of 137,049 stored rows of the dataset average and the
  second image reproduced (largest difference 3.6e-15); QC sheets of 24 evenly spaced targets per set looked at.
- LaMa CelebA-HQ checkpoint: weights-only load is refused (the file holds a Lightning callback object), so it was
  unpickled on the throwaway VM; a listing of the pickle's imports shows only torch tensors, `OrderedDict` and that
  callback class.
- Kate's instruction of 16 Nov: assess the smoke yourself and continue autonomously.

## 7. Limits to state with the results

- Out-of-domain use of every model except `lama_celebahq` on faces; a low R of a model is a statement about the
  metric and the fill, not a ranking of models.
- One sample per window for the diffusion model (seed fixed); no seed average.
- Brain: affine registration only (pre-QC, as the whole brain ladder); subset of targets and cubes.
- Grey, aligned 128 px faces upsampled for the models; colour and background context are absent.

## 8. Amendment 1 (17 Nov 2026, written before any v2 run)

1. **Input version v2 for faces (bug fix).** In v1 the frame was upsampled to the model's resolution before the
   window was masked. The bicubic kernel then carried the hidden border pixels into the visible band next to the
   mask: with only hidden pixels changed, 276 visible input pixels at 256 px changed by up to 0.18 of the grey
   range. This breaks section 3 ("the model sees only the context"). v2 replaces the hidden pixels by the mean of
   the visible pixels within 3 px of the window BEFORE upsampling, then zeroes them at model resolution. Regression
   test: `tests/test_model_fills.py::test_model_input_does_not_depend_on_hidden_pixels` (two frames that differ only
   inside the window give a bit-identical model input). Every face run is repeated as v2 into `faces_v2/`; the v1
   fills and rows stay in `faces/` and are not used for any reported number. v1 values at f = 1/8 for the record
   (R under MSE): LaMa CelebA-HQ 0.111 (CelebA) and 0.119 (IMDb-WIKI), LaMa Places 0.120 and 0.131. The brain input
   has no interpolation and is not affected.
2. **Model on the RQ4 size curve (faces).** LaMa is also run at 5, 16 and 20 px (1/16, 1/5 and 1/4 of the face) on
   the F3 subset of 100 evenly spaced targets per set, the targets of `results/rq4_window_size.json`, and scored
   against the stored rows of those sizes (gate: dataset average and second image reproduced; the summary must
   reproduce the reference values of `rq4_window_size.json`). Reason: a model point at one size cannot show whether
   its place among the references depends on the size of the hidden region. Decided after the v1 result at 1/8 was
   seen, so it is exploratory.
3. **Preemption and budget.** The spot VM was preempted after 2.8 h; the brain run had used 1.42 h of its 6 h and
   finished about 8 targets. The run continues in the same fixed target order; the time used is now stored next to
   the fills (`time_used.json`), so the 6 h budget counts run time across restarts. If the restart lands on a
   slower GPU (T4), the budget stays 6 h of run time and fewer targets are reached.
4. **Spend.** Kate's hard cap for the whole task is 30 USD (17 Nov). Projection at the time of this amendment: at
   most 10 USD in total.
5. **Pins (after the commit security review of 17 Nov).** Weights are pinned by SHA-256 and refused on a mismatch
   (`model_fills_infer.PINNED_SHA256`: big-lama.pt 344c77bb...9ea9, LaMa CelebA-HQ best.ckpt 3060742e...f281; its
   config.yaml is c16dc611...2930); the LaMa repository is checked out at 786f5936b27fb3dacd2b1ad799e4de968ea697e7;
   the brain image is addressed by digest sha256:c59daaf3...d253. The full unpickle of the LaMa CelebA-HQ checkpoint
   needs the explicit `--allow-unpickle` flag and happens only after the hash check. The runs of 16 and 17 Nov used
   the same files (hashes and commit are in their logs and `infer_meta.json`) through scripts without these guards.
   The diffusion model is pinned by its Hub revision once the snapshot hash of the run is recorded
   (`out/sd15_snapshot.txt`).
