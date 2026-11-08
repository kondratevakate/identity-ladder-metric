# Handoff: iris metric panel (8 Nov 2026, after the forced reboot)

Owner files: `src/iris_ladder_metrics*.py`, `tests/test_iris_ladder_metrics.py`, `docs/PROTOCOL_iris_metrics.md`,
`results/iris_ladder_metrics.json`, `results/figures/fig_iris_*`.

Private outputs (UTIRIS redistribution is not granted, so nothing per-image or per-subject leaves this folder):
`C:\Projects\data\projects\identity-ladder-metric\iris-ladder-metrics-20261107-v1\`.

## Done

- Frame P2 (Worldcoin segmentation, 64x256 strip). Kate accepted it on 7 Nov 2026. Sets: UTIRIS NIR (primary), UTIRIS
  VIS and UPOL.
- Panel run on P2. Results are in `results/iris_ladder_metrics.json`. Figures: blocks, ratio and full blocks.
- Donor curve and organ maps, including the 16 look-alikes map:
  - `fig_iris_donor_ladder`;
  - `fig_iris_maps_common`;
  - iris rows in `fig_ladder_maps_all_modalities` (branch donor-ladder-bilateral, file `src/ladder_maps_all_figure.py`
    there, not committed by this session).
- Amendment P3 (protocol) and `src/iris_ladder_metrics_extra.py`:
  - LPIPS AlexNet/VGG, Betti matching, iris-code Hamming distance per window;
  - FID and FWD at set level.
- P3 smoke run (every 8th eye per side):
  - completed for all three sets, all survived the reboot;
  - private files: `rows_extra_*_P2_smoke.csv.gz`, `setlevel_extra_*_P2_smoke.json`, `qc_extra_*_P2.png`.
- Amendment P2e (3-sample margin from the mask edge): tested and rejected by Kate on 8 Nov 2026 ("Keep the shadow").
  The lid shadow is about 0.35 limbus radii wide. It is recorded as a weakness in Amendment P3, and P2 stays the
  frame.

## Update 8 Nov 2026 evening: full P3 done

- Full P3 run and merge finished for NIR, VIS and UPOL (all exit 0). Aggregates only are in the `amendment_p3` block of
  `results/iris_ladder_metrics.json` (`python -I src/iris_ladder_metrics_p3.py --preprocessing P2 --tag full`; input and
  code hashes stored); figures `results/figures/fig_iris_p3_blocks` and `fig_iris_p3_setlevel` (charts only, no eyes).
- Measured: H4 fails (T2, Holm 26 x 4, NIR: LPIPS, Betti matched and unmatched all p_holm = 1). Descriptively the
  dataset mean scores closer than a stranger on every per-window P3 metric except LPIPS-VGG (NIR R_self: Betti matched
  0.39, LPIPS-AlexNet 0.68, Betti unmatched 0.75, iris code 0.89). H5 holds on point values (FID mean 19.7x, FWD 416x the
  stranger) but several set-level bootstrap intervals exclude their own point (flagged). H6 holds (LPIPS R 0.19 < 0.32 < 1.0).
- Left: Kate's verdict on these metrics; window amendment and frame P2 unchanged (wait for Kate's lid-shadow sheets);
  14 blocked metrics; `src/metrics_board.py` has no iris section (not this session's file); QC sheet title says "smoke".

## Not done (before the full run)

- Still blocked, with no definition or model: Topology Distance, plausibility, identity x4, latent x3. This is part of
  the 14 metrics.

## Restart the full P3 run

The run takes about 19 h in a single process on a loaded machine, so it is sharded. Every target is in exactly one
shard. Seeds are fixed per target.

```
set TORCH_HOME=C:\Projects\data\projects\identity-ladder-metric\models-20261107\torch
set PYFWD_DIR=C:\Projects\data\projects\identity-ladder-metric\models-20261107\pytorchfwd-lib
set SNAP=C:\Projects\data\projects\identity-ladder-metric\iris-ladder-metrics-20261107-v1\src_snapshot_20261108
set FUNDUS_METRICS_SRC=%SNAP%\bilateral_src\fundus_ladder_metrics.py
set BRAIN_METRICS_SRC=%SNAP%\cvpr_src\donor_ladder_brain_metrics.py
python -I src/iris_ladder_metrics_extra.py run --preprocessing P2 --sets UTIRIS_NIR --threads 1 --shard 0/4
...  (shards 1/4, 2/4, 3/4; then the same for UTIRIS_VIS and UPOL)
python -I src/iris_ladder_metrics_extra.py merge --preprocessing P2 --sets UTIRIS_NIR --threads 4 --shards 4
```

Notes:
- Each shard prints progress and ETA per eye and saves `extra_<set>_P2_shard<i>of<n>_{rows.csv.gz,patches.npz}`
  privately.
- `merge` refuses to run unless the shards cover every target once. It then computes FID and FWD once and writes
  `rows_extra_<set>_P2_full.csv.gz`, `setlevel_extra_<set>_P2_full.json` and a QC sheet.
- Check free disk space before starting. The shards need tens of MB per set, not GB.
- Shared helper sources are snapshotted, so other sessions' live edits cannot change a running job. The snapshot is
  `src_snapshot_20261108`, with hashes in `SHA256_folders.txt`.
- Neither source matches the hash recorded in `results/iris_ladder_metrics.json` for the P2 panel run:
  - fundus: 8107dadc vs 1ade3d5a;
  - brain: 40555f18 vs 3ac5878f.
  Check whether that changes the panel fills before merging P3 with the panel.
- Status on 8 Nov 2026, after the reboot: UTIRIS NIR and VIS run as 3 shards each, with logs at
  `log_extra_P2_full_<set>_shard<i>of3.txt`. UPOL has not started yet.

## Open decisions for Kate

- P3 smoke QC verdict on the metrics themselves:
  - Betti matching may reward the dataset mean (H4 at risk);
  - FID and FWD show a pairing artefact: rungs built from the same people look closer.
- The amendment for the 14 still-blocked metrics: which definitions or models to fix (Topology Distance,
  plausibility density model, identity and latent encoders for iris).
- Lid shadow: decided ("keep"). Nothing open.
