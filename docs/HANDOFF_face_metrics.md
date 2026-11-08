# Handoff: face metric panel (Amendments F2, F3), 8 Nov 2026, after the forced reboot

Protocol: `docs/PROTOCOL_face_metrics.md` (F2 and F3 written before any of their results). Code, tests, the tier-1 JSON
and figures are in git, commit f2ed748.
Private data: `C:\Projects\data\projects\identity-ladder-metric\`.

## Done
- F2 prep (`face-ladder-metrics-20261108-v2/prep_v6f/`): raw ArcFace norms (gate passed, min cosine 0.99958), PLDA per
  source, window densities. Intact.
- F2 smoke on 4 targets per source: v1 reproduced exactly (4596 rows, max |dMSE| = 0); set level and summary written;
  QC sheets `qc_panel_*_v6f_private.png`, `qc_window_*_v6f_private.png`. Kate said "Запускай" after seeing them.
- Speed fix: one BLAS thread per worker and a cached Cholesky factor (164 s -> 32 s per position; values unchanged).
- F3 geometry: L = 79.55 px (CelebA), 79.47 px (IMDb-WIKI); the 16 px window is f = 0.20; sizes w = 5, 10, 20 px
  (ring 3, 5, 10), positions 215/214, 47/45, 8/8. Code `src/face_ladder_sizes.py`, tests `tests/test_face_ladder_sizes.py`.

## Night 8-9 Nov (done)
- F2 full: 869972 rows reproduce v1 exactly; set level (FID, FWD, topology) on 1000 cells per source; aggregates in
  `results/face_ladder_metrics.json` (commit 85d5a02). Rows stay private in `face-ladder-metrics-20261108-v2/`.
- F3 smoke at w = 10 passed my visual QC (`face-ladder-sizes-20261108-v3/qc_fills_*_v6f_w10_private.png`); minimal F3
  (F3.1, 100 targets per source, 5/10/20 px): rung order Spearman >= 0.93 for MSE, MAE, SSIM, Pearson, TM/GDT-like on
  both sources, lDDT 0.80 (CelebA) / 0.87 (IMDb-WIKI); `results/face_ladder_sizes.json`, `fig_face_window_sizes.*`.
  Small windows pull the look-alike and oracle rungs closer to the target; noise drops below stranger at 5 px.

## Open
- Limitation (F3.1): the F2 panel (LPIPS, PLDA, topology, set level) runs only at 16 px; not at 5/10/20 px.
- `src/metrics_board.py` (not mine) does not read `results/face_ladder_metrics.json`; face F2/F3 are not on the board.
- Master checkout files the face code reads but git does not track (commit there by Kate or the master owner):
  `src/donor_ladder_face.py` (sha256 f134561c...0fe103), `results/donor_ladder_face_v6f.json` (b550a0d2...518e543),
  `results/donor_ladder_face_v6.json`. `FACE_SRC` in `src/face_ladder_metrics.py` points at the master copy.
- Plausibility (Gaussian) rates the mean fills as most plausible (a mode property); report it as a limitation.
