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

## Not done
- F2 full run: stopped by the reboot with 0 of 36 jobs saved (one job = one position x all targets, about 40+ min each,
  so nothing reached disk). `parts_v6f/` is empty; nothing to recover.
- F3 smoke at w = 10: 20 of 94 parts saved and readable (`face-ladder-sizes-20261108-v3/parts_v6f_w10_n4/`); no rows,
  no QC sheet yet. F3 code for the F2 panel at other sizes (densities per size, LPIPS rule) is not written.
- `results/face_ladder_metrics.json` still holds tier-1 only; no F2/F3 numbers; metrics board not updated with them.

## Restart (only when Kate gives faces the CPU; iris and fundus go first)
Environment for the panel: `FUNDUS_METRICS_SRC=.../identity-ladder-bilateral/src/fundus_ladder_metrics.py`,
`BETTI_SRC=.../models-20261107/betti-matching-2d`, `TORCH_HOME=.../models-20261107/torch`.
1. Before restarting F2, split jobs into chunks of about 50 targets per position (as the brain t23 runner), so a crash
   loses minutes, not hours. Then `python src/face_ladder_metrics_panel.py rows --version _v6f --workers 3`
   (resumes from `parts_v6f/`), then `set` and `summary` (same flags). Estimate 10 to 17 h.
2. F3 smoke: `python src/face_ladder_sizes.py rows --version _v6f --denom 8 --limit 4 --workers 1` (resumes), then
   `qc --denom 8 --limit 4`; show the sheet to Kate before the sweep.
3. Disk: C: had 52 GB free after the reboot; F2 rows are about 1 GB.

## Kate's open decisions
- When faces get CPU again, and whether F2 (16 px) or the F3 sweep goes first.
- F3 cost: the F2 panel at three sizes runs on 100 targets per source (about 8 h), tier-1 on all targets
  (about 7.5 h). Accept as written in the amendment or cut?
