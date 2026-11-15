# HANDOFF: RQ4 window size, f = 1/8 (Amendment 10), finished 15 Nov 2026 12:56

Worktree `C:\Projects\02_academia\studies\_worktrees\identity-ladder-rq4size`, branch `cvpr2027-rq4-size` (from
`cvpr2027-paper` 90897a4). Committed on this branch, not pushed. Protocol: `docs/PROTOCOL_donor_ladder_general.md`
section 10, points 1 to 8 (f = 1/16 kept on the curves as the labelled interpolation end, Kate 14 Nov).
Private rows and logs: `C:\Projects\data\projects\identity-ladder-metric\rq4-window-size-20261114-v1\`
(`queue.sh`, `queue.log` with the real exit code of every step).

## Outputs (aggregates only)
- `results/rq4_window_size.json`, `results/figures/fig_rq4_window_size.{png,pdf}`: R_self(MSE) of look-alike (k = 1),
  mean of 16, oracle, dataset mean, true person and shrunk true person against f, 95 % bootstrap intervals.
- `results/cross_modality/common_f8_v1.json`: `common_all.json` format at f = 1/8. QTIM and IXI copied (already
  1/8); faces 10 px, iris 10 px, PAPILA 10 px, GRAPE 11 px recomputed; retina tier-1 metrics only.

## Gates (recomputed k = 1 look-alike equals the stored rows)
Brain 8 mm 14144/14144 (9e-16), brain 32 mm 8840/8840 (0.0), faces 5/10 px subset and full 10 px (272941/272941),
iris and retina smokes with checkpoints at 10 and 16 px (<= 7e-15). All 15 queue steps rc = 0.

## Not run, by protocol
IXI at 8 and 32 mm (no size geometry); UTIRIS VIS and UPOL at 5 and 20 px; f = 1/16 for iris NIR and retina on
every 4th eye only (curves drawn paired on those eyes).

## Code notes
- `src/rq4_checkpoint.py`: block checkpoints (pickles in the private folder only) for the iris and fundus wrappers.
- `src/rq4_brain_lookalike16_size.py`: memory gate; the first cube runs alone to measure the worker peak (1.37 GB at
  32 mm), then a cube starts only if available RAM - 2 GB reserve - growth of busy workers >= that peak. The growth
  term was fixed after the run to count busy workers only (the run itself used the stricter version: 1 to 3 workers,
  65 min for 19 cubes, no paging).
- `tests/test_cross_modality.py`: provenance map gained `common_f8_v1.json` -> `src/rq4_window_size.py`.
