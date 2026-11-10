# Executive summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_e97dbbc5328e", "created_at": "2026-11-10T08:48:25+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-11-10T08:48:25+00:00"}
-->
**69 of 69 printed numbers equal their result files** (within the
printed precision) and still appear in the cited section of the paper.

Each claim page states the claim, its status, the scale and n, and one row per printed number: the text value, the
file value, the source file and key, and the commit that last changed that file. Text issues are flagged, not fixed.

| Claim | Short label | Status | Checks passed |
| --- | --- | --- | --- |
| C1 | Brain MSE: mean and mirror | provisional (pre-QC) | 2/2 |
| C2 | Brain T2: shrunk repeat | provisional (pre-QC) | 6/6 |
| C3 | Mean of 16 look-alikes | provisional (pre-QC) | 11/11 |
| C4 | IXI scanner site | provisional (pre-QC) | 6/6 |
| C5 | HACA3+ session shift | provisional (pre-QC) | 5/5 |
| C6 | Fundus and iris cohorts | provisional (pre-QC) | 12/12 |
| C7 | Face donor: pixel vs identity | verified | 4/4 |
| C8 | Per-window errors reward shrinkage | provisional (pre-QC) | 5/5 |
| C9 | Protein-style scores track MSE | provisional (pre-QC) | 3/3 |
| C10 | LPIPS rejects the mean only | provisional (pre-QC) | 6/6 |
| C11 | FID/FWD reject averaging only | provisional (pre-QC) | 7/7 |
| C12 | Window size keeps the rung order | provisional (pre-QC) | 2/2 |

**Text issues flagged**

- **C8**: 4_results.tex calls lDDT 'the only per-window score that places white noise far beyond a stranger (1.76)'. That holds among the error / similarity scores (MSE 0.86, SSIM 1.03, TM-like 0.95), but LPIPS AlexNet (1.84), LPIPS VGG (1.52), Betti matching (1.66) and persistence (3.90) are also per-window and also place noise beyond a stranger. Suggest 'the only error or similarity score'.
- **C11**: The set-size rule and the brain exclusion were fixed after seeing the brain value; the text says so (post hoc), which is correct and should stay.

Paper sections read: `paper/cvpr2027/sec/0_abstract.tex`, `paper/cvpr2027/sec/4_results.tex`, `paper/cvpr2027/sec/5_synthesis.tex`. Local logbook only; not published.
