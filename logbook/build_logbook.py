"""Build the local Trackio claims logbook for the CVPR 2027 paper (one page per claim).

Usage: python -I -X utf8 logbook/build_logbook.py
Writes logbook/.trackio (generated; rebuilt from scratch on every run). Local only: no Space id, no bucket, nothing
is published (Kate decides that). Pattern: icml-repro/reproductions/xmae/repro_xmae/build_logbook.py; pitfalls from
icml-repro/notes/HANDOFF.md: no manual trackio.init(), short page titles, UTF-8 writes (-X utf8 on Windows).
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from trackio import logbook as lb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import claims as C  # noqa: E402

PROJECT = HERE / ".trackio"
TITLE = "Identity Ladder (CVPR 2027): claims logbook"


def commit_of(src):
    """Last commit touching a results file (its own repository for the fundus worktree); 'n/a' if unknown."""
    p = Path(C.path(src))
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%h %cs", "--", p.name], cwd=p.parent,
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or "n/a"
    except (OSError, subprocess.SubprocessError):
        return "n/a"


def create():
    if PROJECT.exists():
        shutil.rmtree(PROJECT)   # generated output only
    old_cwd, old_find = Path.cwd(), lb.find_project_dir
    try:
        os.chdir(HERE)
        lb.find_project_dir = lambda *a, **k: None   # do not attach to a parent project
        made = Path(lb.create_logbook(title=TITLE, space_id=None))
    finally:
        lb.find_project_dir = old_find
        os.chdir(old_cwd)
    if made.resolve() != PROJECT.resolve():
        raise RuntimeError(f"unexpected logbook path: {made}")


def fmt(v):
    if v is None:
        return "missing"
    return f"{v:.4g}" if abs(v) < 1000 else f"{v:.0f}"


def main():
    results = C.run_all()
    commits = {}
    create()
    meta = lb.read_metadata(PROJECT)
    meta.update({"space_id": None, "private": True, "tags": ["cvpr2027", "identity-ladder", "claims"],
                 "embed_content": False})
    lb.write_metadata(PROJECT, meta)

    ex = lb.ensure_page(PROJECT, "Executive summary")
    pages = {c["id"]: lb.ensure_page(PROJECT, f"{c['id']} {c['label']}") for c in C.CLAIMS}
    nr = lb.ensure_page(PROJECT, "Not run and limits")

    rows, n_ok, n_all, flags = [], 0, 0, []
    for c in C.CLAIMS:
        r = results[c["id"]]
        ok = sum(1 for x in r if x["match"] and x["in_text"] is not False)
        n_ok, n_all = n_ok + ok, n_all + len(r)
        rows.append(f"| {c['id']} | {c['label']} | {c['status']} | {ok}/{len(r)} |")
        flags += [f"- **{c['id']}**: {f}" for f in c.get("flags", [])]
    lb.add_markdown_cell(PROJECT, ex, f"""**{n_ok} of {n_all} printed numbers equal their result files** (within the
printed precision) and still appear in the cited section of the paper.

Each claim page states the claim, its status, the scale and n, and one row per printed number: the text value, the
file value, the source file and key, and the commit that last changed that file. Text issues are flagged, not fixed.

| Claim | Short label | Status | Checks passed |
| --- | --- | --- | --- |
{chr(10).join(rows)}

**Text issues flagged**

{chr(10).join(flags) if flags else "- none"}

Paper sections read: `{C.ABS}`, `{C.RES}`, `{C.SYN}`. Local logbook only; not published.
""", title="Executive summary")
    cid = lb.last_cell_id(PROJECT, page=ex)
    if cid:
        lb.set_cell_pinned(PROJECT, cid, pinned=True, page=ex)

    for c in C.CLAIMS:
        r = results[c["id"]]
        tab = ["| Printed | File value | Match | In text | Source (key) | Commit |", "| --- | --- | --- | --- | --- | --- |"]
        for chk, x in zip(c["checks"], r):
            src = chk[4]
            commits.setdefault(src, commit_of(src))
            where = x["tex"].split("/")[-1] if x["tex"] else "derived"
            tab.append(f"| {x['label']}: {x['text']} | {fmt(x['value'])} | {'yes' if x['match'] else '**NO**'} | "
                       f"{where if x['in_text'] else ('n/a' if x['in_text'] is None else '**missing**')} | "
                       f"`{src.replace('bilateral:', 'bilateral worktree: ')}` ({x['key']}) | {commits[src]} |")
        fl = "\n".join(f"- {f}" for f in c.get("flags", [])) or "- none"
        lb.add_markdown_cell(PROJECT, pages[c["id"]], f"""**{c['status'].upper()}.** {c['claim']}

Scale and n: {c['scale']}.

{chr(10).join(tab)}

Check: the printed number equals the file value within half a unit of its last printed digit, and the printed
snippet is present in the cited .tex section (`logbook/claims.py`, test `tests/test_logbook_claims.py`).

Text issues:
{fl}
""", title=c["label"])
        lb.add_code_cell(PROJECT, pages[c["id"]], output=json.dumps(r, indent=1), title=f"{c['id']} machine-readable",
                         code_paths=["logbook/claims.py"], language="python")

    lb.add_markdown_cell(PROJECT, nr, """Not run, with the reason (from the result files):

- Latent metrics (pullback, Fisher-Rao, LDDMM, plausibility): need a generative model with an inversion encoder per
  modality; options and prices are with Kate; nothing run.
- Brain Betti matching 3D: not available (3D and 2D libraries disagreed, brain Amendment 24 B); 2.5D used instead.
- IXI LPIPS, persistence, FID, FWD: not run (IXI scores tier-1 metrics and Betti 2.5D only).
- Brain FID / FWD: computed but excluded from Fig. 2 (repeat set 138 of 1326 windows; FID biased upward for small sets).
- Fundus window-size sweep: not run.

Limits: brain, fundus and iris results precede the registration and mask QC gates (provisional); the set-level rule
and the brain exclusion were fixed after seeing the brain value; the cross-modality analysis is exploratory.
""", title="Not run and limits")
    print(f"logbook written to {PROJECT}: {n_ok}/{n_all} checks pass")
    return 0 if n_ok == n_all else 1


if __name__ == "__main__":
    sys.exit(main())
