"""FastSurfer segmentation of the donor-ladder boxes on the GCP VM (Amendment 15, "Input clip"). Pre-QC.

Two local steps around the VM runner (`gcp/fastsurfer_run.sh`):
  stage   -> clip every volume of a target to [0, c_t] (c_t = 99.9th percentile of the nonzero voxels of the
             target's oracle), write the clipped copies to a staging folder for upload, log c_t and, per box and
             core rung, the scored fill voxels above c_t. The built volumes are never modified.
  collect -> turn the downloaded `aparc.DKTatlas+aseg.deep.mgz` files into `label_fastsurfer_<name>.npy`
             (same `to_grid` as donor_ladder_brain_seg.segment) and refuse empty label volumes.
Volumes, labels and per-target logs stay in the private folder; the repository only gets run aggregates.
"""
import argparse
import json
import os
import shutil
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import donor_ladder_brain as B  # noqa: E402
import donor_ladder_brain_seg as S  # noqa: E402

CLIP_PERCENTILE = 99.9
FLAG_SHARE = 0.01
STAGE_DIR = os.path.join(S.PRIVATE, "gcp_stage")
CLIP_LOG = os.path.join(S.PRIVATE, "clip_log.json")
DL_DIR = os.path.join(S.PRIVATE, "local_out")   # labels of the local CPU run (gcp_out: the GPU tests)
RUN_JSON = os.path.join(B.ROOT, "results", "donor_ladder_brain_seg_gcp_run.json")


def clip_level(oracle):
    """c_t: the 99.9th percentile of the nonzero voxels of the target's oracle volume."""
    nz = oracle[oracle > 0]
    if nz.size == 0:
        raise ValueError("oracle volume has no nonzero voxels")
    return float(np.percentile(nz, CLIP_PERCENTILE))


def clip(vol, c):
    return np.clip(vol, 0, c).astype(np.float32)


def fill_above(vol_box, valid, c):
    """Scored fill voxels (valid mask, box shape) above the clip level: count and share."""
    n = int(valid.sum())
    above = int((vol_box[valid] > c).sum())
    return {"n_scored": n, "n_above": above, "share": above / n if n else float("nan")}


def stage_target(tid, pos):
    base = os.path.join(S.PRIVATE, tid)
    info = json.load(open(os.path.join(base, "fills.json")))
    oracle_img = nib.load(os.path.join(base, "in", "oracle.nii.gz"))
    oracle = np.asarray(oracle_img.dataobj, dtype=np.float32)
    c = clip_level(oracle)
    out = {"c_t": c, "oracle_voxels_clipped": int((oracle > c).sum()), "oracle_max_over_c": float(oracle.max() / c),
           "boxes": {}, "flags": []}
    dst = os.path.join(STAGE_DIR, tid)
    os.makedirs(dst, exist_ok=True)
    pis = S.box_positions(pos)
    for name in info["volumes"]:
        img = oracle_img if name == "oracle" else nib.load(os.path.join(base, "in", f"{name}.nii.gz"))
        v = oracle if name == "oracle" else np.asarray(img.dataobj, dtype=np.float32)
        if name in S.CORE:
            for box, pi in pis.items():
                if name not in info["boxes"][box]["fills"]:
                    continue
                valid = np.load(os.path.join(base, f"valid_{box}_{name}.npy"))
                r = fill_above(v[S.box_slices(pos, pi)], valid, c)
                out["boxes"].setdefault(box, {})[name] = r
                if r["share"] > FLAG_SHARE:
                    out["flags"].append(f"{box}/{name}: {r['share']:.3%} of scored fill voxels above c_t")
        path = os.path.join(dst, f"{name}.nii.gz")
        if not os.path.exists(path):
            nib.save(nib.Nifti1Image(clip(v, c), img.affine), path + ".part.nii.gz")
            os.replace(path + ".part.nii.gz", path)
    return out


def collect_target(tid, names):
    """Downloaded labels -> label_fastsurfer_<name>.npy on the volume grid. Returns {name: status}."""
    base = os.path.join(S.PRIVATE, tid)
    ref = nib.load(os.path.join(base, "in", "oracle.nii.gz"))
    status = {}
    for n in names:
        src = os.path.join(DL_DIR, f"{tid}_{n}", "mri", "aparc.DKTatlas+aseg.deep.mgz")
        dst = os.path.join(base, f"label_fastsurfer_{n}.npy")
        if os.path.exists(dst):
            status[n] = "done"
            continue
        if not os.path.exists(src):
            status[n] = "missing"
            continue
        lab = S.to_grid(src, ref)
        if not (lab > 0).any():
            status[n] = "empty"   # FastSurfer can exit 0 with an empty segmentation; never scored
            continue
        np.save(dst, lab)
        status[n] = "done"
    return status


def run_summary(status_tsv, usage_log, n_total):
    """Aggregates of the local batch from the runner's status.tsv and usage.log (copied into PRIVATE).
    Only counts and seconds, no subject or file names."""
    done, failed = set(), set()
    for line in open(status_tsv, encoding="utf-8"):
        f = line.rstrip("\n").split("\t")
        if len(f) >= 2:
            (done if f[1] == "DONE" else failed).add(f[0])
    failed -= done
    secs = []
    for line in open(usage_log, encoding="utf-8"):
        f = line.rstrip("\n").split("\t")
        if len(f) >= 7 and f[2] == "seg_cpu" and f[5] == "rc=0":
            secs.append(int(f[6].rstrip("s")))
    s = np.array(secs, dtype=float)
    return {"n_total": n_total, "n_done": len(done), "n_failed": len(failed), "n_left": n_total - len(done) - len(failed),
            "seconds_per_volume": {"n": int(s.size), "median": float(np.median(s)) if s.size else None,
                                   "p10": float(np.percentile(s, 10)) if s.size else None,
                                   "p90": float(np.percentile(s, 90)) if s.size else None}}


def cleanup_stage(tid):
    """Remove a target's staged copies after the upload has been verified."""
    shutil.rmtree(os.path.join(STAGE_DIR, tid), ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("stage", "collect", "run"))
    ap.add_argument("--targets", nargs="+", required=True, help="target folders, e.g. t009 t033")
    ap.add_argument("--source", choices=("local_out", "gcp_out"), default="local_out",
                    help="where the FastSurfer labels were brought back")
    a = ap.parse_args()
    global DL_DIR
    DL_DIR = os.path.join(S.PRIVATE, a.source)
    if a.step == "stage":
        pos, _, _ = B.positions(np.random.default_rng(S.SEED))
        log = json.load(open(CLIP_LOG)) if os.path.exists(CLIP_LOG) else {}
        for tid in a.targets:
            log[tid] = stage_target(tid, pos)
            json.dump(log, open(CLIP_LOG, "w"), indent=1)
            r = log[tid]
            print(f"staged {tid}: oracle voxels clipped {r['oracle_voxels_clipped']}, flags {len(r['flags'])}",
                  flush=True)
        return
    if a.step == "run":
        n_total = sum(len(json.load(open(os.path.join(S.PRIVATE, t, "fills.json")))["volumes"]) for t in a.targets)
        r = run_summary(os.path.join(S.PRIVATE, "local_status.tsv"), os.path.join(S.PRIVATE, "local_usage.log"),
                        n_total)
        r.update({"targets": len(a.targets), "device": "cpu (Kate's laptop, WSL Ubuntu-24.04), 2 runners x 4 threads",
                  "fastsurfer": "v2.5.3 native, torch 2.7.1+cpu, --seg_only --no_cereb --no_hypothal --no_biasfield",
                  "input": "clipped to [0, c_t], Amendment 15 'Input clip'",
                  "note": "seconds include volumes timed while other jobs shared the CPU and the last ones before the "
                          "8 Nov 2026 crash (disk full); not a clean D8 runtime"})
        json.dump(r, open(RUN_JSON, "w"), indent=1)
        print(json.dumps(r))
        return
    out = {}
    for tid in a.targets:
        names = json.load(open(os.path.join(S.PRIVATE, tid, "fills.json")))["volumes"]
        out[tid] = collect_target(tid, names)
    bad = {t: {n: s for n, s in st.items() if s != "done"} for t, st in out.items()}
    bad = {t: v for t, v in bad.items() if v}
    print(json.dumps({"targets": len(out), "not_done": bad}))


if __name__ == "__main__":
    main()
