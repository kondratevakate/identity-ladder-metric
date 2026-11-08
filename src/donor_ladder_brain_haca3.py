"""HACA3+ contrast harmonization of the ladder's QTIM volumes (PROTOCOL_donor_ladder_brain.md, Amendment 17).

Stages (run in order; each is resumable and checks hashes of what it reuses):
  (each scan is cropped/padded to the model grid 192 x 224 x 192 on the fly; no brain voxel may be cropped)
  target    theta, eta and norm value of every ses-01 scan from the model's own target encoder; the per-component
            median is frozen in haca3_target.json (ses-02 never enters it)
  run       HACA3+ inference with --target-theta/--target-eta/--norm-val = the frozen median; fused output put back on
            the native 197 x 233 x 189 grid
  qc        stage-1 gate on the 23 repeat pairs: sheets (private) and before/after session shift per box (aggregate)

Run with the pilot's environment (it has torchio, tensorboard and torch 2.10 CPU):
  C:/Projects/02_academia/studies/identity-ladder-metric/external/registration-env/Scripts/python.exe

Pre-QC sensitivity analysis; never replaces the v3 ladder. Private images stay in C:/Projects/data; the repo gets
aggregates only.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import donor_ladder_brain as B  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RELEASE = os.path.join(r"C:\Projects\02_academia\studies\identity-ladder-metric", "external", "haca3-plus-4ec722b8")
UPSTREAM = "4ec722b8b09523cfe94df8c9e9cd6362c6625c3c"
WEIGHTS = ("haca3-plus_model_38538dab", "fusion_afafc572")
MODEL_SHAPE = (192, 224, 192)
OUT = os.path.join(B.PROJ, "donor-ladder-brain-haca3-20261106-v1")
RESULT = os.path.join(ROOT, "results", "donor_ladder_brain_haca3_stage1.json")
TARGET_SLICES = slice(100, 120)      # as haca3_plus.test for a target image
NUM_BATCHES = 56                     # as the 4 Nov pilot


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dump(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def weights():
    out = {}
    for name in WEIGHTS:
        pointer = os.path.join(RELEASE, "models", f"{name}.pt")
        digest = open(pointer).read().split("sha256:")[1].split()[0]
        w = pointer.replace(".pt", ".verified.pt")
        if sha(w) != digest:
            raise RuntimeError(f"weight digest mismatch: {name}")
        out[name] = w
    return out


def key(row):
    return f"{row.person}_{row.session}"


def table():
    t = B.scans()
    t["key"] = [key(r) for r in t.itertuples()]
    t["image"] = [p.replace("_brain.nii.gz", ".nii.gz") for p in t.path]
    return t


def offset(native_shape, model_shape=MODEL_SHAPE):
    return (np.array(native_shape) - np.array(model_shape)) // 2


def to_model(vol, mask, model_shape=MODEL_SHAPE):
    """Centred crop/pad to the model grid; raises if any mask voxel would be cropped."""
    off = offset(vol.shape, model_shape)
    lo, hi = np.maximum(off, 0), np.minimum(off + np.array(model_shape), vol.shape)
    src = tuple(slice(a, b) for a, b in zip(lo, hi))
    dst = tuple(slice(a, b) for a, b in zip(lo - off, hi - off))
    if np.count_nonzero(mask[src]) != np.count_nonzero(mask):
        raise RuntimeError("model grid would crop brain-mask voxels")
    out = np.zeros(model_shape, dtype=np.float32)
    out[dst] = vol[src]
    return out


def from_model(vol, native_shape):
    """Inverse of to_model; voxels outside the model grid are NaN (missing, never zero)."""
    off = offset(native_shape, vol.shape)
    lo, hi = np.maximum(off, 0), np.minimum(off + np.array(vol.shape), native_shape)
    src = tuple(slice(a, b) for a, b in zip(lo - off, hi - off))
    dst = tuple(slice(a, b) for a, b in zip(lo, hi))
    out = np.full(native_shape, np.nan, dtype=np.float32)
    out[dst] = vol[src]
    return out


def model_affine(affine, native_shape):
    a = affine.copy()
    a[:3, 3] = nib.affines.apply_affine(affine, offset(native_shape))
    return a


def prepare_one(r, dst):
    """Model-grid input for one scan (temporary file; the disk is near full, so inputs are not kept)."""
    img = nib.load(r.image)
    mask = nib.load(r.path).get_fdata(dtype=np.float32) > 0.5
    vol = img.get_fdata(dtype=np.float32)
    if not np.isfinite(vol).all():
        raise RuntimeError(f"non-finite input {r.key}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    nib.save(nib.Nifti1Image(to_model(vol, mask), model_affine(img.affine, vol.shape)), dst)
    return {"input_sha256": sha(r.image), "mask_sha256": sha(r.path), "prepared_sha256": sha(dst)}


def encode_target(rows, w):
    """theta, eta and norm value per scan, exactly as haca3_plus.test computes them for a --target-image."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
    sys.path.insert(0, RELEASE)
    import torch
    from haca3_plus.modules.model import HACA3
    from haca3_plus.test import obtain_single_image
    m = HACA3(beta_dim=5, theta_dim=2, eta_dim=2, pretrained_haca3=w["haca3-plus_model_38538dab"])
    m.theta_encoder.eval()
    m.eta_encoder.eval()
    out = {}
    tmp = os.path.join(OUT, "tmp", "target_input.nii.gz")
    with torch.set_grad_enabled(False):
        for i, r in enumerate(rows.itertuples(), 1):
            rec = prepare_one(r, tmp)
            img, _, norm = obtain_single_image(tmp, False)
            x = img.permute(2, 1, 0).permute(0, 2, 1).flip(1)[TARGET_SLICES].unsqueeze(1)
            theta, _ = m.theta_encoder(x)
            eta = m.eta_encoder(x)
            out[r.key] = {"theta": theta.mean(0).flatten().tolist(), "eta": eta.mean(0).flatten().tolist(),
                          "norm": float(norm), **rec}
            if i % 20 == 0:
                print(f"encoded {i}/{len(rows)}", flush=True)
    os.remove(tmp)
    return out


def median_target(per_scan):
    """Per-component median over scans; returns the frozen target."""
    th = np.array([v["theta"] for v in per_scan.values()])
    et = np.array([v["eta"] for v in per_scan.values()])
    nv = np.array([v["norm"] for v in per_scan.values()])
    return {"theta": np.median(th, 0).tolist(), "eta": np.median(et, 0).tolist(), "norm": float(np.median(nv)),
            "n_scans": len(per_scan),
            "iqr": {"theta": np.subtract(*np.percentile(th, [75, 25], 0)).tolist(),
                    "eta": np.subtract(*np.percentile(et, [75, 25], 0)).tolist(),
                    "norm": float(np.subtract(*np.percentile(nv, [75, 25])))}}


def target(t, w):
    path = os.path.join(OUT, "haca3_target.json")
    if os.path.exists(path):
        print("target frozen:", json.load(open(path))["target"])
        return json.load(open(path))["target"]
    if os.path.isdir(os.path.join(OUT, "harmonized")) and os.listdir(os.path.join(OUT, "harmonized")):
        raise RuntimeError("harmonized outputs exist but no frozen target: refuse to define it after the fact")
    per = encode_target(t[t.session == "ses-01"], w)
    tg = median_target(per)
    dump(os.path.join(OUT, "haca3_target_per_scan.json"), per)
    dump(path, {"target": tg, "protocol": "Amendment 17", "upstream_commit": UPSTREAM,
                "code_sha256": sha(__file__), "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    print("target frozen:", tg)
    return tg


def run_one(r, tg, w):
    k = r.key
    raw = os.path.join(OUT, "raw", f"{k}.nii.gz")
    fused = raw.replace(".nii.gz", "_harmonized_fusion.nii.gz")
    final = os.path.join(OUT, "harmonized", f"{k}.nii.gz")
    if os.path.exists(final):
        return k, "done"
    img = nib.load(r.image)
    native_shape, affine = img.shape, img.affine
    src = os.path.join(OUT, "tmp", f"{k}_input.nii.gz")
    if not os.path.exists(fused):
        rec = prepare_one(r, src)
        env = dict(os.environ, PYTHONPATH=RELEASE, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2",
                   MKL_NUM_THREADS="2", TORCH_FORCE_WEIGHTS_ONLY_LOAD="1")
        cmd = [sys.executable, "-u", "-m", "haca3_plus.test", "--in-path", src,
               "--target-theta", *map(str, tg["theta"]), "--target-eta", *map(str, tg["eta"]),
               "--norm-val", str(tg["norm"]), "--out-path", raw,
               "--harmonization-model", w["haca3-plus_model_38538dab"], "--fusion-model", w["fusion_afafc572"],
               "--num-batches", str(NUM_BATCHES), "--no-bg-removal"]
        with open(os.path.join(OUT, "logs", f"{k}.log"), "w", encoding="utf-8") as log:
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=5400)
        os.remove(src)
    else:
        rec = {}
    h = nib.load(fused)
    vol = h.get_fdata(dtype=np.float32)
    if vol.shape != MODEL_SHAPE or not np.isfinite(vol).all():
        raise RuntimeError(f"bad HACA3+ output {k}: shape {vol.shape}")
    nib.save(nib.Nifti1Image(from_model(vol, native_shape), affine), final)
    rec.update({"fused_sha256": sha(fused), "harmonized_sha256": sha(final)})
    dump(final.replace(".nii.gz", ".json"), rec)
    for o in ("axial", "coronal", "sagittal", "fusion"):        # disk is near full; the record keeps the hashes
        p = raw.replace(".nii.gz", f"_harmonized_{o}.nii.gz")
        if os.path.exists(p):
            os.remove(p)
    return k, "new"


def run(t, tg, w, keys, workers):
    for d in ("raw", "harmonized", "logs", "tmp"):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    rows = {r.key: r for r in t.itertuples()}
    t0 = time.time()

    def job(k):
        try:
            return run_one(rows[k], tg, w)
        except Exception as e:          # recorded, never silent; the run continues with the other scans
            return k, f"FAILED: {type(e).__name__}: {e}"

    status = {}
    with ThreadPoolExecutor(workers) as ex:
        for i, (k, s) in enumerate(ex.map(job, keys), 1):
            status[k] = s
            print(f"{i}/{len(keys)} {s} {(time.time() - t0) / 60:.1f} min", flush=True)
    failed = {k: s for k, s in status.items() if s.startswith("FAILED")}
    dump(os.path.join(OUT, "run_status.json"), {"n": len(keys), "failed": failed,
                                                "minutes": (time.time() - t0) / 60})
    if failed:
        print(f"{len(failed)} FAILED, see run_status.json")


def zscore(vol, mask):
    v = vol[mask]
    return (vol - v.mean()) / v.std()


def box_slices(affine):
    inv = np.linalg.inv(affine)
    out = {}
    for name, b in B.BOXES_MM.items():
        lo = np.rint(inv @ np.array([b[0][0], b[1][0], b[2][0], 1]))[:3].astype(int)
        hi = np.rint(inv @ np.array([b[0][1], b[1][1], b[2][1], 1]))[:3].astype(int)
        out[name] = tuple(slice(a, c) for a, c in zip(np.minimum(lo, hi), np.maximum(lo, hi)))
    return out


def session_shift(z1, z2, m, box):
    """Mean of ses-02 minus ses-01 (z units) over box voxels inside both masks; NaN if none."""
    both = m[box] & np.isfinite(z1[box]) & np.isfinite(z2[box])
    return float(np.mean(z2[box][both] - z1[box][both])) if both.any() else float("nan")


def qc(t):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rep = sorted(set(t.person[t.session == "ses-02"]))
    row = {(r.person, r.session): r for r in t.itertuples()}
    os.makedirs(os.path.join(OUT, "qc"), exist_ok=True)
    pairs, sheet = [], []
    for i, p in enumerate(rep):
        a, b = row[(p, "ses-01")], row[(p, "ses-02")]
        ha, hb = (os.path.join(OUT, "harmonized", f"{r.key}.nii.gz") for r in (a, b))
        if not (os.path.exists(ha) and os.path.exists(hb)):
            pairs.append({"pair": i, "status": "not run"})
            continue
        img = nib.load(a.image)
        boxes = box_slices(img.affine)
        ma = nib.load(a.path).get_fdata(dtype=np.float32) > 0.5
        mb = nib.load(b.path).get_fdata(dtype=np.float32) > 0.5
        m = ma & mb
        z = {}
        for tag, (pa, pb) in {"before": (a.image, b.image), "after": (ha, hb)}.items():
            va, vb = nib.load(pa).get_fdata(dtype=np.float32), nib.load(pb).get_fdata(dtype=np.float32)
            z[tag] = (zscore(va, ma), zscore(vb, mb))
        rec = {"pair": i, "status": "ok"}
        for name, box in boxes.items():
            for tag in ("before", "after"):
                rec[f"{name}_{tag}"] = session_shift(*z[tag], m, box)
        pairs.append(rec)
        sheet.append((i, z, m, boxes))
    # sheets: 6 pairs per page; per pair and box, axial mid-slice: ses-01 / ses-02 before, ses-01 / ses-02 after
    for page in range(0, len(sheet), 6):
        chunk = sheet[page:page + 6]
        fig, axes = plt.subplots(len(chunk), 8, figsize=(16, 2.2 * len(chunk)), squeeze=False)
        for r, (i, z, m, boxes) in enumerate(chunk):
            c = 0
            for name, box in boxes.items():
                zs = (box[2].start + box[2].stop) // 2
                pad = 12
                sl = (slice(box[0].start - pad, box[0].stop + pad), slice(box[1].start - pad, box[1].stop + pad), zs)
                for tag in ("before", "after"):
                    for s in (0, 1):
                        im = np.where(m[sl], z[tag][s][sl], np.nan).T
                        ax = axes[r, c]
                        ax.imshow(im, origin="lower", cmap="gray", vmin=-2.5, vmax=2.5)
                        ax.add_patch(plt.Rectangle((pad - .5, pad - .5), box[0].stop - box[0].start,
                                                   box[1].stop - box[1].start, fill=False, ec="#eb6834", lw=.8))
                        ax.set_title(f"pair {i} {name} {tag} ses-0{s + 1}", fontsize=6)
                        ax.axis("off")
                        c += 1
        fig.suptitle("HACA3+ stage 1 (Amendment 17): z in own brain mask, shared window -2.5..2.5; "
                     "outside both masks = missing (white)", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "qc", f"repeat_pairs_{page // 6 + 1}.png"), dpi=130)
        plt.close(fig)
    ok = [p for p in pairs if p["status"] == "ok"]
    summ = {}
    for name in B.BOXES_MM:
        for tag in ("before", "after"):
            v = np.array([p[f"{name}_{tag}"] for p in ok], dtype=float)
            v = v[np.isfinite(v)]
            summ[f"{name}_{tag}"] = {"mean": float(v.mean()) if v.size else None,
                                     "mean_abs": float(np.abs(v).mean()) if v.size else None, "n": int(v.size)}
    tg = json.load(open(os.path.join(OUT, "haca3_target.json")))
    out = {"status": "pre-QC sensitivity (Amendment 17, stage 1); Kate's visual verdict pending",
           "target": tg["target"], "n_pairs": len(rep), "n_done": len(ok),
           "session_shift_z": summ, "pairs": pairs,
           "definition": "mean over box voxels inside both brain masks of z(ses-02) - z(ses-01), z in each scan's own"
                         " brain mask; before = N4 + affine, after = HACA3+ to the cohort-median contrast"}
    dump(RESULT, out)
    print(json.dumps(summ, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["target", "run", "qc"])
    ap.add_argument("--set", choices=["repeats", "all", "one"], default="repeats")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t = table()
    if a.stage == "target":
        target(t, weights())
    elif a.stage == "run":
        tg = target(t, weights())
        rep = set(t.person[t.session == "ses-02"])
        keys = list(t.key) if a.set == "all" else [k for k, p in zip(t.key, t.person) if p in rep]
        if a.set == "one":
            keys = keys[:1]
        run(t, tg, weights(), keys, a.workers)
    else:
        qc(t)


if __name__ == "__main__":
    main()
