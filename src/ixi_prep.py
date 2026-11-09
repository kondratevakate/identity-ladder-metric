"""IXI preparation for the brain ladder (PROTOCOL_donor_ladder_brain.md, Amendment 23; preparation as Amendment 19).

Stages (private outputs only; scans are keyed ixi_000..ixi_099 by sample order, IXI IDs stay in the private manifest):
  select   eligibility from IXI.xls, stratified sample of 100 (centre x age tertile, seed 20261109), integrity check
           of every sampled T1 (full gzip read, nibabel load, 3D, finite voxels, sha256); writes the private manifest
           and the VM input folder (copies named by key) - runs on the laptop
  strip    SynthStrip masks in the pinned FreeSurfer 8.0.0 image (same digest and call as real_cases.strip) - VM
  prepare  N4 on a 2 mm copy, ANTs affine to the MNI 2009a symmetric template, 1 mm output, own mask by nearest
           neighbour - the same ANTs calls as real_cases.prepare_scan (checked by a test) - VM
  qc       sheet of three slices through both boxes (scan, mask contour) and the D7 automatic flags - laptop
The ladder itself is a separate stage (donor_ladder_brain_ixi.py).
"""
import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

SEED = 20261109
N_SAMPLE = 100
CENTRES = ("Guys", "HH", "IOP")
# pinned FreeSurfer 8.0.0 image, the same digest as real_cases.SEGMENTERS["synthseg"] (checked by a test)
SYNTHSTRIP_IMAGE = "freesurfer/freesurfer@sha256:f3e22b472afd9ca2b269d1b40c061ac891efaec070d09e387f4b024ea681e266"
# the two frozen boxes and their mirrors on the 1 mm template grid, as real_cases.BOXES (checked by a test)
BOXES = {"deep": (((-32, 0), (-19, 13), (-2, 22)), ((1, 33), (-19, 13), (-2, 22))),
         "parietal": (((-45, -13), (-65, -33), (43, 67)), ((13, 45), (-65, -33), (43, 67)))}
DATA_ROOT = os.environ.get("DATA_ROOT", r"C:\Projects\data")
RAW = os.path.join(DATA_ROOT, "datasets", "neuroimaging", "ixi", "version-unverified", "raw")
PRIVATE = os.path.join(DATA_ROOT, "projects", "identity-ladder-metric", "ixi-ladder-20261109-v1")
QTIM_TEMPLATE = os.path.join(DATA_ROOT, "projects", "identity-ladder-metric", "real-cases-run", "template")


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ select (laptop)

def eligible(xls, t1_files):
    """One T1 per IXI ID with a non-missing sex and age. IDs whose spreadsheet rows disagree on sex or by more than
    1 year on age are excluded and counted (Amendment 19, D7), never resolved by hand.
    xls: DataFrame with IXI_ID, SEX_ID (1=m, 2=f), AGE. t1_files: {ixi_id: (filename, centre)}.
    Returns (rows, counts) with rows = [{id, file, centre, sex, age}] sorted by id."""
    counts = {"t1_files": len(t1_files), "no_spreadsheet_row": 0, "missing_sex_or_age": 0, "rows_disagree": 0}
    rows = []
    for iid, (fname, centre) in sorted(t1_files.items()):
        g = xls[xls.IXI_ID == iid]
        if g.empty:
            counts["no_spreadsheet_row"] += 1
            continue
        g = g.dropna(subset=["SEX_ID (1=m, 2=f)", "AGE"])
        if g.empty:
            counts["missing_sex_or_age"] += 1
            continue
        sexes, ages = g["SEX_ID (1=m, 2=f)"].unique(), g.AGE.to_numpy(float)
        if len(sexes) > 1 or ages.max() - ages.min() > 1:
            counts["rows_disagree"] += 1
            continue
        rows.append({"id": int(iid), "file": fname, "centre": centre, "sex": int(sexes[0]), "age": float(ages.mean())})
    counts["eligible"] = len(rows)
    return rows, counts


def stratified_sample(rows, n=N_SAMPLE, seed=SEED):
    """Proportional allocation over centres (largest remainder), then over age tertiles within centre (largest
    remainder), random within strata. No selection by image content (STUDIES-025)."""
    rng = np.random.default_rng(seed)

    def alloc(sizes, total):
        sizes = np.asarray(sizes, float)
        q = total * sizes / sizes.sum()
        base = np.floor(q).astype(int)
        for k in np.argsort(-(q - base), kind="stable")[: total - base.sum()]:
            base[k] += 1
        return base

    by_c = {c: [r for r in rows if r["centre"] == c] for c in CENTRES}
    n_c = alloc([len(by_c[c]) for c in CENTRES], n)
    out = []
    for c, nc in zip(CENTRES, n_c):
        rc = sorted(by_c[c], key=lambda r: (r["age"], r["id"]))
        tert = np.array_split(np.arange(len(rc)), 3)                    # age tertiles within the centre
        for idx, nt in zip(tert, alloc([len(t) for t in tert], nc)):
            pick = rng.choice(idx, size=nt, replace=False)
            out += [rc[i] for i in sorted(pick)]
    return out


def integrity(path):
    """Full gzip read, nibabel load, 3D shape, finite voxels. Returns (ok, reason, sha256)."""
    import nibabel as nib
    try:
        with gzip.open(path, "rb") as f:
            while f.read(1 << 20):
                pass
        v = np.asarray(nib.load(path).dataobj)
        if v.ndim != 3:
            return False, f"not 3D: {v.shape}", None
        if not np.isfinite(v).all():
            return False, "non-finite voxels", None
    except Exception as e:  # noqa: BLE001 - recorded as an exclusion reason, never dropped silently
        return False, f"{type(e).__name__}", None
    return True, "", sha256(path)


def t1_files(raw=RAW):
    out = {}
    for d in sorted(os.listdir(os.path.join(raw, "ixi"))):
        for sub in os.listdir(os.path.join(raw, "ixi", d)):
            t1 = os.path.join(raw, "ixi", d, sub, "T1", "NIfTI")
            for f in os.listdir(t1) if os.path.isdir(t1) else []:
                if f.endswith("-T1.nii.gz"):
                    iid, centre = int(f[3:6]), f.split("-")[1]
                    if iid in out:
                        raise SystemExit(f"two T1 files for one ID (index {len(out)})")
                    out[iid] = (os.path.join(t1, f), centre)
    return out


def select():
    import pandas as pd
    files = t1_files()
    xls = pd.read_excel(os.path.join(RAW, "IXI.xls"))
    rows, counts = eligible(xls, files)
    sample = stratified_sample(rows)
    vm_in = os.path.join(PRIVATE, "vm_in")
    os.makedirs(vm_in, exist_ok=True)
    man, failed = [], 0
    for k, r in enumerate(sample):
        ok, why, h = integrity(r["file"])
        key = f"ixi_{k:03d}"
        man.append(dict(r, key=key, file=os.path.basename(r["file"]), ok=ok, reason=why, sha256=h))
        if ok:
            shutil.copy(r["file"], os.path.join(vm_in, key + ".nii.gz"))
        else:
            failed += 1
    counts.update(sampled=len(sample), integrity_failed=failed,
                  by_centre={c: sum(m["centre"] == c for m in man) for c in CENTRES})
    with open(os.path.join(PRIVATE, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "counts": counts, "scans": man}, f, indent=1)
    with open(os.path.join(vm_in, "sha256.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.writelines(f"{m['sha256']}  {m['key']}.nii.gz\n" for m in man if m["ok"])
    ages = np.array([m["age"] for m in man])
    print(json.dumps(dict(counts, age_min=float(ages.min()), age_median=float(np.median(ages)),
                          age_max=float(ages.max()), female=sum(m["sex"] == 2 for m in man))))


# ------------------------------------------------------------------------------------------------ strip, prepare (VM)

def strip(run, keys, threads):
    """SynthStrip in one container, the same call as real_cases.strip; masks keyed by scan key."""
    out = os.path.join(run, "strip")
    os.makedirs(out, exist_ok=True)
    todo = [k for k in keys if not os.path.exists(os.path.join(out, f"{k}_mask.nii.gz"))]
    with open(os.path.join(out, "strip_list.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.writelines(f"{k}.nii.gz {k}\n" for k in todo)
    loop = ('while read rel key; do mri_synthstrip -i "/bids/$rel" -m "/out/${key}_mask.nii.gz" -t %d >/dev/null 2>&1'
            ' || echo "$key failed"; done < /out/strip_list.txt' % threads)
    start = time.monotonic()
    r = subprocess.run(["docker", "run", "--rm", "-v", f"{os.path.join(run, 'in')}:/bids:ro", "-v", f"{out}:/out",
                        SYNTHSTRIP_IMAGE, "bash", "-c", loop], capture_output=True, text=True, check=False)
    if r.returncode:
        raise SystemExit(f"docker exited {r.returncode}: {r.stderr[-500:]}")
    failed = [w for w in r.stdout.split() if w.startswith("ixi_")]
    print(json.dumps({"stripped": len(todo) - len(failed), "failed": failed,
                      "seconds_per_scan": round((time.monotonic() - start) / max(len(todo), 1), 1)}), flush=True)


def box_mask(shape, affine, bounds):
    """Voxels whose centre lies in the half-open mm bounds of every axis, as real_cases.box_mask (checked by a test)."""
    ijk = np.indices(shape).reshape(3, -1)
    xyz = (affine[:3, :3] @ ijk + affine[:3, 3:4]).reshape((3,) + tuple(shape))
    m = np.ones(shape, bool)
    for a, (lo, hi) in zip(xyz, bounds):
        m &= (a >= lo) & (a < hi)
    return m


def prepare_scan(job):
    """N4, affine and 1 mm resampling of one stripped scan: the ANTs calls of real_cases.prepare_scan, unchanged."""
    run, key = job
    out = os.path.join(run, "prep", key)
    if os.path.exists(out + ".json"):
        with open(out + ".json", encoding="utf-8") as f:
            return json.load(f)
    import ants
    import nibabel as nib
    start = time.monotonic()
    meta = {"scan": key}
    try:
        tdir = os.path.join(run, "template")
        tpl1, tpl2 = (ants.image_read(os.path.join(tdir, f"template_{r}mm.nii.gz")) for r in (1, 2))
        brain1, brain2 = (nib.load(os.path.join(tdir, f"brain_{r}mm.nii.gz")).get_fdata() > 0 for r in (1, 2))
        native = ants.image_read(os.path.join(run, "in", key + ".nii.gz"))
        mask = ants.image_read(os.path.join(run, "strip", f"{key}_mask.nii.gz"))
        small = ants.resample_image(native, (2, 2, 2), use_voxels=False, interp_type=0)
        bias = ants.n4_bias_field_correction(small, shrink_factor=4, return_bias_field=True)
        corrected2 = small / bias
        mask2 = ants.resample_image_to_target(mask, small, interp_type="nearestNeighbor").numpy() > 0.5
        work = tempfile.mkdtemp(prefix=f"{key}_")
        reg = ants.registration(fixed=tpl2, moving=corrected2.new_image_like(corrected2.numpy() * mask2),
                                type_of_transform="Affine", mask=tpl2.new_image_like(brain2.astype(np.float32)),
                                mask_all_stages=True, aff_iterations=(1000, 500, 250, 50), random_seed=123,
                                outprefix=os.path.join(work, "a_"), verbose=False)
        corr2 = float(np.corrcoef(reg["warpedmovout"].numpy()[brain2], tpl2.numpy()[brain2])[0, 1])
        shutil.copy(reg["fwdtransforms"][0], out + "_affine.mat")
        native_bias = ants.resample_image_to_target(bias, native, interp_type="linear")
        corrected = native.new_image_like(native.numpy() / np.maximum(native_bias.numpy(), 1e-6))
        warped = ants.apply_transforms(tpl1, corrected, reg["fwdtransforms"], interpolator="linear").numpy()
        own = ants.apply_transforms(tpl1, mask, reg["fwdtransforms"], interpolator="nearestNeighbor").numpy() > 0.5
        shutil.rmtree(work, ignore_errors=True)
        affine = nib.load(os.path.join(tdir, "template_1mm.nii.gz")).affine
        nib.save(nib.Nifti1Image(warped.astype(np.float32), affine), out + ".nii.gz")
        nib.save(nib.Nifti1Image(own.astype(np.uint8), affine), out + "_brain.nii.gz")
        meta.update(template_corr=float(np.corrcoef(warped[brain1], tpl1.numpy()[brain1])[0, 1]),
                    template_corr_2mm=corr2, registration_seconds=time.monotonic() - start)
        for name, (box, mirror) in BOXES.items():
            meta[f"{name}_box_in_own_brain"] = float(own[box_mask(own.shape, affine, box)].mean())
            meta[f"{name}_mirror_in_own_brain"] = float(own[box_mask(own.shape, affine, mirror)].mean())
    except Exception as e:  # noqa: BLE001 - recorded as an exclusion reason (D7), never dropped silently
        meta["error"] = f"{type(e).__name__}: {e}"
    with open(out + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return meta


def prepare(run, keys, workers, threads):
    os.makedirs(os.path.join(run, "prep"), exist_ok=True)
    keys = [k for k in keys if os.path.exists(os.path.join(run, "strip", f"{k}_mask.nii.gz"))]
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(threads)
    with ProcessPoolExecutor(workers) as ex:
        metas = list(ex.map(prepare_scan, [(run, k) for k in keys]))
    secs = [m["registration_seconds"] for m in metas if "registration_seconds" in m]
    print(json.dumps({"prepared": len(metas), "errors": sum("error" in m for m in metas),
                      "seconds_median": float(np.median(secs)) if secs else None}), flush=True)


# ------------------------------------------------------------------------------------------------ qc (laptop)

def d7_flags(metas):
    """D7 automatic rules as QTIM (real_cases.qc_flags): error or deep box under 95% inside the own mask -> excluded;
    template correlation below median - 3 MAD -> flagged for visual review (not excluded). The parietal box and every
    cube are handled by the ladder's per-cube 95% validity rule (missing rows, never zero), as in QTIM."""
    ok = [m for m in metas if "error" not in m]
    tc = np.array([m["template_corr"] for m in ok])
    med = float(np.median(tc))
    mad = float(np.median(np.abs(tc - med)))
    excl, flag = {}, []
    for m in metas:
        if "error" in m:
            excl[m["scan"]] = "error"
        elif m["deep_box_in_own_brain"] < 0.95:
            excl[m["scan"]] = "deep box under 95% inside own mask"
        elif m["template_corr"] < med - 3 * mad:
            flag.append(m["scan"])
    return {"excluded": excl, "flagged": flag, "template_corr_median": med, "template_corr_mad": mad}


def qc_sheet(prep, keys, path):
    """One row per scan: axial, coronal, sagittal through the centre of each box; mask contour in red."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    fig, ax = plt.subplots(len(keys), 6, figsize=(13, 2.3 * len(keys)), squeeze=False)
    for r, k in enumerate(keys):
        ni = nib.load(os.path.join(prep, k + ".nii.gz"))
        v, m = ni.get_fdata(), nib.load(os.path.join(prep, k + "_brain.nii.gz")).get_fdata() > 0
        inv = np.linalg.inv(ni.affine)
        for b, (name, (box, _)) in enumerate(BOXES.items()):
            c = np.round(inv[:3, :3] @ np.array([np.mean(x) for x in box]) + inv[:3, 3]).astype(int)
            lim = np.percentile(v[m], 99.5)
            for j, (sl, lab) in enumerate(((v[:, :, c[2]], m[:, :, c[2]]), (v[:, c[1], :], m[:, c[1], :]),
                                            (v[c[0], :, :], m[c[0], :, :]))):
                a = ax[r, 3 * b + j]
                a.imshow(sl.T, origin="lower", cmap="gray", vmin=0, vmax=lim)
                a.contour(lab.T, levels=[0.5], colors="red", linewidths=0.5)
                a.set_xticks([]); a.set_yticks([])
                a.set_title(f"{k} {name} {('axial', 'coronal', 'sagittal')[j]}", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("select", "strip", "prepare", "qc"))
    ap.add_argument("--run", help="VM run folder holding in/, template/, strip/, prep/")
    ap.add_argument("--keys", nargs="*", help="scan keys; default all in <run>/in")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--threads", type=int)
    a = ap.parse_args()
    if a.stage == "select":
        return select()
    if a.stage == "qc":
        prep = os.path.join(PRIVATE, "prep")
        metas = [json.load(open(os.path.join(prep, f), encoding="utf-8")) for f in sorted(os.listdir(prep))
                 if f.endswith(".json")]
        keys = a.keys or [m["scan"] for m in metas]
        fl = d7_flags(metas)
        qc_sheet(prep, keys, os.path.join(PRIVATE, f"qc_{len(keys)}.png"))
        print(json.dumps({k: (len(v) if isinstance(v, (dict, list)) else v) for k, v in fl.items()}))
        json.dump(fl, open(os.path.join(PRIVATE, "d7.json"), "w", encoding="utf-8"), indent=1)
        return
    if not (a.run and a.threads):
        raise SystemExit("--run and --threads are required on the VM (no default)")
    if a.stage == "prepare" and not a.workers:
        raise SystemExit("--workers is required (no default)")
    keys = a.keys or sorted(f[:-7] for f in os.listdir(os.path.join(a.run, "in")) if f.endswith(".nii.gz"))
    if a.stage == "strip":
        return strip(a.run, keys, a.threads)
    prepare(a.run, keys, a.workers, a.threads)


if __name__ == "__main__":
    sys.exit(main())
