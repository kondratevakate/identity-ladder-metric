"""BraTS inpainting model on the QTIM ladder cubes, run inside the challenge Docker image (identity ladder, f = 1/8).

The image brainles/brats24_inpainting_ying_weng ships plain PyTorch code in /mlcube_project (unet.py, dataset3D.py,
inference.py). Its own entry point reads one NIfTI pair per case; 14,144 cases would be mostly file I/O. This script
runs the SAME code in memory: `Dataset_Inference.preprocess` (max normalisation, 208 x 208 x 144 crop centred on the
mask, map to [-1, 1]), `model.forward(voided, mask)` for every checkpoint of the ensemble, mean over the ensemble, and
the reverse statements of `Dataset_Inference.get_result_image`. `dump` writes a few cases as NIfTI in the challenge
layout so the official entry point can be run on them; `check` compares its output with the in-memory fills.

Grid: the 1 mm MNI array (RAS+, 197 x 233 x 189) goes into the 240 x 240 x 155 BraTS array by axis flips (default
`--flips 01`: RAS+ to the LPS array order of the BraTS files), centred zero padding in x and y, and a 155-slice z
window that keeps as much of the brain mask as possible and is otherwise centred on the cube. No interpolation.

A case = (target, cube position, frame). Frame 0 hides the cube where it was drawn; frame 1 hides the cube at the
mirrored x position (index a - i) and returns it flipped, i.e. in the mirrored frame of the ladder code. The model
always sees the scan in its true orientation with one cube set to zero.

Usage (inside the container, /work = the job folder):
  python3 brain_container.py infer --job /work/job --ckpt /ckpt --out /work/out [--targets 0,110,220] [--positions 0,5]
                                   [--budget-hours 6]   (targets are run in the order given; resumable per target)
  python3 brain_container.py dump  --job JOB --out DIR --cases 0:0:0,0:5:1      (numpy + nibabel only)
  python3 brain_container.py check --job JOB --fills OUT --official DIR --cases 0:0:0,0:5:1
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

BRATS = (240, 240, 155)
AFFINE = np.array([[-1., 0, 0, 0], [0, -1, 0, 239], [0, 0, 1, 0], [0, 0, 0, 1]])   # header of the BraTS files
CROP = [208, 208, 144]                                                               # inference.py default


def geometry(job, mask, start, frame, flips):
    """Slices that place the flipped scan in the BraTS array and the cube inside it.
    Returns (flip tuple, src z slice, dst x/y offsets, cube slices in the BraTS array)."""
    n = job["shape"]
    size = job["size"]
    s = list(start)
    if frame == 1:
        s[0] = job["mirror_constant"] - start[0] - (size - 1)          # mirrored x range, ascending
    if s[0] < 0 or s[0] + size > n[0]:
        return None
    for d in flips:
        s[d] = n[d] - s[d] - size
    zz = np.flatnonzero(mask.any((0, 1)))
    b0, b1 = int(zz[0]), int(zz[-1])
    if 2 in flips:
        b0, b1 = n[2] - 1 - b1, n[2] - 1 - b0
    if b1 - b0 + 1 > BRATS[2]:
        lo, hi = b0, b1 - BRATS[2] + 1
    else:
        lo, hi = max(0, b1 - BRATS[2] + 1), min(b0, n[2] - BRATS[2])
    z0 = int(np.clip(s[2] + size // 2 - BRATS[2] // 2, lo, hi))
    off = ((BRATS[0] - n[0]) // 2, (BRATS[1] - n[1]) // 2)
    cube = (slice(off[0] + s[0], off[0] + s[0] + size), slice(off[1] + s[1], off[1] + s[1] + size),
            slice(s[2] - z0, s[2] - z0 + size))
    if cube[2].start < 0 or cube[2].stop > BRATS[2]:
        return None
    return z0, off, cube


def to_brats(arr, z0, off, flips):
    fl = tuple(slice(None, None, -1) if d in flips else slice(None) for d in range(3))
    a = arr[fl]
    out = np.zeros(BRATS, arr.dtype)
    out[off[0]:off[0] + a.shape[0], off[1]:off[1] + a.shape[1], :] = a[:, :, z0:z0 + BRATS[2]]
    return out


def cube_back(c, frame, flips):
    """Cube cut from the BraTS array back to the ladder frame: undo the flips, then mirror x for frame 1."""
    fl = tuple(slice(None, None, -1) if d in flips else slice(None) for d in range(3))
    c = c[fl]
    return c[::-1] if frame == 1 else c


def case_arrays(job, vol, mask, start, frame, flips):
    g = geometry(job, mask, start, frame, flips)
    if g is None:
        return None
    z0, off, cube = g
    v = to_brats(vol, z0, off, flips).astype(np.float64)
    hide = np.zeros(BRATS, np.float64)
    hide[cube] = 1.0
    v[cube] = 0.0                                                     # the model never sees the hidden cube
    return v, hide, cube


def parse_cases(txt):
    return [tuple(int(x) for x in c.split(":")) for c in txt.split(",")]


def load_target(job_dir, i):
    z = np.load(os.path.join(job_dir, f"tgt_{i:03d}.npz"))
    return z["vol"], z["mask"]


def dump(a):
    import nibabel as nib
    job = json.load(open(os.path.join(a.job, "job.json")))
    flips = () if a.flips == "none" else tuple(int(c) for c in a.flips)
    for k, (i, pi, fr) in enumerate(parse_cases(a.cases)):
        vol, mask = load_target(a.job, i)
        v, hide, _ = case_arrays(job, vol, mask, job["starts"][pi], fr, flips)
        name = f"BraTS-GLI-{k:05d}-000"
        os.makedirs(os.path.join(a.out, name), exist_ok=True)
        nib.save(nib.Nifti1Image(v.astype(np.float32), AFFINE), os.path.join(a.out, name, f"{name}-t1n-voided.nii.gz"))
        nib.save(nib.Nifti1Image(hide.astype(np.float32), AFFINE), os.path.join(a.out, name, f"{name}-mask.nii.gz"))
        print(name, (i, pi, fr), flush=True)


def check(a):
    import nibabel as nib
    job = json.load(open(os.path.join(a.job, "job.json")))
    flips = () if a.flips == "none" else tuple(int(c) for c in a.flips)
    worst = 0.0
    for k, (i, pi, fr) in enumerate(parse_cases(a.cases)):
        vol, mask = load_target(a.job, i)
        _, _, cube = case_arrays(job, vol, mask, job["starts"][pi], fr, flips)
        hits = glob.glob(os.path.join(a.official, "**", f"BraTS-GLI-{k:05d}-000*.nii.gz"), recursive=True)
        if len(hits) != 1:
            raise SystemExit(f"official output of case {k}: {len(hits)} files")
        off = cube_back(nib.load(hits[0]).get_fdata()[cube], fr, flips)
        mine = np.load(os.path.join(a.fills, f"fill_{i:03d}.npz"))["fill"][pi, fr]
        d = float(np.max(np.abs(off - mine)))
        rel = d / float(np.abs(off).max())
        worst = max(worst, rel)
        print(f"case {k} {(i, pi, fr)}: max |official - in-memory| = {d:.3g} ({rel:.2e} of the cube maximum)", flush=True)
    print(json.dumps({"worst_relative_difference": worst, "passed": worst < 1e-4}), flush=True)
    if worst >= 1e-4:
        raise SystemExit("in-memory fills differ from the official entry point")


def infer(a):
    import torch
    sys.path.insert(0, "/mlcube_project")
    import unet
    from dataset3D import Dataset_Inference
    from pathlib import Path
    dev = "cuda"
    job = json.load(open(os.path.join(a.job, "job.json")))
    flips = () if a.flips == "none" else tuple(int(c) for c in a.flips)
    ckpts = sorted(str(p) for p in Path(a.ckpt).rglob("*.ckpt"))
    if not ckpts:
        raise SystemExit(f"no checkpoint under {a.ckpt}")
    models = [unet.model.load_from_checkpoint(c, map_location=torch.device(dev)).eval().to(dev) for c in ckpts]
    ds = Dataset_Inference(Path("/nonexistent"), crop_shape=CROP, center_on_mask=True)
    n_t = len(job["stats"])
    targets = [int(x) for x in a.targets.split(",")] if a.targets else list(range(n_t))
    positions = [int(x) for x in a.positions.split(",")] if a.positions else list(range(len(job["starts"])))
    os.makedirs(a.out, exist_ok=True)
    size = job["size"]
    meta = {"image": "brainles/brats24_inpainting_ying_weng", "checkpoints": [os.path.basename(c) for c in ckpts],
            "crop_shape": CROP, "flips": list(flips), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    json.dump(meta, open(os.path.join(a.out, "infer_meta.json"), "w"), indent=1)
    t0, n_done, n_todo = time.time(), 0, 0
    todo = [i for i in targets if not os.path.exists(os.path.join(a.out, f"fill_{i:03d}.npz"))]
    print(f"targets {len(targets)} (to run {len(todo)}), positions {len(positions)}, ensemble {len(models)}", flush=True)
    # time already spent by earlier (preempted) runs is kept next to the fills, so the budget survives a restart
    used_path = os.path.join(a.out, "time_used.json")
    used0 = json.load(open(used_path))["seconds"] if os.path.exists(used_path) else a.used_hours * 3600
    for i in todo:
        used = used0 + time.time() - t0
        if a.budget_hours and used / 3600 >= a.budget_hours:                    # stop rule: time only, never a result
            print(f"BUDGET_REACHED after {n_todo} targets in this run, {used / 3600:.2f} h in total", flush=True)
            break
        vol, mask = load_target(a.job, i)
        fill = np.full((len(job["starts"]), 2, size, size, size), np.nan, np.float32)
        maxv = np.full((len(job["starts"]), 2), np.nan, np.float32)
        cases = [(pi, fr) for pi in positions for fr in (0, 1)]
        for b0 in range(0, len(cases), a.batch):
            chunk, xs, ms, info = cases[b0:b0 + a.batch], [], [], []
            for pi, fr in chunk:
                c = case_arrays(job, vol, mask, job["starts"][pi], fr, flips)
                if c is None:
                    continue                                          # cube leaves the grid: missing, never zero
                v, hide, cube = c
                mv, box, x, m = ds.preprocess(v, hide)
                xs.append(x)
                ms.append(m)
                info.append((pi, fr, float(mv), box, cube))
            if not xs:
                continue
            X, Mk = torch.stack(xs).float().to(dev), torch.stack(ms).to(dev)
            with torch.no_grad():
                pred = np.mean([mdl.forward(X, Mk).float().cpu().numpy() for mdl in models], axis=0)
            for k, (pi, fr, mv, box, cube) in enumerate(info):
                p = (pred[k, 0] + 1) / 2 * mv                         # reverse of normalize and of the max scaling
                pm = np.zeros_like(p)
                mk = Mk[k, 0].cpu().numpy()
                pm[mk] = p[mk]
                full = np.zeros(BRATS, np.float32)
                full[box] = pm
                fill[pi, fr] = cube_back(full[cube], fr, flips)
                maxv[pi, fr] = mv
            n_done += len(info)
        p = os.path.join(a.out, f"fill_{i:03d}.npz")
        np.savez_compressed(p + ".tmp.npz", fill=fill, max_v=maxv)
        os.replace(p + ".tmp.npz", p)
        json.dump({"seconds": used0 + time.time() - t0}, open(used_path, "w"))
        n_todo += 1
        el = time.time() - t0
        print(f"target {n_todo}/{len(todo)}: {n_done} cubes, {el / 60:.1f} min, {el / max(n_done, 1):.2f} s per cube, "
              f"ETA {el / n_todo * (len(todo) - n_todo) / 60:.0f} min", flush=True)
    print("INFER_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("infer", "dump", "check"))
    ap.add_argument("--job", required=True)
    ap.add_argument("--out")
    ap.add_argument("--ckpt")
    ap.add_argument("--targets", default="")
    ap.add_argument("--positions", default="")
    ap.add_argument("--cases", default="")
    ap.add_argument("--fills")
    ap.add_argument("--official")
    ap.add_argument("--flips", default="01", help="axes flipped before placing the scan in the BraTS array")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--budget-hours", type=float, default=0.0, help="infer: start no new target after this time")
    ap.add_argument("--used-hours", type=float, default=0.0,
                    help="infer: hours spent by earlier runs when no time_used.json exists yet")
    a = ap.parse_args()
    {"infer": infer, "dump": dump, "check": check}[a.stage](a)


if __name__ == "__main__":
    main()
