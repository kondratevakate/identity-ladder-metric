"""Model side of the model-fill runs (faces): standalone, needs only numpy and torch, so it runs on a GPU VM that
holds nothing but the job file (aligned z-scored grey target frames, window positions, no names or identifiers).

Job file (written by `model_fills_face.py export`): npz with, per source s, `X_s` [n_targets, S, S] float16 frames,
`pos_s` [n_pos, 2] window corners (y0, x0), `tidx_s` target indices, and the scalars `w` (window side) and `size`.
Optional `cells_s` [n, 2] (row of X_s, window index): only these windows are run and the others stay NaN (not run).

The model sees the whole frame with ONE window hidden and returns that window. Grey level: fixed affine
u = (clip(z, -ZCLIP, ZCLIP) + ZCLIP) / (2 ZCLIP) to [0, 1], three equal channels in, channel mean out, inverse affine
back. Model resolution `--model-size` px (area down), mask nearest. Input version v3: every visible pixel the model
sees is computed from visible pixels of the frame only, and no invented value is mixed into it (`--input-mode`):
  norm     mask-aware bicubic upsampling: bicubic(frame * visible) / bicubic(visible); the default
  nearest  pixel replication (no interpolation at all); sensitivity variant
  native   the model runs at the frame resolution, no resampling; sensitivity variant (--model-size = frame size)
After that the hidden pixels are zeroed. v1 upsampled before masking (hidden border values leaked into the context);
v2 replaced the hidden pixels by the mean of a 3 px ring before upsampling, which drew a faint outline around the
window that the models continued into the hole (PROTOCOL_model_fills.md, amendments 1 and 2).

Backends:
  torchscript  model(image[B,3,H,W] in [0,1], mask[B,1,H,W]) -> image (big-lama TorchScript export)
  lama_repo    generator built from the advimman/lama repository (`--lama-repo`, `--config`), weights from a
               training checkpoint (`generator.*` keys), forward of its DefaultInpaintingTrainingModule
  sd_inpaint   diffusers inpainting pipeline (`--ckpt` = model id or folder), fixed seed per (target, window)

Usage: python model_fills_infer.py --job JOB.npz --model NAME --backend B --ckpt PATH --out DIR [--device cuda]
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

ZCLIP = 4.0
SEED = 20261116
INPUT_VERSION = "v3"             # v1: hidden values leaked into the context; v2: prefill drew an outline (see above)
INPUT_MODES = ("norm", "nearest", "native")
NORM_FLOOR = 0.5                 # floor of the visible weight; visible pixels have at least 0.77, hidden ones are zeroed


def z_to_u(z):
    return (np.clip(z, -ZCLIP, ZCLIP) + ZCLIP) / (2 * ZCLIP)


def u_to_z(u):
    return u * (2 * ZCLIP) - ZCLIP


def sha256(path):
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Base:
    """Frame in, frame out. Subclasses implement `forward(x, m)` on [B,3,H,W] in [0,1] and [B,1,H,W] in {0,1}."""

    def __init__(self, device, model_size, frame_size):
        import torch
        self.torch, self.device, self.ms, self.fs = torch, device, model_size, frame_size
        self.info = {}
        self.input_mode = "norm"

    def model_input(self, z, hide):
        """Masked frame at the model resolution and its mask. No visible value depends on a hidden pixel."""
        torch, F = self.torch, self.torch.nn.functional
        x = torch.from_numpy(z_to_u(z).astype(np.float32))[:, None].to(self.device)
        m0 = torch.from_numpy(hide.astype(np.float32))[:, None].to(self.device)
        v = 1 - m0
        if self.input_mode == "native":
            if self.ms != self.fs:
                raise SystemExit("input mode native needs --model-size equal to the frame size")
            xh, m = x * v, m0
        elif self.input_mode == "nearest":
            xh = F.interpolate(x * v, size=self.ms, mode="nearest")
            m = F.interpolate(m0, size=self.ms, mode="nearest")
        elif self.input_mode == "norm":
            num = F.interpolate(x * v, size=self.ms, mode="bicubic", align_corners=False)
            den = F.interpolate(v, size=self.ms, mode="bicubic", align_corners=False)
            xh = (num / den.clamp(min=NORM_FLOOR)).clamp(0, 1)
            m = F.interpolate(m0, size=self.ms, mode="nearest")
        else:
            raise SystemExit(f"unknown input mode {self.input_mode}")
        return xh * (1 - m), m                                  # hidden pixels carry no target information

    def __call__(self, z, hide, keys=None):
        torch, F = self.torch, self.torch.nn.functional
        x, m = self.model_input(z, hide)
        with torch.no_grad():
            o = self.forward(x.repeat(1, 3, 1, 1), m, keys)
        o = F.interpolate(o.float().mean(1, keepdim=True), size=self.fs, mode="area")[:, 0].clamp(0, 1)
        return u_to_z(o.cpu().numpy().astype(np.float64))


class TorchScript(Base):
    def __init__(self, ckpt, device, model_size, frame_size):
        super().__init__(device, model_size, frame_size)
        torch = self.torch
        self.net = torch.jit.load(ckpt, map_location=device).eval()
        with torch.no_grad():
            o = self.net(torch.full((1, 3, model_size, model_size), 0.5, device=device),
                         torch.zeros((1, 1, model_size, model_size), device=device))
        if tuple(o.shape) != (1, 3, model_size, model_size):
            raise RuntimeError(f"unexpected output shape {tuple(o.shape)}")
        self.scale = 255.0 if float(o.max()) > 1.5 else 1.0     # some exports return 0..255
        self.info = {"output_scale": self.scale}

    def forward(self, x, m, keys):
        return self.net(x, m) / self.scale


class LamaRepo(Base):
    def __init__(self, ckpt, device, model_size, frame_size, repo, config, allow_unpickle=False):
        super().__init__(device, model_size, frame_size)
        torch = self.torch
        sys.path.insert(0, repo)
        from omegaconf import OmegaConf
        from saicinpainting.training.modules import make_generator
        cfg = OmegaConf.load(config)
        self.concat = bool(cfg.training_model.get("concat_mask", True))
        gen = make_generator(cfg, **cfg.generator)
        try:
            state = torch.load(ckpt, map_location="cpu", weights_only=True)["state_dict"]
            how = "weights_only"
        except Exception as e:                                   # checkpoint holds non-tensor objects
            # A full unpickle runs code from the file. It is allowed only for a checkpoint whose SHA-256 was pinned
            # and verified by the caller (PINNED_SHA256) and only with the explicit opt-in flag.
            if not allow_unpickle:
                raise SystemExit(f"weights-only load refused ({type(e).__name__}); a full unpickle needs "
                                 "--allow-unpickle and a pinned --ckpt-sha256") from e
            print(f"weights_only load failed ({type(e).__name__}); full unpickle of the hash-verified file", flush=True)
            state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
            how = "full_unpickle_after_sha256_check"
        gs = {k[len("generator."):]: v for k, v in state.items() if k.startswith("generator.")}
        missing = gen.load_state_dict(gs, strict=True)
        self.net = gen.to(device).eval()
        self.info = {"concat_mask": self.concat, "load": how, "n_generator_tensors": len(gs),
                     "load_report": str(missing)}

    def forward(self, x, m, keys):
        torch = self.torch
        inp = torch.cat([x, m], 1) if self.concat else x
        pred = self.net(inp)
        return m * pred + (1 - m) * x


class SDInpaint(Base):
    def __init__(self, ckpt, device, model_size, frame_size, steps, revision=None):
        super().__init__(device, model_size, frame_size)
        torch = self.torch
        from diffusers import StableDiffusionInpaintPipeline
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            ckpt, revision=revision, torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
            safety_checker=None, requires_safety_checker=False).to(device)
        self.pipe.set_progress_bar_config(disable=True)
        self.steps = steps
        self.info = {"steps": steps, "prompt": "", "guidance_scale": 1.0, "hub_revision": revision}

    def forward(self, x, m, keys):
        torch = self.torch
        gens = [torch.Generator(self.device).manual_seed(int(k)) for k in keys]
        out = self.pipe(prompt=[""] * len(x), image=x, mask_image=m, height=self.ms, width=self.ms,
                        num_inference_steps=self.steps, guidance_scale=1.0, generator=gens, output_type="pt").images
        out = out.to(x.dtype)
        return m * out + (1 - m) * x


# Weights used in the paper, pinned by SHA-256 (PROTOCOL_model_fills.md, section 2). A checkpoint file is loaded only
# if its hash is the one passed with --ckpt-sha256, and that value must be one of these pins.
PINNED_SHA256 = {
    "big-lama.pt": "344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9",
    "lama-celeba-hq/lama-fourier/best.ckpt": "3060742ea7464288d7d5fa8e83c940ddbd359ba1640f79135759da9c8c81f281",
}


def verify_ckpt(ckpt, expected):
    """Refuse a checkpoint file whose SHA-256 is not the pinned, expected value. Returns the verified hash."""
    if expected not in PINNED_SHA256.values():
        raise SystemExit("--ckpt-sha256 is required for a checkpoint file and must be one of the pinned hashes")
    got = sha256(ckpt)
    if got != expected:
        raise SystemExit(f"checkpoint hash mismatch: {ckpt} has {got}, expected {expected}")
    return got


def run(job, model, backend, ckpt, out_dir, device, batch, model_size, repo=None, config=None, steps=20,
        every=1, ckpt_sha256=None, allow_unpickle=False, revision=None, input_mode="norm", sample=0):
    """`every`: keep every n-th target of the job (a stated subset for slow models); 1 = all.
    File checkpoints (torchscript, lama_repo) are hash-checked before loading; the diffusers model is pinned by its
    Hub `revision` (commit hash) when one is given."""
    import torch
    J = np.load(job)
    w, size = int(J["w"]), int(J["size"])
    if backend in ("torchscript", "lama_repo"):
        verify_ckpt(ckpt, ckpt_sha256)
    if backend == "torchscript":
        net = TorchScript(ckpt, device, model_size, size)
    elif backend == "lama_repo":
        net = LamaRepo(ckpt, device, model_size, size, repo, config, allow_unpickle)
    elif backend == "sd_inpaint":
        net = SDInpaint(ckpt, device, model_size, size, steps, revision)
    else:
        raise SystemExit(f"unknown backend {backend}")
    if input_mode not in INPUT_MODES:
        raise SystemExit(f"unknown input mode {input_mode}")
    net.input_mode = input_mode
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "infer_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {"sources": {}, "seconds": 0.0, "windows": 0}
    meta.update({"model": model, "backend": backend, "ckpt": os.path.basename(str(ckpt)), "ckpt_sha256": sha256(ckpt),
                 "device": device, "window_px": w, "zclip": ZCLIP, "model_size": model_size, "every": every,
                 "input_version": INPUT_VERSION, "input_mode": input_mode, "sample": sample,
                 "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
                 **net.info})
    for src in sorted(k[2:] for k in J.files if k.startswith("X_")):
        path = os.path.join(out_dir, f"{src}.npy")
        if os.path.exists(path):
            print(f"{src}: done before, skipped", flush=True)
            continue
        X, pos, tidx = J[f"X_{src}"][::every], J[f"pos_{src}"], J[f"tidx_{src}"][::every]
        fills = np.full((len(X), len(pos), w, w), np.nan, np.float32)
        if f"cells_{src}" in J.files:
            if every != 1:
                raise SystemExit("--every and a cell list are mutually exclusive")
            items = [(int(a), int(pk)) for a, pk in J[f"cells_{src}"]]
        else:
            items = [(a, pk) for a in range(len(X)) for pk in range(len(pos))]
        t0 = time.time()
        for b0 in range(0, len(items), batch):
            chunk = items[b0:b0 + batch]
            Z = np.stack([X[a].astype(np.float32) for a, _ in chunk])
            H = np.zeros_like(Z, bool)
            for k, (_, pk) in enumerate(chunk):
                y0, x0 = pos[pk]
                H[k, y0:y0 + w, x0:x0 + w] = True
            keys = [SEED + sample * 10_000_000 + int(tidx[a]) * 1000 + pk for a, pk in chunk]   # sample 0 = main run
            O = net(Z, H, keys)
            for k, (a, pk) in enumerate(chunk):
                y0, x0 = pos[pk]
                fills[a, pk] = O[k, y0:y0 + w, x0:x0 + w]
            if (b0 // batch) % 50 == 0 or b0 + batch >= len(items):
                el, n = time.time() - t0, b0 + len(chunk)
                print(f"{src}: {n}/{len(items)} windows, {el:.0f} s, {el / n * 1000:.1f} ms per window, "
                      f"ETA {el / n * (len(items) - n) / 60:.1f} min", flush=True)
        ran = np.zeros(fills.shape[:2], bool)
        ran[tuple(np.array(items).T)] = True
        if not np.isfinite(fills[ran]).all():
            raise RuntimeError(f"{src}: non-finite model output")
        np.save(path + ".tmp.npy", fills)
        os.replace(path + ".tmp.npy", path)
        meta["sources"][src] = {"n_targets": int(len(X)), "n_positions": int(len(pos)),
                                "target_index": [int(i) for i in tidx], "seconds": round(time.time() - t0, 1)}
        meta["seconds"] = round(meta["seconds"] + time.time() - t0, 1)
        meta["windows"] += len(items)
        json.dump(meta, open(meta_path, "w"), indent=1)
    meta["ms_per_window"] = round(meta["seconds"] / max(meta["windows"], 1) * 1000, 2)
    json.dump(meta, open(meta_path, "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "sources"}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", required=True, choices=("torchscript", "lama_repo", "sd_inpaint"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--model-size", type=int, required=True)
    ap.add_argument("--lama-repo")
    ap.add_argument("--config")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--ckpt-sha256", help="pinned SHA-256 of the checkpoint file (required for file checkpoints)")
    ap.add_argument("--allow-unpickle", action="store_true",
                    help="lama_repo: allow a full unpickle of the hash-verified checkpoint if weights-only fails")
    ap.add_argument("--revision", help="sd_inpaint: Hub commit hash of the model repository")
    ap.add_argument("--input-mode", default="norm", choices=INPUT_MODES,
                    help="how the frame reaches the model resolution (see the module text); default norm")
    ap.add_argument("--sample", type=int, default=0,
                    help="index of the sample for stochastic models (shifts every seed); 0 = the reported run")
    a = ap.parse_args()
    run(a.job, a.model, a.backend, a.ckpt, a.out, a.device, a.batch, a.model_size, a.lama_repo, a.config, a.steps,
        a.every, a.ckpt_sha256, a.allow_unpickle, a.revision, a.input_mode, a.sample)


if __name__ == "__main__":
    main()
