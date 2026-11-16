#!/usr/bin/env bash
# Face model fills on the GPU VM (identity ladder). Inputs in ~/mf: the job files of `model_fills_face.py export`
# and model_fills_infer.py. Usage: bash faces_vm.sh MODEL [JOB.npz] [OUT_SUFFIX] [EVERY]
#   MODEL = lama_places | lama_celebahq | sd15_inpaint
#   JOB   = faces_job_n0.npz (10 px windows, f = 1/8; default), faces_job_w16_n0.npz (16 px panel windows) or
#           faces_job_w16cells_n0.npz (16 px, only the stored set-level cells)
#   EVERY = keep every n-th target (slow models only; 1 with a cell-list job)
set -euo pipefail
cd ~/mf
mkdir -p ckpt out
MODEL="$1"; JOB="${2:-faces_job_n0.npz}"; OUT="out/${MODEL}${3:-}"; EVERY="${4:-1}"
case "$MODEL" in
  lama_places)   # big-lama (Places), TorchScript export from the IOPaint model release, Apache-2.0
    [ -f ckpt/big-lama.pt ] || curl -sL -o ckpt/big-lama.pt \
      https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt
    sha256sum ckpt/big-lama.pt
    python3 model_fills_infer.py --job "$JOB" --model lama_places --backend torchscript \
      --ckpt ckpt/big-lama.pt --out "$OUT" --model-size 256 --batch 64 ;;
  lama_celebahq) # LaMa trained on CelebA-HQ 256 (lama-fourier), generator from the advimman/lama repository
    [ -d lama ] || git clone --depth 1 https://github.com/advimman/lama.git
    git -C lama rev-parse HEAD
    python3 -c "import omegaconf, kornia" 2>/dev/null || pip install -q omegaconf kornia
    sha256sum ckpt/lama-celeba-hq/lama-fourier/models/best.ckpt
    PYTHONPATH=lama python3 model_fills_infer.py --job "$JOB" --model lama_celebahq --backend lama_repo \
      --ckpt ckpt/lama-celeba-hq/lama-fourier/models/best.ckpt --lama-repo lama \
      --config ckpt/lama-celeba-hq/lama-fourier/config.yaml --out "$OUT" --model-size 256 --batch 64 ;;
  sd15_inpaint)  # Stable Diffusion 1.5 inpainting (CreativeML OpenRAIL-M), empty prompt, 512 px, 20 steps
    python3 -c "import diffusers, transformers, accelerate" 2>/dev/null || pip install -q diffusers transformers accelerate
    python3 model_fills_infer.py --job "$JOB" --model sd15_inpaint --backend sd_inpaint \
      --ckpt stable-diffusion-v1-5/stable-diffusion-inpainting --out "$OUT" --model-size 512 --batch 16 \
      --steps 20 --every "$EVERY" ;;
  *) echo "unknown model $MODEL"; exit 2 ;;
esac
