#!/usr/bin/env bash
# Face model fills on the GPU VM (identity ladder). Inputs in ~/mf: the job files of `model_fills_face.py export`
# and model_fills_infer.py. Usage: bash faces_vm.sh MODEL [JOB.npz] [OUT_SUFFIX] [EVERY]
#   MODEL = lama_places | lama_celebahq | sd15_inpaint
#   JOB   = faces_job_n0.npz (10 px windows, f = 1/8; default), faces_job_w16_n0.npz (16 px panel windows),
#           faces_job_w16cells_n0.npz (16 px, only the stored set-level cells), faces_job_w{5,20}_n100.npz (RQ4 sizes)
#   EVERY = keep every n-th target (slow models only; 1 with a cell-list job)
set -euo pipefail
cd ~/mf
mkdir -p ckpt out
MODEL="$1"; JOB="${2:-faces_job_n0.npz}"; OUT="out/${MODEL}${3:-}"; EVERY="${4:-1}"
# Pins (PROTOCOL_model_fills.md, section 2): weights by SHA-256, code by commit. model_fills_infer.py refuses a
# checkpoint whose hash differs from the pin. SD_REVISION = Hub commit of the diffusion model, if set.
SHA_BIG_LAMA=344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9
SHA_LAMA_CELEBAHQ=3060742ea7464288d7d5fa8e83c940ddbd359ba1640f79135759da9c8c81f281
LAMA_COMMIT=786f5936b27fb3dacd2b1ad799e4de968ea697e7
SD_REVISION="${SD_REVISION:-}"
case "$MODEL" in
  lama_places)   # big-lama (Places), TorchScript export from the IOPaint model release, Apache-2.0
    [ -f ckpt/big-lama.pt ] || curl -sL -o ckpt/big-lama.pt \
      https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt
    python3 model_fills_infer.py --job "$JOB" --model lama_places --backend torchscript \
      --ckpt ckpt/big-lama.pt --ckpt-sha256 "$SHA_BIG_LAMA" --out "$OUT" --model-size 256 --batch 64 ;;
  lama_celebahq) # LaMa trained on CelebA-HQ 256 (lama-fourier), generator from the advimman/lama repository
    [ -d lama ] || git clone https://github.com/advimman/lama.git
    [ "$(git -C lama rev-parse HEAD)" = "$LAMA_COMMIT" ] || git -C lama checkout -q "$LAMA_COMMIT"
    [ "$(git -C lama rev-parse HEAD)" = "$LAMA_COMMIT" ] || { echo "lama repository is not at $LAMA_COMMIT"; exit 3; }
    python3 -c "import omegaconf, kornia" 2>/dev/null || pip install -q omegaconf kornia
    # The checkpoint holds a Lightning callback object, so weights-only loading is refused. The full unpickle is
    # allowed only with the explicit flag and after the SHA-256 check inside model_fills_infer.py.
    PYTHONPATH=lama python3 model_fills_infer.py --job "$JOB" --model lama_celebahq --backend lama_repo \
      --ckpt ckpt/lama-celeba-hq/lama-fourier/models/best.ckpt --ckpt-sha256 "$SHA_LAMA_CELEBAHQ" --allow-unpickle \
      --lama-repo lama --config ckpt/lama-celeba-hq/lama-fourier/config.yaml --out "$OUT" --model-size 256 --batch 64 ;;
  sd15_inpaint)  # Stable Diffusion 1.5 inpainting (CreativeML OpenRAIL-M), empty prompt, 512 px, 20 steps
    python3 -c "import diffusers, transformers, accelerate" 2>/dev/null || pip install -q diffusers transformers accelerate
    python3 model_fills_infer.py --job "$JOB" --model sd15_inpaint --backend sd_inpaint \
      --ckpt stable-diffusion-v1-5/stable-diffusion-inpainting ${SD_REVISION:+--revision "$SD_REVISION"} \
      --out "$OUT" --model-size 512 --batch 16 --steps 20 --every "$EVERY" ;;
  *) echo "unknown model $MODEL"; exit 2 ;;
esac
