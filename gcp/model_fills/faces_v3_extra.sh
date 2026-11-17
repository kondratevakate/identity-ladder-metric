#!/usr/bin/env bash
# Second sample of the diffusion model at 1/8 on the same people as the reported run (every 16th person), to
# measure how much its position depends on the seed (PROTOCOL_model_fills.md, amendment 3). Starts after the main
# v3 queue has finished; uses its own copy of the inference script so that nothing running is replaced.
cd ~/mf
while ! grep -q FACES_V3_DONE queue.log; do sleep 20; done
REFS=~/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-inpainting/refs/main
SD_REVISION="$(cat "$REFS")"
echo "$(date -u +%H:%M:%S) START v3 sd15_inpaint sample 1" >> queue.log
python3 model_fills_infer_s.py --job faces_job_n0.npz --model sd15_inpaint --backend sd_inpaint \
  --ckpt stable-diffusion-v1-5/stable-diffusion-inpainting --revision "$SD_REVISION" --out out/sd15_inpaint_s1 \
  --model-size 512 --batch 16 --steps 20 --every 16 --input-mode norm --sample 1 > log_v3_sd15_inpaint_s1.txt 2>&1
rc=$?
echo "$(date -u +%H:%M:%S) END v3 sd15_inpaint sample 1 rc=$rc" >> queue.log
echo "$(date -u +%H:%M:%S) EXTRA_DONE" >> queue.log
