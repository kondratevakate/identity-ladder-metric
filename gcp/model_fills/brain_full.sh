#!/usr/bin/env bash
# Budgeted brain run (PROTOCOL_model_fills.md, section 5): targets in the fixed order of job/order.json, cubes
# 0,4,...,28 of the 32 random positions, both sides; no new target after BUDGET hours of run time in total.
# Resumable after a preemption: finished targets are skipped and the time already used is read from time_used.json
# (USED = hours spent before that file existed).
set -uo pipefail
cd ~/mf
BUDGET="${1:?budget in hours}"; USED="${2:-0}"
CK=$(cat brain/ckpt_dir.txt)
TARGETS=$(python3 -c "import json;print(','.join(map(str,json.load(open('brain/job/order.json'))['order'])))")
POS=$(python3 -c "import json;print(','.join(map(str,json.load(open('brain/job/order.json'))['positions'])))")
echo "$(date -u +%H:%M:%S) START brain budget ${BUDGET} h, used before ${USED} h" >> queue_brain.log
docker run --rm --gpus all -v "$HOME/mf":/work -v "$CK":/ckpt --entrypoint python3 \
  brainles/brats24_inpainting_ying_weng@sha256:c59daaf334ffccce63d76f65e8f95cedd794dbedd71a3eeefd21b437475fd253 /work/brain_container.py infer --job /work/brain/job --ckpt /ckpt \
  --out /work/brain/brats24_unet --targets "$TARGETS" --positions "$POS" --flips 01 --batch 1 \
  --budget-hours "$BUDGET" --used-hours "$USED" >> log_brain_full.txt 2>&1
rc=$?
echo "$(date -u +%H:%M:%S) END brain rc=$rc" >> queue_brain.log
[ $rc -eq 0 ] && touch brain/BRAIN_DONE
