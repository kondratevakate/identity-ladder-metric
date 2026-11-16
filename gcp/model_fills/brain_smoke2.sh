#!/usr/bin/env bash
# Brain smoke, part 2 (after the official run of brain_smoke.sh): in-memory loop for 3 targets x 3 cubes x 2 frames in
# four array orientations, then the check against the official output. One case per forward pass.
set -uo pipefail
cd ~/mf
CASES=0:0:0,0:5:1,110:11:0
CK=$(cat brain/ckpt_dir.txt)
for FL in 01 none 012 0; do
  rm -rf brain/smoke_$FL
  docker run --rm --gpus all -v "$HOME/mf":/work -v "$CK":/ckpt --entrypoint python3 \
    brainles/brats24_inpainting_ying_weng@sha256:c59daaf334ffccce63d76f65e8f95cedd794dbedd71a3eeefd21b437475fd253 /work/brain_container.py infer --job /work/brain/job --ckpt /ckpt \
    --out /work/brain/smoke_$FL --targets 0,110,220 --positions 0,5,11 --flips $FL --batch 1 2>&1 | grep -v Warning | tail -4
  echo "flips $FL rc=${PIPESTATUS[0]}"
done
python3 brain_container.py check --job brain/job --fills brain/smoke_01 --official brain/official_out --cases $CASES
echo "check rc=$?"
echo SMOKE2_DONE
