#!/usr/bin/env bash
# Brain smoke on the VM: official entry point (brats package) against the in-memory loop on the same cases, then the
# in-memory loop for 3 targets x 3 cubes x 2 frames in four array orientations (orientation check).
set -uo pipefail
cd ~/mf
CASES=0:0:0,0:5:1,110:11:0
rm -rf brain/official_in brain/official_out
python3 brain_container.py dump --job brain/job --out brain/official_in --cases $CASES
python3 - <<'PY'
from brats import Inpainter
from brats.constants import InpaintingAlgorithms
Inpainter(algorithm=InpaintingAlgorithms.BraTS24_1, cuda_devices="0").infer_batch(
    data_folder="brain/official_in", output_folder="brain/official_out", log_file="brain/official.log")
PY
echo "official rc=$?"; ls brain/official_out | head
CK=$(dirname "$(find ~ /tmp -name '*.ckpt' -path '*14230865*' 2>/dev/null | head -1)")
[ -n "$CK" ] && [ "$CK" != "." ] || CK=$(dirname "$(find ~ /tmp -name '*.ckpt' 2>/dev/null | grep -v lama | head -1)")
echo "checkpoint dir: $CK"; find "$CK" -name '*.ckpt' | wc -l
echo "$CK" > brain/ckpt_dir.txt
for FL in 01 none 012 0; do
  docker run --rm --gpus all -v "$HOME/mf":/work -v "$CK":/ckpt --entrypoint python3 \
    brainles/brats24_inpainting_ying_weng:latest /work/brain_container.py infer --job /work/brain/job --ckpt /ckpt \
    --out /work/brain/smoke_$FL --targets 0,110,220 --positions 0,5,11 --flips $FL 2>&1 | grep -v Warning | tail -4
done
python3 brain_container.py check --job brain/job --fills brain/smoke_01 --official brain/official_out --cases $CASES
echo SMOKE_DONE
