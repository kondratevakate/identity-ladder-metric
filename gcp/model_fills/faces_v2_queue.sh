#!/usr/bin/env bash
# Face runs, input version v2 (hidden pixels replaced before upsampling). One sequence, real exit codes in queue.log.
# The v1 outputs (upsampling before masking) are kept in ~/mf/out_v1_leak.
cd ~/mf
run() {
  echo "$(date -u +%H:%M:%S) START v2 $*" >> queue.log
  bash faces_vm.sh "$@" > "log_v2_$1${3:-}_e${4:-1}.txt" 2>&1
  local rc=$?
  echo "$(date -u +%H:%M:%S) END v2 $* rc=$rc" >> queue.log
}
run lama_celebahq faces_job_n0.npz
run lama_places faces_job_n0.npz
run lama_celebahq faces_job_w16_n0.npz _w16
run lama_places faces_job_w16_n0.npz _w16
for W in 20 5; do
  run lama_celebahq faces_job_w${W}_n100.npz _w${W}_n100
  run lama_places faces_job_w${W}_n100.npz _w${W}_n100
done
run sd15_inpaint faces_job_w16cells_n0.npz _w16 1
run sd15_inpaint faces_job_n0.npz "" 16
cat ~/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-inpainting/refs/main > out/sd15_snapshot.txt 2>/dev/null
echo "$(date -u +%H:%M:%S) FACES_V2_DONE" >> queue.log
