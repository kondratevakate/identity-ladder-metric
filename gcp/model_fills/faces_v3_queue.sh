#!/usr/bin/env bash
# Face runs, input version v3 (mask-aware upsampling; PROTOCOL_model_fills.md, amendment 2). One sequence, real exit
# codes in queue.log. Finished steps are skipped by model_fills_infer.py, so the queue can be restarted at any time.
# Older outputs are kept: ~/mf/out_v1_leak (v1) and ~/mf/out_v2_rim (v2).
# Order: what the main text needs first (both LaMa weights and the diffusion model at 1/8, then the 16 px panel
# window), then the sensitivity runs and the size curve. The first step had started under an earlier order of the
# same steps; this queue waits for it instead of starting it twice.
cd ~/mf
while pgrep -f "[m]odel_fills_infer.py" >/dev/null; do sleep 10; done
run() {
  echo "$(date -u +%H:%M:%S) START v3 $*" >> queue.log
  bash faces_vm.sh "$@" > "log_v3_$1${3:-}.txt" 2>&1
  local rc=$?
  echo "$(date -u +%H:%M:%S) END v3 $* rc=$rc" >> queue.log
}
run lama_celebahq faces_job_n0.npz
run lama_places faces_job_n0.npz
# The diffusion model is pinned to the Hub snapshot that is already on this disk.
REFS=~/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-inpainting/refs/main
if [ -f "$REFS" ]; then export SD_REVISION="$(cat "$REFS")"; cp "$REFS" out/sd15_snapshot.txt; fi
run sd15_inpaint faces_job_n0.npz "" 16
run lama_celebahq faces_job_w16_n0.npz _w16
run lama_places faces_job_w16_n0.npz _w16
run sd15_inpaint faces_job_w16cells_n0.npz _w16 1
for MODE in nearest native; do          # sensitivity to the input construction: every 8th target, 10 px windows
  run lama_celebahq faces_job_n0.npz "_$MODE" 8 "$MODE"
  run lama_places faces_job_n0.npz "_$MODE" 8 "$MODE"
done
run lama_celebahq faces_job_w20_n100.npz _w20_n100
run lama_places faces_job_w20_n100.npz _w20_n100
run lama_celebahq faces_job_w5_n100.npz _w5_n100
run lama_places faces_job_w5_n100.npz _w5_n100
echo "$(date -u +%H:%M:%S) FACES_V3_DONE" >> queue.log
