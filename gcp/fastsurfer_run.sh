#!/usr/bin/env bash
# Resumable FastSurfer segmentation for the brain donor-ladder audit (Amendment 15; Kate 6 Nov: --seg_only for all
# volumes, no surfaces). Runs on the shared GCP VM mriface-gpu; flags match real_cases.run_segmenter except the
# device (cuda instead of cpu).
# Inputs are the clipped copies from `src/donor_ladder_brain_seg_gcp.py stage` (Amendment 15, "Input clip").
# Usage: FS_IMG=<pinned digest> FS_THREADS=<n> fastsurfer_run.sh <list-file of "tNNN/name" lines>
# Every volume ends as DONE or FAILED (exit code, log path) in status.tsv; DONE volumes are skipped on rerun.
set -u
LIST=$1
BASE=$HOME/fastsurfer; IN=$BASE/in; OUT=$BASE/out; LOGS=$BASE/logs
IMG=${FS_IMG:?set FS_IMG to the pinned image digest}
THREADS=${FS_THREADS:?set FS_THREADS explicitly}
BUCKET=gs://fastsurfer-20261106-eg454009/fastsurfer
mkdir -p "$OUT" "$LOGS"
touch "$BASE/status.tsv"
while read -r item <&3; do   # list on fd 3: docker/gcloud in the loop read stdin
  [ -z "$item" ] && continue
  if [ -e "$BASE/STOP" ]; then   # the avatar-audit session can hand the GPU back to itself; rerun resumes
    echo "$(date -u +%FT%TZ) stopped by $BASE/STOP before $item" | tee -a "$BASE/usage.log"; break
  fi
  # Never share the GPU with the avatar audit: stop if anything beyond our own containers (~3.6 GB each) holds it.
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  mine=$(sudo docker ps -q --filter "ancestor=$IMG" | wc -l)
  if [ $(( used - mine * 4096 )) -gt 1024 ]; then
    echo "$(date -u +%FT%TZ) stopped before $item: GPU ${used} MiB used, ${mine} own containers" | tee -a "$BASE/usage.log"
    break
  fi
  sid=${item//\//_}
  if grep -q -P "^${sid}\tDONE\t" "$BASE/status.tsv"; then continue; fi
  if [ ! -s "$IN/$item.nii.gz" ]; then
    echo -e "$sid\tFAILED\tmissing_input\t-\t$(date -u +%FT%TZ)" >> "$BASE/status.tsv"; continue
  fi
  for attempt in 1 2; do   # D8: one retry, then counted as failed
    rm -rf "${OUT:?}/$sid"
    t0=$(date -u +%FT%TZ); s0=$(date +%s)
    sudo docker run --rm --gpus all --user "$(id -u):$(id -g)" \
      -v "$IN:/in:ro" -v "$OUT:/out" "$IMG" \
      --t1 "/in/$item.nii.gz" --sid "$sid" --sd /out --seg_only --no_cereb --no_hypothal --no_biasfield \
      --device cuda --viewagg_device cuda --threads "$THREADS" \
      < /dev/null > "$LOGS/${sid}_${attempt}.log" 2>&1   # docker must not eat the list on stdin
    rc=$?; t1=$(date -u +%FT%TZ); secs=$(( $(date +%s) - s0 ))
    lab="$OUT/$sid/mri/aparc.DKTatlas+aseg.deep.mgz"
    # FastSurfer can exit 0 after a Python traceback (seen 6 Nov: empty conform -> create_mask AssertionError),
    # so success also needs the label file, the brain mask and a log without a traceback.
    [ $rc -eq 0 ] && [ ! -s "$lab" ] && rc=97
    [ $rc -eq 0 ] && [ ! -s "$OUT/$sid/mri/mask.mgz" ] && rc=98
    [ $rc -eq 0 ] && grep -q -E 'Traceback|AssertionError|RuntimeError'"$LOGS/${sid}_${attempt}.log" && rc=99
    echo -e "$t0\t$t1\tseg\t$sid\tattempt$attempt\trc=$rc\t${secs}s" >> "$BASE/usage.log"
    [ $rc -eq 0 ] && break
  done
  if [ $rc -eq 0 ]; then
    if gcloud storage rsync -r "$OUT/$sid" "$BUCKET/out/$sid" --quiet >> "$LOGS/upload.log" 2>&1; then
      echo -e "$sid\tDONE\t$secs\tattempt$attempt\t$t1" >> "$BASE/status.tsv"
    else
      echo -e "$sid\tFAILED\tupload\t$secs\t$t1" >> "$BASE/status.tsv"
    fi
  else
    echo -e "$sid\tFAILED\trc=$rc\t$LOGS/${sid}_2.log\t$t1" >> "$BASE/status.tsv"
  fi
done 3< "$LIST"
echo "finished $(date -u +%FT%TZ): $(grep -c -P '\tDONE\t' "$BASE/status.tsv") done, $(grep -c -P '\tFAILED\t' "$BASE/status.tsv") failed lines"
