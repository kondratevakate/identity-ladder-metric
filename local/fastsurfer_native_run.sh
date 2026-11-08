#!/usr/bin/env bash
# Resumable FastSurfer segmentation on Kate's own machine (WSL Ubuntu), Amendment 15 with the input clip.
# Native install of FastSurfer v2.5.3 (same tag as the pinned image deepmi/fastsurfer@sha256:8aa18bae...), CPU torch,
# flags as real_cases.run_segmenter: --seg_only --no_cereb --no_hypothal --no_biasfield, device cpu.
# Inputs: the clipped copies from `src/donor_ladder_brain_seg_gcp.py stage` (PRIVATE/gcp_stage/tNNN/<name>.nii.gz).
# Usage (inside WSL): FS_THREADS=<n> PRIVATE=/mnt/c/.../donor-ladder-brain-seg-20261106-v1 \
#                     fastsurfer_native_run.sh <list-file of "tNNN/name" lines> [run tag]
# Every volume ends as DONE or FAILED (exit code, log path) in status.tsv; DONE volumes are skipped on rerun.
# Only the label file and the FastSurfer log are copied back to PRIVATE/local_out/<sid>/; work dirs are removed.
set -u
LIST=$1; TAG=${2:-main}
THREADS=${FS_THREADS:?set FS_THREADS explicitly}
PRIVATE=${PRIVATE:?set PRIVATE to the private run folder}
FS_HOME=$HOME/fastsurfer-local/FastSurfer
BASE=$HOME/fastsurfer-local; WORK=$BASE/work; LOGS=$BASE/logs
IN=$PRIVATE/gcp_stage; BACK=$PRIVATE/local_out
mkdir -p "$WORK" "$LOGS" "$BACK"
touch "$BASE/status.tsv"
export FASTSURFER_HOME=$FS_HOME PATH="$FS_HOME/.venv/bin:$PATH"
while read -r item <&3; do   # list on fd 3: nothing in the loop may read it from stdin
  [ -z "$item" ] && continue
  if [ -e "$BASE/STOP" ]; then
    echo "$(date -u +%FT%TZ) [$TAG] stopped by $BASE/STOP before $item" | tee -a "$BASE/usage.log"; break
  fi
  sid=${item//\//_}
  if grep -q -P "^${sid}\tDONE\t" "$BASE/status.tsv"; then continue; fi
  if [ ! -s "$IN/$item.nii.gz" ]; then
    echo -e "$sid\tFAILED\tmissing_input\t-\t$(date -u +%FT%TZ)" >> "$BASE/status.tsv"; continue
  fi
  for attempt in 1 2; do   # D8: one retry, then counted as failed
    rm -rf "${WORK:?}/$sid"
    t0=$(date -u +%FT%TZ); s0=$(date +%s)
    log="$LOGS/${sid}_${attempt}.log"
    "$FS_HOME/run_fastsurfer.sh" --t1 "$IN/$item.nii.gz" --sid "$sid" --sd "$WORK" --seg_only \
      --no_cereb --no_hypothal --no_biasfield --device cpu --viewagg_device cpu --threads "$THREADS" \
      < /dev/null > "$log" 2>&1
    rc=$?; t1=$(date -u +%FT%TZ); secs=$(( $(date +%s) - s0 ))
    lab="$WORK/$sid/mri/aparc.DKTatlas+aseg.deep.mgz"
    # exit 0 is not enough (6 Nov: empty conform -> create_mask AssertionError with exit 0)
    [ $rc -eq 0 ] && [ ! -s "$lab" ] && rc=97
    [ $rc -eq 0 ] && [ ! -s "$WORK/$sid/mri/mask.mgz" ] && rc=98
    [ $rc -eq 0 ] && grep -q -E 'Traceback|AssertionError|RuntimeError' "$log" && rc=99
    echo -e "$t0\t$t1\tseg_cpu\t$sid\tattempt$attempt\trc=$rc\t${secs}s\t[$TAG]" >> "$BASE/usage.log"
    [ $rc -eq 0 ] && break
  done
  if [ $rc -eq 0 ]; then
    mkdir -p "$BACK/$sid/mri" && cp "$lab" "$BACK/$sid/mri/" && cp "$log" "$BACK/$sid/fastsurfer.log" \
      && echo -e "$sid\tDONE\t$secs\tattempt$attempt\t$t1" >> "$BASE/status.tsv" \
      || echo -e "$sid\tFAILED\tcopy_back\t$secs\t$t1" >> "$BASE/status.tsv"
    rm -rf "${WORK:?}/$sid"
  else
    echo -e "$sid\tFAILED\trc=$rc\t$LOGS/${sid}_2.log\t$t1" >> "$BASE/status.tsv"
  fi
  n_done=$(grep -c -P '\tDONE\t' "$BASE/status.tsv"); n_fail=$(grep -c -P '\tFAILED\t' "$BASE/status.tsv")
  echo "$(date -u +%FT%TZ) [$TAG] $sid rc=$rc ${secs}s | total done $n_done, failed $n_fail"
done 3< "$LIST"
echo "$(date -u +%FT%TZ) [$TAG] finished: $(grep -c -P '\tDONE\t' "$BASE/status.tsv") done, $(grep -c -P '\tFAILED\t' "$BASE/status.tsv") failed"
