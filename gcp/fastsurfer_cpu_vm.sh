#!/usr/bin/env bash
# FastSurfer remainder on a GCP CPU VM (brain protocol, Amendment 15, "Device, remainder", 9 Nov 2026).
# Same device class as the laptop run: native FastSurfer v2.5.3, CPU torch 2.7.1, the same flags and clipped inputs,
# the same resumable runner (local/fastsurfer_native_run.sh). Spot VM with stop on preemption: rerun skips DONE.
#   setup                         install FastSurfer v2.5.3 natively and check version, torch and device
#   pull                          copy the staged inputs from the bucket and check every sha256 against sha256.txt
#   run <list> <runners> <threads> run the list on <runners> parallel runners, sync labels and status to the bucket
# Only labels and FastSurfer logs go back to the bucket (cpu_out/); inputs stay on the VM disk, deleted with the VM.
set -euo pipefail
BUCKET=${BUCKET:?set BUCKET to gs://<bucket>/fastsurfer}
BASE=$HOME/fastsurfer-local; FS_HOME=$BASE/FastSurfer
export PRIVATE=$HOME/private
FS_TAG=v2.5.3; FS_VERSION=2.5.3+926af8d; TORCH=2.7.1+cpu
HERE=$(cd "$(dirname "$0")" && pwd)

check() {
  local v t
  v=$("$FS_HOME/run_fastsurfer.sh" --version 2>/dev/null | head -1)
  t=$("$FS_HOME/.venv/bin/python" -c "import torch; print(torch.__version__, torch.cuda.is_available())")
  [[ "$v" == "$FS_VERSION"* ]] || { echo "FastSurfer version '$v', want $FS_VERSION" >&2; exit 2; }
  [[ "$t" == "$TORCH False" ]] || { echo "torch '$t', want '$TORCH False'" >&2; exit 2; }
  echo "fastsurfer $v | torch $t | $(nproc) vCPU"
}

case ${1:?setup|pull|run} in
  setup)
    sudo apt-get update -qq && sudo apt-get install -y -qq git python3-venv python3-dev build-essential >/dev/null
    mkdir -p "$BASE"
    [ -d "$FS_HOME" ] || git clone -q --branch "$FS_TAG" --depth 1 https://github.com/Deep-MI/FastSurfer.git "$FS_HOME"
    python3 -m venv "$FS_HOME/.venv"
    "$FS_HOME/.venv/bin/pip" install -q --upgrade pip
    # CPU wheels first (torch 2.7.1+cpu satisfies the tag's torch==2.7.1 pin), then the tag's own pinned requirements
    "$FS_HOME/.venv/bin/pip" install -q "torch==${TORCH%+cpu}" torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
    "$FS_HOME/.venv/bin/pip" install -q -r "$FS_HOME/requirements.txt"
    check ;;
  pull)
    mkdir -p "$PRIVATE/gcp_stage"
    # only the listed files (repro + remainder), from the inputs staged on 6 Nov; sha256 against the laptop copies
    awk '{print $2}' "$HERE/sha256.txt" | while read -r f; do
      mkdir -p "$PRIVATE/gcp_stage/$(dirname "$f")"
      gcloud storage cp "$BUCKET/in/$f" "$PRIVATE/gcp_stage/$f" --quiet --no-user-output-enabled
    done
    (cd "$PRIVATE/gcp_stage" && sha256sum -c --quiet "$HERE/sha256.txt") || { echo "input sha256 mismatch" >&2; exit 3; }
    echo "inputs: $(find "$PRIVATE/gcp_stage" -name '*.nii.gz' | wc -l) files, sha256 ok" ;;
  run)
    LIST=${2:?list}; N=${3:?runners}; export FS_THREADS=${4:?threads}
    check
    rm -f "$BASE"/list_r*.txt
    awk -v n="$N" -v b="$BASE" 'NF { print > (b "/list_r" (NR - 1) % n ".txt") }' "$LIST"
    pids=()
    for f in "$BASE"/list_r*.txt; do
      k=$(basename "$f" .txt)
      bash "$HERE/fastsurfer_native_run.sh" "$f" "$k" > "$BASE/$k.out" 2>&1 & pids+=($!)
    done
    sync_out() {
      gcloud storage rsync -r "$PRIVATE/local_out" "$BUCKET/cpu_out" --quiet
      gcloud storage cp "$BASE/status.tsv" "$BASE/usage.log" "$BUCKET/cpu_meta/" --quiet
    }
    while [ -n "$(jobs -rp)" ]; do sleep 600; sync_out || echo "sync failed $(date -u +%FT%TZ)" >&2; done
    rc=0; for p in "${pids[@]}"; do wait "$p" || rc=$?; done
    sync_out
    echo "runners finished rc=$rc: $(grep -c -P '\tDONE\t' "$BASE/status.tsv") done, $(grep -c -P '\tFAILED\t' "$BASE/status.tsv") failed"
    exit $rc ;;
  *) echo "unknown mode $1" >&2; exit 1 ;;
esac
