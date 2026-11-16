#!/usr/bin/env bash
# One-screen status of the VM runs.
cd ~/mf
echo "--- $(date -u +%H:%M:%S) UTC"; cat queue.log 2>/dev/null
pgrep -af 'fq.sh|model_fills_infer|brain_container|brain_smoke|brain_full' | cut -c1-140
for f in log_*.txt; do echo "== $f: $(tail -1 "$f" | cut -c1-170)"; done
ls out 2>/dev/null | tr '\n' ' '; echo; ls brain 2>/dev/null | tr '\n' ' '; echo
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; df -h / | tail -1 | awk '{print "disk free " $4}'
