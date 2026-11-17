#!/usr/bin/env bash
# Idempotent restart of the model-fill runs after any VM start (spot preemptions included). Starts only what is
# neither running nor finished. Dead-man switch: the VM shuts itself down 9 h after the first resume of each boot, so
# an unattended machine cannot keep billing; the results are downloaded and the VM deleted by the operator before that.
cd ~/mf
BOOT=$(cut -d' ' -f1 /proc/sys/kernel/random/boot_id)
if [ "$(cat deadman.set 2>/dev/null)" != "$BOOT" ]; then
  sudo shutdown -h +540 >/dev/null 2>&1
  echo "$BOOT" > deadman.set
fi
if [ ! -f out/.v3 ]; then                 # outputs of older input versions are kept aside (v1: leak, v2: outline)
  if [ -f out/.v2 ] && [ ! -d out_v2_rim ]; then mv out out_v2_rim; fi
  if [ -d out ] && [ ! -f out/.v2 ] && [ ! -d out_v1_leak ]; then mv out out_v1_leak; fi
  mkdir -p out && touch out/.v3
fi
if ! grep -q FACES_V3_DONE queue.log 2>/dev/null && ! pgrep -f faces_v3_queue.s[h] >/dev/null; then
  nohup bash faces_v3_queue.sh >/dev/null 2>&1 &
fi
if [ ! -f brain/BRAIN_DONE ] && ! pgrep -f brain_full.s[h] >/dev/null; then
  nohup bash brain_full.sh 6 1.42 >/dev/null 2>&1 &
fi
sleep 2
F=$(grep -c FACES_V3_DONE queue.log 2>/dev/null); B=$([ -f brain/BRAIN_DONE ] && echo 1 || echo 0)
echo "STATE faces_done=${F:-0} brain_done=$B $(date -u +%H:%M:%S)"
tail -2 queue.log 2>/dev/null; tail -1 queue_brain.log 2>/dev/null; tail -1 log_brain_full.txt 2>/dev/null | cut -c1-110
if [ "${F:-0}" -ge 1 ] && [ "$B" = 1 ]; then echo ALL_DONE; fi
