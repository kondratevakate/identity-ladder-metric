#!/usr/bin/env bash
# Idempotent restart of the model-fill runs after any VM start (spot preemptions included). Starts only what is
# neither running nor finished. Dead-man switch: the VM shuts itself down 10 h after the first resume, so an
# unattended machine cannot keep billing; the results are downloaded and the VM deleted by the operator before that.
cd ~/mf
[ -f deadman.set ] || { sudo shutdown -h +600 >/dev/null 2>&1; touch deadman.set; }
if [ ! -f out/.v2 ]; then                 # outputs of input version v1 (upsampling before masking) are kept aside
  [ -d out ] && [ ! -d out_v1_leak ] && mv out out_v1_leak
  mkdir -p out && touch out/.v2
fi
if ! grep -q FACES_V2_DONE queue.log 2>/dev/null && ! pgrep -f faces_v2_queue.s[h] >/dev/null; then
  nohup bash faces_v2_queue.sh >/dev/null 2>&1 &
fi
if [ ! -f brain/BRAIN_DONE ] && ! pgrep -f brain_full.s[h] >/dev/null; then
  nohup bash brain_full.sh 6 1.42 >/dev/null 2>&1 &
fi
sleep 2
F=$(grep -c FACES_V2_DONE queue.log 2>/dev/null); B=$([ -f brain/BRAIN_DONE ] && echo 1 || echo 0)
echo "STATE faces_done=${F:-0} brain_done=$B $(date -u +%H:%M:%S)"
tail -2 queue.log 2>/dev/null; tail -1 queue_brain.log 2>/dev/null; tail -1 log_brain_full.txt 2>/dev/null | cut -c1-110
if [ "${F:-0}" -ge 1 ] && [ "$B" = 1 ]; then echo ALL_DONE; fi
