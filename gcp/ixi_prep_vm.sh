#!/usr/bin/env bash
# IXI preparation on a GCP CPU VM (brain protocol, Amendment 23; Kate, 9 Nov 2026: "на машине").
#   setup                  Docker, the pinned FreeSurfer 8.0.0 image (SynthStrip), a venv with the laptop's pinned
#                          antspyx / nibabel / numpy versions; checks the image digest and the versions
#   check-in               sha256 of every uploaded scan against in/sha256.txt (written on the laptop)
#   run <threads> <workers> SynthStrip with <threads>, then prepare on <workers> x (<threads> / <workers>) threads
# Only prep/ (registered volumes, own masks, per-scan JSON) goes back; the VM is deleted at the end.
set -euo pipefail
RUN=$HOME/ixi-prep
IMAGE=freesurfer/freesurfer@sha256:f3e22b472afd9ca2b269d1b40c061ac891efaec070d09e387f4b024ea681e266
PY=$RUN/.venv/bin/python
trap 'echo "failed (rc=$?) at line $LINENO: $BASH_COMMAND" >&2' ERR   # no silent exit under set -e

check() {
  sudo docker image inspect "$IMAGE" --format '{{index .RepoDigests 0}}' | grep -q "${IMAGE#*@}" \
    || { echo "image digest differs" >&2; exit 2; }
  v=$($PY -c "from importlib.metadata import version as v; print(v('antspyx'), v('nibabel'), v('numpy'))")
  [[ "$v" == "0.6.3 5.4.2 1.26.4" ]] || { echo "versions '$v', want '0.6.3 5.4.2 1.26.4'" >&2; exit 2; }
  echo "image ok | antspyx nibabel numpy $v | $(nproc) vCPU"
}

case ${1:?setup|check-in|run} in
  setup)
    command -v docker >/dev/null || { sudo apt-get update -qq && sudo apt-get install -y -qq docker.io >/dev/null; }
    sudo apt-get install -y -qq python3-venv >/dev/null
    sudo docker pull -q "$IMAGE" >/dev/null
    [ -x "$PY" ] || python3 -m venv "$RUN/.venv"
    "$RUN/.venv/bin/pip" install -q antspyx==0.6.3 nibabel==5.4.2 numpy==1.26.4
    check ;;
  check-in)
    (cd "$RUN/in" && sha256sum -c --quiet sha256.txt) || { echo "input sha256 mismatch" >&2; exit 3; }
    echo "inputs: $(ls "$RUN"/in/*.nii.gz | wc -l) scans, sha256 ok" ;;
  run)
    T=${2:?threads}; W=${3:?workers}
    check
    # docker needs root; the masks are made world-readable for the unprivileged prepare stage
    sudo env DATA_ROOT=/nonexistent "$PY" "$RUN/ixi_prep.py" strip --run "$RUN" --threads "$T"
    sudo chown -R "$USER" "$RUN/strip"
    "$PY" "$RUN/ixi_prep.py" prepare --run "$RUN" --workers "$W" --threads $((T / W))
    echo "done: $(ls "$RUN"/prep/*.json | wc -l) json, $(grep -l '"error"' "$RUN"/prep/*.json | wc -l) errors" ;;
  *) echo "unknown mode $1" >&2; exit 1 ;;
esac
