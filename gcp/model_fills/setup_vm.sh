#!/usr/bin/env bash
# One-off VM setup for the model-fill runs: Docker with the NVIDIA runtime (BraTS inpainting algorithms are shipped as
# Docker images by the `brats` package) and the LaMa CelebA-HQ checkpoint.
set -uo pipefail
cd ~/mf
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sudo sh
fi
if ! dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update -qq && sudo apt-get install -y -qq nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
fi
sudo usermod -aG docker "$USER"
sudo docker run --rm --gpus all ubuntu:22.04 nvidia-smi --query-gpu=name --format=csv,noheader && echo DOCKER_GPU_OK
python3 --version
pip install -q brats nibabel 2>&1 | tail -2
python3 -c "import brats, importlib.metadata as m; print('brats', m.version('brats'))"
# LaMa CelebA-HQ (lama-fourier): Hugging Face mirror of the authors' LaMa_models.zip
mkdir -p ckpt/lama-celeba-hq/lama-fourier/models
B=https://huggingface.co/camenduru/big-lama/resolve/main/lama-celeba-hq/lama-fourier
[ -f ckpt/lama-celeba-hq/lama-fourier/models/best.ckpt ] || curl -sL -o ckpt/lama-celeba-hq/lama-fourier/models/best.ckpt $B/models/best.ckpt
[ -f ckpt/lama-celeba-hq/lama-fourier/config.yaml ] || curl -sL -o ckpt/lama-celeba-hq/lama-fourier/config.yaml $B/config.yaml
ls -l ckpt/lama-celeba-hq/lama-fourier/models/best.ckpt ckpt/lama-celeba-hq/lama-fourier/config.yaml
sha256sum ckpt/lama-celeba-hq/lama-fourier/models/best.ckpt
echo SETUP_DONE
