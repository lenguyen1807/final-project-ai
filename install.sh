#!/bin/bash
set -x #echo on

# Install nvidia
curl -fsSL https://get.gpuup.sh | bash

# Update system
sudo add-apt-repository ppa:quentiumyt/nvtop -y
sudo apt update -y
sudo apt upgrade -y
sudo apt-get install build-essential nodejs npm nvidia-gds btop nvtop unzip -y

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Reboot
sudo reboot
