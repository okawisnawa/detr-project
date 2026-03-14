#!/bin/bash
# scripts/setup_env.sh - Robust Environment Setup for DETR Research
# Author: I Putu Oka Wisnawa, S.Kom., M.T.

set -e # Fail fast

ENV_NAME="detr-env"
YAML_FILE="environment.yml"

echo "=== [1/4] Checking Prerequisites ==="
if ! command -v micromamba &> /dev/null; then
    echo "Error: micromamba not found. Please install it in your distrobox first."
    exit 1
fi

# Inisialisasi shell agar perintah 'micromamba activate' tersedia
eval "$(micromamba shell hook --shell bash)"

echo "=== [2/4] Environment Synchronization ==="
# Cek apakah environment sudah ada
if micromamba env list | grep -q "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists. Updating dependencies..."
    # 'install' pada micromamba akan melakukan update jika file yaml berubah
    micromamba install -n $ENV_NAME -f $YAML_FILE -y
else
    echo "Creating new environment '$ENV_NAME'..."
    micromamba create -f $YAML_FILE -y
fi

echo "=== [3/4] Hardware & Driver Verification ==="
# Mengaktifkan env untuk verifikasi
micromamba activate $ENV_NAME

# Verifikasi dinamis (tidak hardcoded) agar lebih profesional
python -c "
import torch
import sys

print(f'Python Version: {sys.version.split()[0]}')
print(f'PyTorch Version: {torch.__version__}')

if torch.cuda.is_available():
    curr_gpu = torch.cuda.get_device_name(0)
    print(f'CUDA Available: Yes')
    print(f'GPU Detected: {curr_gpu}')
    if '4060' in curr_gpu:
        print('Optimal Hardware Match: RTX 4060 confirmed.')
else:
    print('ERROR: CUDA not available. Check your NVIDIA drivers/distrobox setup.')
    sys.exit(1)
"

echo "=== [4/4] Project Structure Check ==="
# Memastikan folder data tetap ada (via .gitkeep)
mkdir -p data/raw data/processed checkpoints
touch data/raw/.gitkeep data/processed/.gitkeep checkpoints/.gitkeep

echo "----------------------------------------------------------"
echo "Setup Gold Standard Complete!"
echo "To start working, run: micromamba activate $ENV_NAME"
echo "----------------------------------------------------------"
