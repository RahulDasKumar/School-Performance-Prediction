#!/bin/bash
#SBATCH --job-name=outcome-concept-gen
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --partition=gpu             # Change to your cluster's GPU partition name
#SBATCH --gres=gpu:1                # 1 GPU (Qwen 1.5B fits on any modern GPU)
#SBATCH --mem=32G                   # 32GB RAM: model (~3GB bfloat16) + Neo4j + Python
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00             # 4 hours; adjust based on your dataset size
#SBATCH --ntasks=1




module load cuda
module load python/3.12.3
module load pytorch/2.3.0-cuda12.1





# ── Set HuggingFace cache to scratch/shared storage ──────────────────────────
# Avoids re-downloading 3GB model every run. Set to your HPC scratch directory:
# export HF_HOME=/scratch/${USER}/.cache/huggingface
# export TRANSFORMERS_CACHE=${HF_HOME}/hub

# ── Activate virtual environment ──────────────────────────────────────────────
# Update this path to where you created your venv on the cluster:
VENV_PATH="${HOME}/venvs/outcome-concept-gen"

if [ ! -d "${VENV_PATH}" ]; then
    echo "Creating virtual environment at ${VENV_PATH}..."
    python3 -m venv "${VENV_PATH}"
    source "${VENV_PATH}/bin/activate"

    # Install remaining dependencies
    pip install --quiet -r requirements.txt
else
    source "${VENV_PATH}/bin/activate"
fi

echo "Python: $(which python)"
echo "PyTorch CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

# ── Run from the script's directory so relative paths work ───────────────────
cd "$(dirname "$0")"

# ── Pre-download the model if not already cached ─────────────────────────────
python - <<EOF
from huggingface_hub import snapshot_download
import os

model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
cache_dir = "/users/rdas6/Models"
model_dir = os.path.join(cache_dir, "models--Qwen--Qwen2.5-Coder-7B-Instruct")

if os.path.isdir(model_dir):
    print(f"Model already cached at {model_dir}")
else:
    print(f"Downloading {model_name}...")
    snapshot_download(model_name, cache_dir=cache_dir)
    print("Download complete")
EOF

# ── Run the main script ───────────────────────────────────────────────────────
echo "Starting generate mapping at $(date)"
python categorize_counties.py
echo "Finished at $(date)"
