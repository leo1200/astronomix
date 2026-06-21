#!/bin/bash
#SBATCH --job-name=astx_strong
#SBATCH --account=hk-project-pai00101
#SBATCH --partition=accelerated-h200
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Phase 3: strong scaling, 1 GPU vs 4 H200 (single node), all setups x all
# solver modes.  Uses the best pallas_block_shape from the Phase 2 sweep.

source "$(dirname "$0")/_env.sh"
start_gpu_logger
trap stop_gpu_logger EXIT
cd "$REPO"

python pytests/scaling_campaign.py --phase strong --setup all --gpus 4 --steps 10 --tag h200 \
  --block-shape ${BEST_BLOCK:-8,8,8}

stop_gpu_logger
echo "DONE strong_h200"
