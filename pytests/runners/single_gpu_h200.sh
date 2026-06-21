#!/bin/bash
#SBATCH --job-name=astx_single
#SBATCH --account=hk-project-pai00101
#SBATCH --partition=accelerated-h200
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Phase 2 (pallas_block_shape sweep) + Phase 1 (single-GPU runtime/memory for
# every setup x solver-mode) on one H200.  fp32, LSRK4, donate, fixed steps.

source "$(dirname "$0")/_env.sh"
start_gpu_logger
trap stop_gpu_logger EXIT
cd "$REPO"

echo "############ Phase 2: pallas_block_shape sweep (hydro, FD Pallas) ############"
python pytests/scaling_campaign.py --phase block --gpus 1 --steps 20 --tag h200

echo "############ Phase 1: single-GPU runtime + memory (all setups) ############"
python pytests/scaling_campaign.py --phase single --setup all --gpus 1 --steps 10 --tag h200 \
  --block-shape ${BEST_BLOCK:-8,8,8}

stop_gpu_logger
echo "DONE single_gpu_h200"
