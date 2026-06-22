#!/bin/bash
#SBATCH --job-name=astx_weak4
#SBATCH --account=hk-project-pai00101
#SBATCH --partition=accelerated-h200
#SBATCH --time=01:30:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Phase 4 (weak scaling), HEADLINE: G = 16 across 4 nodes.
# With per-GPU (128 x 2048 x 2048) this is the 2048^3 cube; the defaults below
# are the SAFE per-GPU block -- override BY/BZ via env once Phase-1 bytes/cell
# confirm 2048^3 fits 16 GPUs.
REPO="/hkfs/home/project/hk-project-pai00101/hd_bn306/astronomix"
source "$REPO/pytests/runners/_env.sh"
start_gpu_logger
trap stop_gpu_logger EXIT
cd "$REPO"

BX=${BX:-128}; BY=${BY:-2048}; BZ=${BZ:-2048}
STEPS=${STEPS:-10}; DT=${DT:-0.4}; BLK=${BLK:-4,4,8}

echo "==== weak rung G=16 (per-GPU ${BX}x${BY}x${BZ}, global $((BX*16))x${BY}x${BZ}) ===="
srun --ntasks=16 --ntasks-per-node=4 --gpus-per-task=1 \
  python pytests/weak_scaling_hydro.py --bx $BX --by $BY --bz $BZ \
    --steps $STEPS --dt $DT --block-shape $BLK --tag h200

stop_gpu_logger
echo "DONE weak_h200_4node"
