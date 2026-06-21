#!/bin/bash
#SBATCH --job-name=astx_weak1
#SBATCH --account=hk-project-pai00101
#SBATCH --partition=accelerated-h200
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Phase 4 (weak scaling), low rungs on one node: G = 1, 2, 4.
# Per-GPU block (BX x BY x BZ) and step/dt/block tuned from Phase-1 bytes/cell.
source "$(dirname "$0")/_env.sh"
start_gpu_logger
trap stop_gpu_logger EXIT
cd "$REPO"

BX=${BX:-128}; BY=${BY:-1280}; BZ=${BZ:-1280}
STEPS=${STEPS:-10}; DT=${DT:-0.4}; BLK=${BLK:-8,8,8}

for G in 1 2 4; do
  echo "==== weak rung G=$G (per-GPU ${BX}x${BY}x${BZ}) ===="
  srun --ntasks=$G --ntasks-per-node=$G --gpus-per-task=1 \
    python pytests/weak_scaling_hydro.py --bx $BX --by $BY --bz $BZ \
      --steps $STEPS --dt $DT --block-shape $BLK --tag h200 \
    || echo "weak G=$G returned nonzero"
done

stop_gpu_logger
echo "DONE weak_h200_1node"
