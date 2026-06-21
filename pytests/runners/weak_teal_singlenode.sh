#!/bin/bash
#SBATCH --job-name=astx_weakteal
#SBATCH --account=hk-project-pai00101
#SBATCH --partition=accelerated-h200-8
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Weak scaling on ONE Teal node (up to 8 H200), SINGLE process per rung via
# autocvd -- no jax.distributed.  Ladder G = 1,2,4,8: global grid = (BX*G, BY, BZ).
REPO="/hkfs/home/project/hk-project-pai00101/hd_bn306/astronomix"
source "$REPO/pytests/runners/_env.sh"
start_gpu_logger
trap stop_gpu_logger EXIT
cd "$REPO"

BX=${BX:-128}; BY=${BY:-1024}; BZ=${BZ:-1024}
STEPS=${STEPS:-10}; DT=${DT:-0.4}; BLK=${BLK:-8,8,8}

for G in 1 2 4 8; do
  echo "==== weak rung G=$G (per-GPU ${BX}x${BY}x${BZ}, global $((BX*G))x${BY}x${BZ}) ===="
  python pytests/weak_scaling_hydro.py --gpus $G --bx $BX --by $BY --bz $BZ \
    --steps $STEPS --dt $DT --block-shape $BLK --tag h200teal \
    || echo "weak G=$G returned nonzero"
done

stop_gpu_logger
echo "DONE weak_teal_singlenode"
