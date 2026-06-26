#!/bin/bash
#SBATCH --job-name=astx_wh100_2n
#SBATCH --account=hk-project-pai00101
#SBATCH --partition=accelerated-h100
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Multi-node weak scaling on H100: G=8 across 2 nodes (one proc per GPU).
REPO="/hkfs/home/project/hk-project-pai00101/hd_bn306/astronomix"
source "$REPO/pytests/runners/_env.sh"
start_gpu_logger
trap stop_gpu_logger EXIT
cd "$REPO"

BX=${BX:-128}; BY=${BY:-2048}; BZ=${BZ:-2048}
STEPS=${STEPS:-10}; DT=${DT:-0.4}; BLK=${BLK:-4,4,8}

echo "==== weak rung G=8 on 2 H100 nodes (per-GPU ${BX}x${BY}x${BZ}, global $((BX*8))x${BY}x${BZ}) ===="
srun --ntasks=8 --ntasks-per-node=4 --gpu-bind=none \
  python pytests/weak_scaling_hydro.py --bx $BX --by $BY --bz $BZ \
    --steps $STEPS --dt $DT --block-shape $BLK --tag h100
rc=$?
stop_gpu_logger
echo "DONE weak_h100_2node rc=$rc"
exit $rc
