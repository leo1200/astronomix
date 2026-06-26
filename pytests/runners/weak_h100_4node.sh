#!/bin/bash
#SBATCH --job-name=astx_wh100_4n
#SBATCH --account=hk-project-pai00101
#SBATCH --partition=accelerated-h100
#SBATCH --time=01:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Multi-node weak scaling on H100: G=16 across 4 nodes (one proc per GPU).
REPO="/hkfs/home/project/hk-project-pai00101/hd_bn306/astronomix"
source "$REPO/pytests/runners/_env.sh"
start_gpu_logger
trap stop_gpu_logger EXIT
cd "$REPO"

BX=${BX:-128}; BY=${BY:-2048}; BZ=${BZ:-2048}
STEPS=${STEPS:-10}; DT=${DT:-0.4}; BLK=${BLK:-4,4,8}

echo "==== weak rung G=16 on 4 H100 nodes (per-GPU ${BX}x${BY}x${BZ}, global $((BX*16))x${BY}x${BZ}) ===="
srun --ntasks=16 --ntasks-per-node=4 --gpus-per-task=1 \
  python pytests/weak_scaling_hydro.py --bx $BX --by $BY --bz $BZ \
    --steps $STEPS --dt $DT --block-shape $BLK --tag h100
rc=$?
stop_gpu_logger
echo "DONE weak_h100_4node rc=$rc"
exit $rc
