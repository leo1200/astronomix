#!/bin/bash
#SBATCH --job-name=astx_mg100
#SBATCH --account=hk-project-pai00101
#SBATCH --partition=dev_accelerated-h100
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Reliable multi-GPU results on the dev H100 queue (4 GPU): weak scaling
# G=1,2,4 (single process; saves each rung immediately) + strong scaling
# 1-vs-4 for all setups (capped to fit the 1h dev limit).
REPO="/hkfs/home/project/hk-project-pai00101/hd_bn306/astronomix"
source "$REPO/pytests/runners/_env.sh"
start_gpu_logger
trap stop_gpu_logger EXIT
cd "$REPO"

echo "############ WEAK scaling G=1,2,4 (H100, single process) ############"
for G in 1 2 4; do
  echo "==== weak G=$G ===="
  python pytests/weak_scaling_hydro.py --gpus $G --bx 128 --by 2048 --bz 2048 \
    --steps 10 --dt 0.4 --block-shape 4,4,8 --tag h100dev || echo "weak G=$G nonzero"
done

echo "############ STRONG scaling 1-vs-4 (H100, all setups, capped) ############"
python pytests/scaling_campaign.py --phase strong --setup all --gpus 4 --steps 8 \
  --tag h100 --block-shape 4,4,8 --nmax 256

stop_gpu_logger
echo "DONE dev_multigpu_h100"
