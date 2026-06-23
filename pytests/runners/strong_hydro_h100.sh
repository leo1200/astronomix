#!/bin/bash
#SBATCH --job-name=astx_strhyd
#SBATCH --account=hk-project-pai00101
#SBATCH --partition=dev_accelerated-h100
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Dedicated hydro strong-scaling sweep (1 GPU vs 4 H100) with the FULL N ladder
# [128,256,384,512,640] -> up to grid 1280x640x640 = 524M cells.  The earlier
# dev_multigpu run used --nmax 256, capping hydro to 2 points; this removes the
# cap so the lean FD-Pallas headline gets all 5.  The memory-heavy FV-JAX /
# FD-JAX single-GPU baselines OOM at large N -> recorded as NaN (run continues).
REPO="/hkfs/home/project/hk-project-pai00101/hd_bn306/astronomix"
source "$REPO/pytests/runners/_env.sh"
start_gpu_logger
trap stop_gpu_logger EXIT
cd "$REPO"

echo "############ STRONG scaling 1-vs-4 (H100, hydro, full ladder) ############"
python pytests/scaling_campaign.py --phase strong --setup hydro --gpus 4 --steps 8 \
  --tag h100 --block-shape 4,4,8

stop_gpu_logger
echo "DONE strong_hydro_h100"
