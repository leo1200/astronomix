#!/bin/bash
#SBATCH --job-name=astx_block
#SBATCH --account=hk-project-pai00101
#SBATCH --partition=dev_accelerated-h100
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Additional single-GPU Pallas block-shape sweep (hydro FD-Pallas) at a
# larger resolution to confirm the (4,4,8) optimum is robust. Broadened
# candidate set. Distinct tag so it does not clobber the original N=256 sweep.
REPO="/hkfs/home/project/hk-project-pai00101/hd_bn306/astronomix"
source "$REPO/pytests/runners/_env.sh"
start_gpu_logger
trap stop_gpu_logger EXIT
cd "$REPO"

BLOCKN=${BLOCKN:-384}; STEPS=${STEPS:-20}; TAG=${TAG:-h100_n${BLOCKN}}
echo "==== single-GPU block-shape sweep (hydro FD-Pallas, N=${BLOCKN}) ===="
python pytests/scaling_campaign.py --phase block --gpus 1 \
  --block-n "$BLOCKN" --steps "$STEPS" --tag "$TAG"

stop_gpu_logger
echo "DONE block_sweep_dev rc=$?"
