#!/bin/bash
#SBATCH --job-name=astx_distdev
#SBATCH --account=hk-project-pai00101
#SBATCH --partition=dev_accelerated-h100
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Fast single-node validation of the multi-process GPU-binding fix on the dev
# queue (reliable, minutes).  Reproduces the exact failure we hit: 4 processes
# on ONE node doing an intra-node NCCL collective.  --gpu-bind=none exposes all
# 4 GPUs to every task; each process selects its own via local_device_ids.
# (Does not exercise inter-node network NCCL -- that needs the 2-node job.)
REPO="/hkfs/home/project/hk-project-pai00101/hd_bn306/astronomix"
source "$REPO/pytests/runners/_env.sh"
cd "$REPO"
srun --ntasks=4 --ntasks-per-node=4 --gpu-bind=none python pytests/_dist_sanity.py
rc=$?
echo "DONE dist_sanity_dev1node rc=$rc"
exit $rc
