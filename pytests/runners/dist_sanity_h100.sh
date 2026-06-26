#!/bin/bash
#SBATCH --job-name=astx_distchk
#SBATCH --account=hk-project-pai00101
#SBATCH --partition=accelerated-h100
#SBATCH --time=00:15:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Validate jax.distributed rendezvous across 2 H100 nodes (8 procs) before a
# full multi-node run. Job and srun task counts MATCH (avoids the mismatch that
# made the earlier 2-proc smoke time out).
REPO="/hkfs/home/project/hk-project-pai00101/hd_bn306/astronomix"
source "$REPO/pytests/runners/_env.sh"
cd "$REPO"
srun --ntasks=8 --ntasks-per-node=4 --gpus-per-task=1 python pytests/_dist_sanity.py
rc=$?
echo "DONE dist_sanity rc=$rc"
exit $rc
