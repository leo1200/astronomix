#!/bin/bash
#SBATCH --job-name=astx_smoke
#SBATCH --account=hk-project-pai00101
#SBATCH --partition=dev_accelerated-h100
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Cheap validation on the H100 dev queue:
#  1) multi-GPU Pallas sharding probe (1->2->4 GPUs) -- settles the JAX-0.10 question
#  2) small single-GPU campaign for all setups (validates FV/Pallas/self-gravity)
#  3) 2-process srun of the weak driver (validates jax.distributed + sharded IC)

source "$(dirname "$0")/_env.sh"
start_gpu_logger
trap stop_gpu_logger EXIT

cd "$REPO"
echo "############ (1) multi-GPU Pallas probe ############"
for G in 1 2 4; do
  echo "---- probe G=$G ----"
  python pytests/_mgpu_pallas_probe.py --gpus $G --N 96 --steps 5 --block-shape 8,8,8 || echo "PROBE G=$G returned nonzero"
done

echo "############ (2) single-GPU campaign smoke (all setups, tiny N) ############"
# Override the N sweep via a tiny ad-hoc run by monkeypatching not available;
# instead run the campaign with the smallest grids it supports through --steps.
python pytests/scaling_campaign.py --phase single --setup all --gpus 1 --steps 3 --nmax 96 --tag h100smoke || echo "single smoke nonzero"

echo "############ (3) 2-process distributed weak-driver smoke ############"
srun --ntasks=2 --gpus-per-task=1 --ntasks-per-node=2 \
  python pytests/weak_scaling_hydro.py --bx 64 --by 128 --bz 128 --steps 3 \
    --block-shape 8,8,8 --tag h100smoke || echo "weak smoke nonzero"

stop_gpu_logger
echo "DONE smoke"
