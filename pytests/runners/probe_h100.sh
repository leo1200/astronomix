#!/bin/bash
#SBATCH --job-name=astx_probe09
#SBATCH --account=hk-project-pai00101
#SBATCH --partition=dev_accelerated-h100
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Decisive check: does the FD Pallas backend work under multi-GPU sharding on
# jax 0.9 (single process)?  Runs the 1/2/4-GPU sharding probe in astrojax09.

REPO="/hkfs/home/project/hk-project-pai00101/hd_bn306/astronomix"
source "$REPO/pytests/runners/_env.sh"
cd "$REPO"

for G in 1 2 4; do
  echo "############ probe G=$G (jax $($CONDA_PREFIX/bin/python -c 'import jax;print(jax.__version__)')) ############"
  python pytests/_mgpu_pallas_probe.py --gpus $G --N 96 --steps 5 --block-shape 8,8,8 || echo "PROBE G=$G nonzero"
done
echo "DONE probe09"
