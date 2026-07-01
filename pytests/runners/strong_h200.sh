#!/bin/bash
#SBATCH --job-name=astx_strong8
#SBATCH --account=hk-project-pai00101
#SBATCH --partition=accelerated-h200-8
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Strong scaling 1 GPU vs 8 H200 (single 8-GPU node, single process), pure
# HYDRO and MHD, FD-Pallas (the production solver), full N ladder up to N=1024
# -> grid 2N x N x N = 2048x1024x1024 ~ 2.1e9 cells at the top.
#
# Why 8xH200 single-node:
#   * Strong scaling needs the 1-GPU baseline, so the grid must fit on ONE GPU;
#     the harness also builds the IC on one device before resharding.  H200's
#     140 GB holds the 1024-scale grid (hydro ~120 GiB); H100's 94 GB OOMs
#     around N=768.  It is also the only single-node 8-GPU option, and MHD has
#     no sharded IC for a multi-host alternative.
#   * FD-Pallas only: at these sizes the FV/FD-JAX baselines OOM and would
#     waste the walltime budget; Pallas is the headline solver.
#   * run_strong_scaling checkpoints the NPZ after every N rung, so a wall-clock
#     timeout still leaves a complete prefix of the curve.
REPO="/hkfs/home/project/hk-project-pai00101/hd_bn306/astronomix"
source "$REPO/pytests/runners/_env.sh"
start_gpu_logger
trap stop_gpu_logger EXIT
cd "$REPO"

BLK=${BLK:-4,4,8}; STEPS=${STEPS:-10}
GPUS=${GPUS:-8}; TAG=${TAG:-h200}   # override GPUS=4 TAG=h200g4 for the 4-GPU sweep

for SETUP in hydro mhd; do
  echo "############ STRONG scaling 1-vs-${GPUS} (H200, ${SETUP}, FD-Pallas, up to N=1024) ############"
  python pytests/scaling_campaign.py --phase strong --setup "$SETUP" --gpus "$GPUS" \
    --steps "$STEPS" --tag "$TAG" --block-shape "$BLK" --solver pallas
done

stop_gpu_logger
echo "DONE strong_h200_1v${GPUS}"
