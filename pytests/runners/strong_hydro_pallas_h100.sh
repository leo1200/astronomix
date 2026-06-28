#!/bin/bash
#SBATCH --job-name=astx_strpal
#SBATCH --account=hk-project-pai00101
#SBATCH --partition=dev_accelerated-h100
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Pallas-only hydro strong-scaling sweep (1 GPU vs 4 H100), FULL ladder
# [128..1024] -> grid 2N x N x N.  Pallas-only because:
#   * The all-solver run timed out on the FD-JAX baseline (107 s/run at N=512),
#     so it never reached N>=640.  FD-Pallas is the headline solver and is
#     ~10x leaner/faster, so it carries the curve to high N within the 1h dev cap.
#   * run_strong_scaling checkpoints the NPZ after every N rung, so every
#     completed rung persists even if the top rung times out.
#   * Single-GPU baseline OOMs around N~896 (94 GB / 60 B/cell ~ 1.5e9 cells);
#     those rungs record NaN speedup (caught per cell) and the sweep continues.
# NOTE: writes to the SAME hydro_h100_strong_scaling NPZ key but a separate
# pallas-tagged file so it does NOT clobber the committed 3-solver N<=512 curve.
REPO="/hkfs/home/project/hk-project-pai00101/hd_bn306/astronomix"
source "$REPO/pytests/runners/_env.sh"
start_gpu_logger
trap stop_gpu_logger EXIT
cd "$REPO"

echo "############ STRONG scaling 1-vs-4 (H100, hydro, PALLAS-only, full ladder) ############"
python pytests/scaling_campaign.py --phase strong --setup hydro --gpus 4 --steps 8 \
  --tag h100pallas --block-shape 4,4,8 --solver pallas

stop_gpu_logger
echo "DONE strong_hydro_pallas_h100"
