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

# Dedicated hydro strong-scaling sweep (1 GPU vs 4 H100), all solvers, with the
# FULL N ladder [128,256,384,512,640] -> up to grid 1280x640x640 = 524M cells.
# Rationale:
#   * The earlier all-solver run timed out at the 1h dev limit AND saved nothing
#     (the NPZ was written only once, at the very end).
#   * run_strong_scaling now CHECKPOINTS the NPZ after every N rung (N-outer
#     loop), so even if the top rung doesn't finish in 1h, every completed rung
#     persists with all three solvers intact.
#   * The memory-heavy FV/FD-JAX baselines OOM at large N -> recorded as NaN
#     (fast, caught per cell); the lean FD-Pallas headline carries the curve to
#     N=640.  --steps 8 keeps per-run cost bounded.
REPO="/hkfs/home/project/hk-project-pai00101/hd_bn306/astronomix"
source "$REPO/pytests/runners/_env.sh"
start_gpu_logger
trap stop_gpu_logger EXIT
cd "$REPO"

echo "############ STRONG scaling 1-vs-4 (H100, hydro, all solvers, full ladder) ############"
python pytests/scaling_campaign.py --phase strong --setup hydro --gpus 4 --steps 8 \
  --tag h100 --block-shape 4,4,8

stop_gpu_logger
echo "DONE strong_hydro_h100"
