#!/bin/bash
#SBATCH --job-name=astx_early
#SBATCH --account=hk-project-pai00101
#SBATCH --partition=dev_accelerated-h100
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Early real results on the reliably-fast dev H100 queue while the H200 jobs
# wait: pallas_block_shape sweep + single-GPU runtime/memory (all setups), 1 GPU.
REPO="/hkfs/home/project/hk-project-pai00101/hd_bn306/astronomix"
source "$REPO/pytests/runners/_env.sh"
start_gpu_logger
trap stop_gpu_logger EXIT
cd "$REPO"

echo "############ block sweep (H100) ############"
python pytests/scaling_campaign.py --phase block --gpus 1 --steps 20 --tag h100
echo "############ single-GPU sweep (H100, all setups) ############"
python pytests/scaling_campaign.py --phase single --setup all --gpus 1 --steps 10 --tag h100 --nmax 512
stop_gpu_logger
echo "DONE early_h100"
