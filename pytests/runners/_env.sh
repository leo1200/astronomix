#!/bin/bash
# Shared environment setup for astronomix GPU jobs on HoreKa.
# Source this from a runner after the #SBATCH block:  source "$(dirname "$0")/_env.sh"
# (No `set -e`: module/micromamba shell functions misbehave under it.)

export REPO="/hkfs/home/project/hk-project-pai00101/hd_bn306/astronomix"
cd "$REPO"

module purge
module load devel/cuda/12.4

source ~/.bashrc
# jax 0.9 env (astrojax09) is the campaign default: jax 0.10 breaks the sharded
# Pallas path. Override with ASTRO_ENV=astrocu12 if needed.
ASTRO_ENV="${ASTRO_ENV:-astrojax09}"
micromamba activate "$ASTRO_ENV"

# Make the env's bundled NVIDIA libs discoverable (jax[cuda12] pip wheels).
NV="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia"
for lib in cusparse cublas cufft cudnn nvjitlink cuda_nvrtc cuda_runtime; do
  if [ -d "$NV/$lib/lib" ]; then
    export LD_LIBRARY_PATH="$NV/$lib/lib:${LD_LIBRARY_PATH:-}"
  fi
done

# JAX runtime knobs: grow GPU memory on demand, leave a little headroom on the
# shared queues.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92

# JAX 0.10 defaults to the Shardy partitioner, which rejects this codebase's
# integer mesh-axis names and its with_sharding_constraint usage.  Fall back to
# the GSPMD partitioner (the model the multi-GPU/sharding code was written for).
export JAX_USE_SHARDY_PARTITIONER=false

# Record exactly which GPU(s) we got.
echo "=== job $SLURM_JOB_ID on $SLURM_JOB_NODELIST ==="
echo "partition: ${SLURM_JOB_PARTITION:-?}  nodes: ${SLURM_JOB_NUM_NODES:-1}  gpus(visible): ${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi -L || true

# Background GPU usage logger (per-job CSV); stopped via stop_gpu_logger.
GPU_LOG="${REPO}/pytests/scaling_results/gpu_usage_${SLURM_JOB_ID}.csv"
start_gpu_logger() {
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total \
    --format=csv -l 5 > "$GPU_LOG" 2>/dev/null &
  GPU_LOGGER_PID=$!
}
stop_gpu_logger() {
  [ -n "${GPU_LOGGER_PID:-}" ] && kill "$GPU_LOGGER_PID" 2>/dev/null || true
}
