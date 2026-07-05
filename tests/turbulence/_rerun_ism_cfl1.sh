#!/bin/bash
# Re-run ISM_N512 at cfl 1.0 (the cfl 1.5 run went unstable: first_bad_snap=5).
# Keeps positivity stack intact per the run prompt (redist + vmaxcap 50 + protect).
set -o pipefail
export PATH="/export/home/lstorcks/.local/share/mamba/envs/astx/bin:$PATH"
export PYTHONPATH="/export/home/lstorcks/agent-home/astronomix-refactor-port:$PYTHONPATH"
cd /export/home/lstorcks/agent-home/astronomix-refactor-port/tests/turbulence

echo "=== ISM_N512 cfl1.0 START $(date) ==="
eval $(autocvd -q)
export CUDA_VISIBLE_DEVICES
echo "ISM_N512 (cfl1.0) using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python paper_turbulence.py \
  --tag ISM_N512 --outdir data_fig14 \
  --eos iso --N 512 --mturb 10 --beta 0.1 \
  --F0 3.5 --cfl 1.0 --tcross 5 --nsnap 6 \
  --stage_mode redist --rhomin 0.02 --vmaxcap 50 --protect 1 \
  2>&1 | tee data_fig14/ISM_N512_cfl1.log
echo "=== ISM_N512 cfl1.0 EXIT ${PIPESTATUS[0]} $(date) ==="
