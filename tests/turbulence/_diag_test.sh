#!/bin/bash
set -o pipefail
export PATH="/export/home/lstorcks/.local/share/mamba/envs/astx/bin:$PATH"
export PYTHONPATH="/export/home/lstorcks/agent-home/astronomix-refactor-port:$PYTHONPATH"
cd /export/home/lstorcks/agent-home/astronomix-refactor-port/tests/turbulence
eval $(autocvd -q); export CUDA_VISIBLE_DEVICES
echo "diag test GPU=$CUDA_VISIBLE_DEVICES"
python paper_turbulence.py --tag diagtest --outdir data_repro --eos iso --N 128 \
  --mturb 20 --beta 0.1 --F0 3.5 --cfl 1.5 --tcross 4 --nsnap 60 \
  --stage_mode redist --protect 1 --rhomin 0.005 --vmaxcap 1000000000 --vacuum_rest 0 \
  --diag 1 2>&1 | grep -E "diag diagtest|FIRST NaN|no NaN|wall|Traceback|Error" | tail -25
