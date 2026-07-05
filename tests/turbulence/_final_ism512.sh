#!/bin/bash
# Final ISM_N512 with the validated fix: vacuum-rest (auto-on for supersonic).
# cfl 1.5 to match the 256/448 columns (vacuum-rest, not a lower cfl, is the
# real stabiliser). ICM_N512 is already done+good, so ISM only. One 512^3 job.
set -o pipefail
export PATH="/export/home/lstorcks/.local/share/mamba/envs/astx/bin:$PATH"
export PYTHONPATH="/export/home/lstorcks/agent-home/astronomix-refactor-port:$PYTHONPATH"
cd /export/home/lstorcks/agent-home/astronomix-refactor-port/tests/turbulence
echo "=== ISM_N512 (vacuum-rest) START $(date) ==="
eval $(autocvd -q); export CUDA_VISIBLE_DEVICES
echo "ISM_N512 GPU=$CUDA_VISIBLE_DEVICES"
python paper_turbulence.py --tag ISM_N512 --outdir data_fig14 \
  --eos iso --N 512 --mturb 10 --beta 0.1 --F0 3.5 --cfl 1.5 --tcross 5 --nsnap 6 \
  --stage_mode redist --rhomin 0.02 --vmaxcap 50 --protect 1 \
  2>&1 | tee data_fig14/ISM_N512_vacrest.log
echo "=== ISM_N512 EXIT ${PIPESTATUS[0]} $(date) ==="
