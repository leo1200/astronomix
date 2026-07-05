#!/bin/bash
# ISM_N512 with the firewall (nan_safe 1) + vacuum_rest (auto-on) for a finite
# run to t/tc=5. cfl 1.5 (matches 256/448). Single 512^3 job (ICM done).
# MUST verify physics is preserved post-run (firewall can zero a severe event):
# M_turb(t/tc) should stay ~10 (NOT 0), E_K nonzero.
set -o pipefail
export PATH="/export/home/lstorcks/.local/share/mamba/envs/astx/bin:$PATH"
export PYTHONPATH="/export/home/lstorcks/agent-home/astronomix-refactor-port:$PYTHONPATH"
cd /export/home/lstorcks/agent-home/astronomix-refactor-port/tests/turbulence
echo "=== ISM_N512 (firewall) START $(date) ==="
eval $(autocvd -q); export CUDA_VISIBLE_DEVICES
echo "ISM_N512 GPU=$CUDA_VISIBLE_DEVICES"
python paper_turbulence.py --tag ISM_N512 --outdir data_fig14 \
  --eos iso --N 512 --mturb 10 --beta 0.1 --F0 3.5 --cfl 1.5 --tcross 5 --nsnap 6 \
  --stage_mode redist --rhomin 0.02 --vmaxcap 50 --protect 1 --nan_safe 1 \
  2>&1 | tee data_fig14/ISM_N512_fw.log
echo "=== ISM_N512 EXIT ${PIPESTATUS[0]} $(date) ==="
