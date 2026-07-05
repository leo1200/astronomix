#!/bin/bash
export PATH="/export/home/lstorcks/.local/share/mamba/envs/astx/bin:$PATH"
export PYTHONPATH="/export/home/lstorcks/agent-home/astronomix-refactor-port:$PYTHONPATH"
cd /export/home/lstorcks/agent-home/astronomix-refactor-port/tests/turbulence
eval $(autocvd -q); export CUDA_VISIBLE_DEVICES
python paper_turbulence.py --tag pp_dbg --outdir data_repro --eos iso --N 128 --mhd 0 \
  --mturb 40 --beta 0.1 --F0 3.5 --cfl 1.0 --tcross 4 --nsnap 8 \
  --stage_mode redist --protect 1 --rhomin 0.005 --vmaxcap 1000000000 \
  --vacuum_rest 0 --nan_safe 0 --pp_flux 1 2>&1 | grep -vE "libEGL|EGL device" | tail -30
