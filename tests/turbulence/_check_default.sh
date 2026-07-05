#!/bin/bash
set -o pipefail
export PATH="/export/home/lstorcks/.local/share/mamba/envs/astx/bin:$PATH"
export PYTHONPATH="/export/home/lstorcks/agent-home/astronomix-refactor-port:$PYTHONPATH"
cd /export/home/lstorcks/agent-home/astronomix-refactor-port/tests/turbulence
run () { local tag="$1"; shift
  eval $(autocvd -q); export CUDA_VISIBLE_DEVICES
  python paper_turbulence.py --tag "$tag" --outdir data_repro --eos iso --N 128 \
    --beta 0.1 --F0 3.5 --tcross 4 --nsnap 8 --stage_mode redist --protect 1 "$@" \
    2>&1 | tee "data_repro/${tag}.log" | grep -E "vacuum_rest=|first_bad_snap"
}
mkdir -p data_repro
echo "--- check_std (mturb10 rho02, auto-default) ---"; run chk_std  --mturb 10 --rhomin 0.02  --cfl 1.5
echo "--- check_void (mturb20 rho005, auto-default) ---"; run chk_void --mturb 20 --rhomin 0.005 --cfl 1.5
echo "=== CHECK DONE ==="
