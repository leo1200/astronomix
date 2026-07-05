#!/bin/bash
# 384^3 diagnostic: does the 512^3-type crash reproduce at 384^3 (between
# stable-256 and unstable-512)? Fine per-snapshot diagnostics (nsnap 200) +
# progress bar pinpoint exactly when/how. vacuum_rest auto-on (supersonic).
set -o pipefail
export PATH="/export/home/lstorcks/.local/share/mamba/envs/astx/bin:$PATH"
export PYTHONPATH="/export/home/lstorcks/agent-home/astronomix-refactor-port:$PYTHONPATH"
cd /export/home/lstorcks/agent-home/astronomix-refactor-port/tests/turbulence
run () { local tag="$1"; shift
  echo "=== ${tag} START $(date +%H:%M:%S) ==="
  eval $(autocvd -q); export CUDA_VISIBLE_DEVICES
  echo "${tag} GPU=$CUDA_VISIBLE_DEVICES"
  python paper_turbulence.py --tag "$tag" --outdir data_repro --eos iso --N 384 \
    --beta 0.1 --F0 3.5 --cfl 1.5 --tcross 5 --nsnap 200 \
    --stage_mode redist --protect 1 --rhomin 0.02 --vmaxcap 50 --diag 1 "$@" \
    2>&1 | grep -E "diag ${tag}|FIRST NaN|no NaN|wall|eos=|Traceback|Error" | tail -40
  echo "=== ${tag} DONE $(date +%H:%M:%S) ==="
}
mkdir -p data_repro
run diag384_M10 --mturb 10
echo "=== DIAG384 DONE $(date +%H:%M:%S) ==="
