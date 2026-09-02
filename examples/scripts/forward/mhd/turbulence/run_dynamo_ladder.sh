#!/usr/bin/env bash
# Submit the dynamo convergence ladder to the GPU queue, one job per (code, N).
# Every job needs a single A100 -- the AthenaPK binary is an AMPERE80 build, so
# both codes have to run on A100s for the wall-clock numbers to be comparable.
#
#   bash run_dynamo_ladder.sh              # all three codes at N = 64, 128, 256
#   bash run_dynamo_ladder.sh 64 128       # only the resolutions given
#
# When the queue has drained:
#   python make_convergence_figures.py
set -euo pipefail

REPO=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
HERE="$REPO/examples/scripts/forward/mhd/turbulence"
PYTHON=/export/home/lstorcks/.local/share/mamba/envs/astx/bin/python
RESOLUTIONS=("${@:-64 128 256}")
read -r -a RESOLUTIONS <<< "${RESOLUTIONS[*]}"

for N in "${RESOLUTIONS[@]}"; do
    pq sub -t a100 -n 1 --name "astx_dyn_n$N" -- bash -c \
        "cd $REPO && ./examples/gallery/supernova_showcase/run.sh \
         $HERE/dynamo_convergence.py --n $N --save-slices"
    for SCHEME in plm ppm; do
        pq sub -t a100 -n 1 --name "apk_${SCHEME}_n$N" -- bash -c \
            "$PYTHON $HERE/athenapk_turb.py --n $N --scheme $SCHEME"
    done
done

pq stat
