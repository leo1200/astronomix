#!/usr/bin/env bash
# Submit the full resolution ladder for both codes to the GPU queue, one job per
# (code, N). Each job needs a single A100; they are independent and run
# concurrently as slots free up.
#
#   bash run_ladder.sh            # astronomix + AthenaK at N = 64, 128, 256
#   bash run_ladder.sh 64 128     # only the resolutions given
#
# When every job has left the queue, reduce and plot:
#   PYTHONPATH=$REPO python spectra.py --all && python make_figures.py
set -euo pipefail

REPO=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
HERE="$REPO/examples/scripts/forward/hydro/turbulence"
RESOLUTIONS=("${@:-64 128 256}")
read -r -a RESOLUTIONS <<< "${RESOLUTIONS[*]}"

for N in "${RESOLUTIONS[@]}"; do
    pq sub -t a100 -n 1 --name "astx_turb_n$N" -- bash -c \
        "cd $REPO && PYTHONPATH=$REPO python $HERE/driven_turbulence.py --n $N"
    # AthenaK needs its per-resolution dedt calibration first — its dedt is not
    # a resolution-invariant control (see check_athenak_driving.py) — so the two
    # steps are chained inside one job.
    pq sub -t a100 -n 1 --name "athk_turb_n$N" -- bash -c \
        "cd $REPO && python $HERE/calibrate_athenak.py --n $N --target 0.3175 && \
         python $HERE/athenak_turb.py --n $N --calibrated"
done

# Riemann-solver control: the more diffusive HLLE at one resolution, to show
# that the reconstruction order — not the flux function — drives the difference.
pq sub -t a100 -n 1 --name "athk_turb_n128_hlle" -- bash -c \
    "cd $REPO && python $HERE/athenak_turb.py --n 128 --calibrated --rsolver hlle"

pq stat
