#!/usr/bin/env bash
# End-to-end pipeline check at N=32 for one turnover time: both codes plus the
# shared spectral estimator. Cheap enough to run interactively; it exists to
# catch configuration errors before the real ladder is submitted.
set -euo pipefail

REPO=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
HERE="$REPO/examples/scripts/forward/hydro/turbulence"
export PYTHONPATH="$REPO"

echo "=== astronomix N=32 ==="
python "$HERE/driven_turbulence.py" --n 32 --nturn 1.0 --nsnap 3 --tag smoke32

echo "=== AthenaK N=32 ==="
python "$HERE/athenak_turb.py" --n 32 --nturn 1.0 --nsnap 3 --tag smoke32

echo "=== spectra ==="
python "$HERE/spectra.py" \
    --astronomix "$HERE/data/astronomix_smoke32.npz" \
    --athenak /export/data/lstorcks/turb_spectra/smoke32 \
    --tstart 0.0 --out spectra_smoke.npz

echo "=== figures ==="
python "$HERE/make_figures.py" --spectra spectra_smoke.npz
