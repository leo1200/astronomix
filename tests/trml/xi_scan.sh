#!/usr/bin/env bash
# Damkohler (xi) scan: run the two-phase TRML+tracer sim and the Fokker-Planck
# analysis at fixed N=64, M, chi for a range of xi = t_sh / t_coolmin. Each xi
# gets its own equilibrium, data, figures and a compact data/fp_result_xi*.npz
# that xi_scan_summary.py collates.
set -e
cd "$(dirname "$0")"
ROOT=/export/home/lstorcks/agent-home/astronomix-tracers
export PYTHONPATH=$ROOT

for XI in 3 10 30 100 300 1000; do
  SUF="_xi${XI}"
  echo "============================================================"
  echo "=== xi = ${XI}  (suffix ${SUF}) ==="
  echo "============================================================"
  TRML_PRESET=long64 TRML_XI=$XI TRML_SUFFIX=$SUF python -u trml_tracers.py
  TRML_SUFFIX=$SUF JAX_PLATFORMS=cpu python fokker_planck_test.py \
      | grep -iE "Check 0|L1  =|best dt|support:|J_downward" || true
done
echo "=== xi scan complete ==="
