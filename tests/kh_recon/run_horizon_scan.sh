#!/bin/bash
# Horizon scan: locate the OPTIMIZATION frontier. At T=80 t_g recovery is
# hopeless for all methods (past the practical frontier). Scan SHORTER horizons
# with single shooting (cheap, M=1, band-limited control KCTRL=6) to find where
# recovery succeeds (lowk_err<<1, early-stop fires) vs fails (lowk_err~1+).
# One T per call so we can spread T values across free GPUs.
#   GPU=1 TLIST="5 15 30" bash run_horizon_scan.sh
set -u
DIR=/export/home/lstorcks/agent-home/astronomix/tests/kh_recon
GPU=${GPU:-1}; KCTRL=${KCTRL:-6}; N=${N:-64}; STEPS=${STEPS:-300}
TLIST=${TLIST:-"5 10 20 40"}
LOG=$DIR/run_horizon_scan_gpu${GPU}.log
echo "HORIZON SCAN gpu=$GPU TLIST=$TLIST start $(date +%H:%M)" > "$LOG"
for T in $TLIST; do
  echo "=== single T=$T ===" >> "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 KH_N=$N KH_TREC=$T KH_STEPS=$STEPS \
    KH_LR=3e-3 KH_KCTRL=$KCTRL KH_NOISE=1e-2 KH_KCUT=$KCTRL \
    KH_MSMODE=single KH_M=1 KH_MU=0 \
    KH_OUT="$DIR/data/hz_single_T${T}_k${KCTRL}.npz" \
    python "$DIR/multiple_shooting.py" >> "$LOG" 2>&1
  echo "  finished T=$T at $(date +%H:%M)" >> "$LOG"
done
echo "HORIZON SCAN gpu=$GPU done $(date +%H:%M)" >> "$LOG"
