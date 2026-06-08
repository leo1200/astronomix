#!/bin/bash
# Horizon scan of the mode-space (observable-subspace) reconstruction. Recovery
# should hold while the SVD frontier k_rec(T) > kmax, and fail once k_rec drops
# below the control band -- the operational test of the information frontier.
#   GPU=1 RE=2000 TLIST="20 40 80" bash run_modes_scan.sh
set -u
DIR=/export/home/lstorcks/agent-home/astronomix/tests/kh_recon
GPU=${GPU:-1}; RE=${RE:-2000}; N=${N:-64}; STEPS=${STEPS:-200}; KMAX=${KMAX:-6}
TLIST=${TLIST:-"10 20 40"}
LOG=$DIR/run_modes_scan_Re${RE%.*}_gpu${GPU}.log
echo "MODES SCAN gpu=$GPU Re=$RE TLIST=$TLIST start $(date +%H:%M)" > "$LOG"
for T in $TLIST; do
  echo "=== modes Re=$RE T=$T ===" >> "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 KH_N=$N KH_TREC=$T KH_RE=$RE KH_STEPS=$STEPS \
    KH_KMAX=$KMAX KH_NOISE=1e-2 KH_LR=3e-3 \
    KH_OUT="$DIR/data/modes_T${T}_Re${RE%.*}.npz" \
    python "$DIR/recover_modes.py" >> "$LOG" 2>&1
  echo "  done T=$T $(date +%H:%M)" >> "$LOG"
done
echo "MODES SCAN gpu=$GPU done $(date +%H:%M)" >> "$LOG"
