#!/bin/bash
# Stage 4 reconstructions (serialized on one GPU). Usage:
#   GPU=4 TREC=80 N=128 bash run_stage4.sh
# Runs single / hard / soft at the chosen horizon, then a soft-MS segment sweep.
set -u
DIR=/export/home/lstorcks/agent-home/astronomix/tests/kh_recon
GPU=${GPU:-4}; TREC=${TREC:-80}; N=${N:-128}; STEPS=${STEPS:-300}
run () { # mode M mu
  echo "=== $1 M=$2 mu=$3 T=$TREC ==="
  CUDA_VISIBLE_DEVICES=$GPU KH_N=$N KH_TREC=$TREC KH_STEPS=$STEPS \
    KH_MSMODE=$1 KH_M=$2 KH_MU=$3 KH_KCUT=4 \
    KH_OUT="$DIR/data/ms_$1_M$2_T${TREC%.*}_s0.npz" \
    python "$DIR/multiple_shooting.py" >> "$DIR/run_stage4.log" 2>&1
}
echo "STAGE 4 start $(date +%H:%M)" > "$DIR/run_stage4.log"
run single 1 0          # single shooting
run hard   4 1000       # hard MS (stiff full-field continuity)
run soft   4 30         # soft MS (low-pass continuity)
run soft   2 30         # segment sweep
run soft   8 30
echo "STAGE 4 done $(date +%H:%M)" >> "$DIR/run_stage4.log"
