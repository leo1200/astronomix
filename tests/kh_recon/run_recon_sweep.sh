#!/bin/bash
# Fixed reconstruction: band-limited control (prior) + early stop. Single vs
# soft-MS with a mu sweep, at one horizon. Usage:
#   GPU=4 TREC=80 KCTRL=6 N=64 bash run_recon_sweep.sh
set -u
DIR=/export/home/lstorcks/agent-home/astronomix/tests/kh_recon
GPU=${GPU:-4}; TREC=${TREC:-80}; KCTRL=${KCTRL:-6}; N=${N:-64}; STEPS=${STEPS:-200}
LOG=$DIR/run_recon_sweep_T${TREC%.*}_k${KCTRL}.log
echo "RECON SWEEP T=$TREC KCTRL=$KCTRL start $(date +%H:%M)" > "$LOG"
run () { # mode M mu
  echo "=== $1 M=$2 mu=$3 ===" >> "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 KH_N=$N KH_TREC=$TREC KH_STEPS=$STEPS \
    KH_LR=3e-3 KH_KCTRL=$KCTRL KH_NOISE=1e-2 KH_KCUT=$KCTRL \
    KH_MSMODE=$1 KH_M=$2 KH_MU=$3 \
    KH_OUT="$DIR/data/rec_${1}_M${2}_mu${3}_T${TREC%.*}_k${KCTRL}.npz" \
    python "$DIR/multiple_shooting.py" >> "$LOG" 2>&1
}
run single 1 0
for mu in 3 10 30 100 300; do run soft 4 $mu; done
echo "RECON SWEEP done $(date +%H:%M)" >> "$LOG"
