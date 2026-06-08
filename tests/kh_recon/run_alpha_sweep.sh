#!/bin/bash
# Tikhonov L-curve: at a fixed short horizon, sweep the regularization alpha for
# single shooting (band-limited control). The reg term 0.5*alpha*<seed^2> must be
# comparable to the data misfit (~1e-8 at convergence) to bite -- with <seed^2>~1e-4
# that means alpha~1e-3..1e-1 (the earlier 1e-8..1e-5 grid was inert). Expect a U:
# too small -> null-space blowup (lowk_err>1), too large -> over-shrink toward 0
# (lowk_err->1), optimal alpha recovers (lowk_err<<1). lr=1e-3 to avoid overshoot.
#   GPU=5 TREC=10 ALIST="3e-4 1e-3 3e-3 1e-2 3e-2 1e-1" bash run_alpha_sweep.sh
set -u
DIR=/export/home/lstorcks/agent-home/astronomix/tests/kh_recon
GPU=${GPU:-1}; TREC=${TREC:-10}; KCTRL=${KCTRL:-6}; N=${N:-64}; STEPS=${STEPS:-400}
LR=${LR:-1e-3}; ALIST=${ALIST:-"3e-4 1e-3 3e-3 1e-2 3e-2 1e-1"}
LOG=$DIR/run_alpha_sweep_T${TREC}_gpu${GPU}.log
echo "ALPHA SWEEP gpu=$GPU T=$TREC lr=$LR ALIST=$ALIST start $(date +%H:%M)" > "$LOG"
for A in $ALIST; do
  echo "=== single T=$TREC alpha=$A ===" >> "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 KH_N=$N KH_TREC=$TREC KH_STEPS=$STEPS \
    KH_LR=$LR KH_KCTRL=$KCTRL KH_NOISE=1e-2 KH_KCUT=$KCTRL KH_ALPHA=$A \
    KH_MSMODE=single KH_M=1 KH_MU=0 \
    KH_OUT="$DIR/data/alpha_T${TREC}_a${A}.npz" \
    python "$DIR/multiple_shooting.py" >> "$LOG" 2>&1
  echo "  done alpha=$A $(date +%H:%M)" >> "$LOG"
done
echo "ALPHA SWEEP gpu=$GPU done $(date +%H:%M)" >> "$LOG"
