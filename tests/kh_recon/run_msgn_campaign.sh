#!/bin/bash
# Large-statistics MS-vs-SS campaign at T=80 t_g (genuine mode-2), kx2-6.
# NRUN cold inits each for: single GN, constrained GN-MS M=4/8/16 (feasible interiors).
# Resumable (skips existing outputs). One sequential worker per GPU; round-robin assign.
#   NRUN=100 NGPU=8 bash run_msgn_campaign.sh
set -u
DIR=/export/home/lstorcks/agent-home/astronomix/tests/kh_recon
NRUN=${NRUN:-100}; NGPU=${NGPU:-8}; T=${T:-80}
mkdir -p "$DIR/data"
Q=$DIR/.campaign_queue; rm -f ${Q}_*.txt
# build job list (method:M:init) and round-robin onto GPUs
idx=0
for i in $(seq 0 $((NRUN-1))); do
  for spec in single:1 msgn:4 msgn:8 msgn:16; do
    g=$((idx % NGPU)); echo "$spec:$i" >> ${Q}_$g.txt; idx=$((idx+1))
  done
done
echo "CAMPAIGN: NRUN=$NRUN NGPU=$NGPU T=$T -> $idx jobs ($(date +%H:%M))"
run_worker () {
  local g=$1
  while IFS=: read method M init; do
    if [ "$method" = single ]; then
      out=$DIR/data/camp_single_T${T}_i${init}.npz
      [ -f "$out" ] && continue
      CUDA_VISIBLE_DEVICES=$g KH_INIT=$init KH_SEED=0 KH_INITSCALE=1e-2 KH_TREC=$T \
        KH_KMIN=2 KH_KMAX=6 KH_ITERS=50 KH_OUT="$out" python "$DIR/recover_gn.py" >/dev/null 2>&1
    else
      out=$DIR/data/camp_msgn_M${M}_T${T}_i${init}.npz
      [ -f "$out" ] && continue
      CUDA_VISIBLE_DEVICES=$g KH_INIT=$init KH_SEED=0 KH_INITSCALE=1e-2 KH_TREC=$T KH_M=$M \
        KH_INTERIOR=feasible KH_KMIN=2 KH_KMAX=6 KH_ITERS=40 KH_OUT="$out" python "$DIR/recover_msgn.py" >/dev/null 2>&1
    fi
  done < ${Q}_$g.txt
}
for g in $(seq 0 $((NGPU-1))); do run_worker $g & done
wait
echo "CAMPAIGN done $(date +%H:%M)"
