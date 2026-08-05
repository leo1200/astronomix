#!/usr/bin/env bash
# Step-cost profile of the TI setup: isolate cooling / conduction / forcing.
cd "$(dirname "$(readlink -f "$0")")"
RUN=../supernova_showcase/run.sh   # the env wrapper lives with the Cas A showcase
T=0.35   # ~230 steps at dt ~1.5e-3
OUT=/export/data/lstorcks/supernova_showcase/prof
mkdir -p $OUT
run () { label="$1"; shift; s=$(date +%s.%N)
  $RUN casa_ti_phase.py --n 64 --t-end $T --nsnap 2 --ou --f0 10.7 --kf 0.589 --tcorr 0.5 \
      --save-state $OUT/$label.npz "$@" > $OUT/$label.log 2>&1
  e=$(date +%s.%N); n=$(grep -c "^progress" $OUT/$label.log)
  dt=$(grep "^progress" $OUT/$label.log | tail -1 | sed 's/.*dt = //')
  echo "$label: wall $(echo "$e-$s"|bc)s  progress_lines=$n  final_dt=$dt"; }
run baseline_implicit --conduction 1.0
run explicit_cooling  --conduction 1.0 --explicit-cooling
run no_cooling        --conduction 1.0 --no-cooling
run no_conduction     --explicit-cooling
