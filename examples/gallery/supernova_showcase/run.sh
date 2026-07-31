#!/usr/bin/env bash
# ↓ ————————————————————————————————————————————————————————————— ↓
# Run wrapper: clean CUDA env, then exec the astx python.
# Usage: ./run.sh [ENVVAR=1 ...] script.py [args...]
#        (JAX_ENABLE_X64=1 must be set via env before python starts)
# ↑ ————————————————————————————————————————————————————————————— ↑
set -euo pipefail

strip_path() {
    # remove any /export/home/lstorcks/cuda entries from a colon list
    echo "$1" | tr ':' '\n' | grep -v '^/export/home/lstorcks/cuda' | paste -sd: -
}

export PATH="$(strip_path "${PATH:-}")"
export LD_LIBRARY_PATH="$(strip_path "${LD_LIBRARY_PATH:-}")"
unset CUDA_HOME CUDA_ROOT CUDA_PATH
# default: no preallocation (friendly on shared nodes); big runs should set
# XLA_PYTHON_CLIENT_PREALLOCATE=true (+ MEM_FRACTION) to avoid fragmentation
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
# import the working tree (or ASTRO_ROOT override, e.g. a bisect worktree),
# not the site-packages astronomix wheel
ASTRO_ROOT="${ASTRO_ROOT:-/export/home/lstorcks/jf1uids}"
# Fail loudly if that root has no astronomix. A missing path is silently dropped
# from PYTHONPATH and the STALE site-packages wheel gets imported instead, so an
# A/B against a worktree would compare the wrong code and still look healthy.
# The usual cause: a worktree under /tmp, which is node-local and therefore
# invisible to the pq compute node. Put A/B worktrees on /export.
if [ ! -f "$ASTRO_ROOT/astronomix/__init__.py" ]; then
    echo "run.sh: ASTRO_ROOT=$ASTRO_ROOT has no astronomix/ (on $(hostname))." >&2
    echo "run.sh: refusing to fall back to the site-packages wheel." >&2
    exit 2
fi
export PYTHONPATH="$ASTRO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

ASTX=/export/home/lstorcks/.local/share/mamba/envs/astx
exec "$ASTX/bin/python" "$@"
