#!/bin/bash
# Builds the reference driver of the turbulence-driving tests from the C++ TurbGen.
#
# usage: bash build.sh <path to TurbGen.h> <build directory>
#
# TurbGen.h: github.com/chfeder/turbulence_generator, commit d9a7d742139ad873567ea4e53d47c88f9fa856ae. The header
# is copied into the build directory and TurbGen_instrumented.patch applied (it records every Gaussian deviate
# drawn). -ffp-contract=off: plain IEEE arithmetic without fused multiply-add, which is what a default x86-64
# build of TurbGen does; the amplitudes of the modes exactly at k_min and k_max are of round-off size and change
# with contraction.
set -e
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$2"
cp "$1" "$2/TurbGen_instrumented.h"
patch "$2/TurbGen_instrumented.h" "$here/TurbGen_instrumented.patch"
g++ -std=c++17 -O2 -ffp-contract=off -I "$2" -o "$2/turbgen_ref" "$here/turbgen_ref.cpp"
