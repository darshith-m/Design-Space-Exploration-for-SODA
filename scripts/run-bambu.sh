#!/bin/bash

set -e
set -o pipefail

if [ -d output/$1 ] 
then
    rm -rf output/$1
    mkdir -p output/$1
else
    mkdir -p output/$1
fi

# PLATFORM=asap7
# DEVICE=asap7-BC
# DEVICE=asap7-TC
# DEVICE=asap7-WC

PLATFORM=nangate45
DEVICE=nangate45

cp output/05$1.ll output/$1/input.ll
cp output/main_kernel_test_$1.xml output/$1/main_kernel_test.xml

pushd output/$1;

docker run -u $(id -u):$(id -g) -v $(pwd):/working_dir --rm agostini01/soda \
opt -O2 -strip-debug input.ll \
  -S -o visualize.ll

docker run -u $(id -u):$(id -g) -v $(pwd):/working_dir --rm agostini01/soda \
bambu -v3 --print-dot \
  -lm --soft-float \
--compiler=I386_CLANG12  \
--device=${DEVICE} \
--clock-period=10 --no-iob \
--experimental-setup=BAMBU-BALANCED-MP \
--channels-number=2 \
--memory-allocation-policy=NO_BRAM \
--disable-function-proxy \
--generate-tb=main_kernel_test.xml \
--simulate --simulator=VERILATOR \
--max-sim-cycles=4294967295 \
--top-fname=main_kernel \
--time=3600 \
--num-accelerators=8 \
input.ll 2>&1 | tee bambu-log

popd

# Patch nangate
SCRIPTDIR=scripts
BAMBUDIR=output/$1
KERNELNAME=main_kernel
source ${SCRIPTDIR}/patch_openroad_synt.sh
