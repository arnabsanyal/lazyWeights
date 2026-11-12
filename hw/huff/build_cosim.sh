#!/bin/bash
# Verilator co-simulation binary for one engine configuration:
#   hw/huff/build_cosim.sh <LOOKUP> <BUF> <FIFO> <PORT>   -> build/vl_<L>_<B>_<F>_<P>/cosim
set -e
H=$(cd "$(dirname "$0")" && pwd); R=$(cd $H/../.. && pwd)
[ -n "${ASIC_ENV:-}" ] && export PATH=$ASIC_ENV/bin:$PATH   # yosys, OpenSTA, OpenROAD, Verilator, Icarus (README)
L=$1 B=$2 F=$3 P=$4; D=$R/build/vl_${L}_${B}_${F}_${P}
verilator --cc --exe --build -j 8 -O3 --trace -Wno-fatal -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-DECLFILENAME \
  -GLOOKUP=$L -GBUF=$B -GFIFO=$F -GPORT=$P --top-module huff_engine -Mdir $D \
  -CFLAGS "-O2 -std=c++17 -I$H -DCFG_LOOKUP=$L -DCFG_BUF=$B -DCFG_FIFO=$F -DCFG_PORT=$P" \
  $H/rtl/huff_lane.sv $H/rtl/huff_arb.sv $H/rtl/huff_engine.sv $H/tb/cosim_main.cpp -o cosim > $D.log 2>&1 || { tail -30 $D.log; exit 1; }
echo $D/cosim
