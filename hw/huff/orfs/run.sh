#!/bin/bash
# ORFS flow for one hw/huff/orfs/<variant>: hw/huff/orfs/run.sh <variant> [make target]  (logs/results under
# build/orfs/<variant>/ via WORK_HOME)
set -u
H=$(cd "$(dirname "$0")" && pwd); R=$(cd $H/../../.. && pwd)
[ -n "${ASIC_ENV:-}" ] && export PATH=$ASIC_ENV/bin:$PATH   # yosys, OpenSTA, OpenROAD, Verilator, Icarus (README)
export YOSYS_EXE=$(which yosys) OPENROAD_EXE=$(which openroad) OPENSTA_EXE=$(which sta)
export HUFF_RTL=$R/hw/huff/rtl HUFF_ORFS=$H
export WORK_HOME=$R/build/orfs/$1
mkdir -p $WORK_HOME
make -C ${ORFS:-$HOME/work/OpenROAD-flow-scripts}/flow DESIGN_CONFIG=$H/$1/config.mk ${2:-finish} > $WORK_HOME/flow.log 2>&1
echo "$1 rc=$?"
