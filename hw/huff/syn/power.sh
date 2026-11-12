#!/bin/bash
# Lane power with real switching activity: gate-level replay (Icarus, ASAP7 cell models) of a co-sim stimulus on
# build/syn/<name>/netlist.v, VCD of the decode cycles -> OpenROAD read_power_activities + report_power at 1 GHz.
#   hw/huff/syn/power.sh <syn name> <stim .hex (from cosim --stim)> <tag>   -> build/power/<name>_<tag>.power.txt
set -eu
H=$(cd "$(dirname "$0")/.." && pwd); R=$(cd $H/../.. && pwd)
[ -n "${ASIC_ENV:-}" ] && export PATH=$ASIC_ENV/bin:$PATH   # yosys, OpenSTA, OpenROAD, Verilator, Icarus (README)
N=$1 S=$(readlink -f $2) T=$3 O=$R/build/syn/$1 P=$R/build/power
C=${ORFS:-$HOME/work/OpenROAD-flow-scripts}/flow/platforms/asap7/verilog/stdcell
NL=$(wc -l < $S); DS=2049   # 2048 table-load clocks + start, then decode
mkdir -p $P/inc_$T; cp $S.vh $P/inc_$T/stim_params.vh
if [ -z "${POWER_ONLY:-}" ]; then   # POWER_ONLY=1: reuse the VCD
iverilog -g2012 -o $P/${N}_$T.vvp -DNLINES=$NL -DDECODE_START=$DS -DSTIM_HEX="\"$S\"" -DVCD="\"$P/${N}_$T.vcd\"" \
  -I$P/inc_$T $H/syn/tb_lane_gl.v $O/netlist.v $C/asap7sc7p5t_*.v 2> $P/${N}_$T.iverilog.log
vvp -n $P/${N}_$T.vvp > $P/${N}_$T.vvp.log
fi
cat > $P/${N}_$T.power.tcl <<TCL
read_lef ${ORFS:-$HOME/work/OpenROAD-flow-scripts}/flow/platforms/asap7/lef/asap7_tech_1x_201209.lef
read_lef ${ORFS:-$HOME/work/OpenROAD-flow-scripts}/flow/platforms/asap7/lef/asap7sc7p5t_28_R_1x_220121a.lef
foreach f [glob $O/*.lib] { if {![string match *merged* \$f]} { read_liberty \$f } }
read_liberty ${ORFS:-$HOME/work/OpenROAD-flow-scripts}/flow/platforms/asap7/lib/NLDM/asap7sc7p5t_SEQ_RVT_TT_nldm_220123.lib
read_verilog $O/netlist.v
link_design huff_lane
create_clock -name clk -period 1000 [get_ports clk]
read_power_activities -scope tb/dut -vcd $P/${N}_$T.vcd
report_power
exit
TCL
openroad -no_init -no_splash -exit $P/${N}_$T.power.tcl > $P/${N}_$T.power.txt 2>&1
grep -E "^Total|^Sequential|^Combinational" $P/${N}_$T.power.txt
