#!/bin/bash
# Post-synthesis area / timing of the Huffman engine blocks on ASAP7 (RVT, TT corner) without OpenROAD-flow-scripts (its
# checkout needs a newer Yosys than ecw-asic's 0.38): Yosys synth + ABC on the ASAP7 liberty, OpenSTA timing at
# CLK_PS (default 1000 = 1 GHz).
#   hw/huff/syn/synth.sh <top: huff_lane | huff_arb | huff_engine> <name> "<param value ...>"   -> build/syn/<name>/
set -eu
H=$(cd "$(dirname "$0")/.." && pwd); R=$(cd $H/../.. && pwd)
[ -n "${ASIC_ENV:-}" ] && export PATH=$ASIC_ENV/bin:$PATH   # yosys, OpenSTA, OpenROAD, Verilator, Icarus (README)
TOP=$1 N=$2 PAR=$3 CLK=${CLK_PS:-1000}
O=$R/build/syn/$N; mkdir -p $O
if [ "${PLATFORM:-asap7}" = nangate45 ]; then   # sanity check on a second library (liberty time unit ns)
  SEQ=${ORFS:-$HOME/work/OpenROAD-flow-scripts}/flow/platforms/nangate45/lib/NangateOpenCellLibrary_typical.lib; LIBS=$SEQ
  TU=0.001; HI="LOGIC1_X1 Z"; LO="LOGIC0_X1 Z"
else
L=${ORFS:-$HOME/work/OpenROAD-flow-scripts}/flow/platforms/asap7/lib/NLDM
LIBS=""; for f in AO_RVT_TT_nldm_211120 INVBUF_RVT_TT_nldm_220122 OA_RVT_TT_nldm_211120 SIMPLE_RVT_TT_nldm_211120; do
  [ -f $O/$f.lib ] || gunzip -c $L/asap7sc7p5t_$f.lib.gz > $O/$f.lib; LIBS="$LIBS $O/$f.lib"; done
SEQ=$L/asap7sc7p5t_SEQ_RVT_TT_nldm_220123.lib; LIBS="$LIBS $SEQ"
TU=1; HI="TIEHIx1_ASAP7_75t_R H"; LO="TIELOx1_ASAP7_75t_R L"
fi
CLKU=$(python3 -c "print($CLK * $TU)"); IOU=$(python3 -c "print(50 * $TU)")   # STA constraints in liberty units
CH=""; set -- $PAR; while [ $# -ge 2 ]; do CH="$CH -set $1 $2"; shift 2; done; CH="chparam$CH $TOP"; [ -n "$PAR" ] || CH=""
[ -f $O/merged.lib ] || { [ $TU = 1 ] && python3 $H/syn/merge_lib.py $O/merged.lib $LIBS || cp $SEQ $O/merged.lib; }
LIBARGS="-liberty $O/merged.lib"
echo "${PLATFORM:-asap7} $CLK" > $O/platform.txt
AD=${ABC_D:-$((CLK * 6 / 10))}   # ABC target below the clock: its delay model has no clk-to-q, setup or wires
if [ "${REPAIR:-0}" = 1 ]; then   # buffering / sizing left to OpenROAD repair_design / repair_timing (below)
  printf "strash\ndch\nmap -D $AD\ntopo\nstime -p\n" > $O/abc.script
else
  printf "strash\ndch\nmap -D $AD\nbuffer -p\nupsize -D $AD\ndnsize -D $AD\nstime -p\n" > $O/abc.script
fi
cat > $O/synth.ys <<YS
read_verilog -sv $H/rtl/huff_lane.sv $H/rtl/huff_arb.sv $H/rtl/huff_engine.sv $H/../rans/rtl/rans_lane.sv
$CH
hierarchy -check -top $TOP
synth -top $TOP -flatten -run begin:fine
techmap -autoproc -map $H/syn/kogge_stone.v -map +/techmap.v
synth -top $TOP -flatten -run fine:
dfflibmap -liberty $SEQ
abc -D $CLK -script $O/abc.script $LIBARGS
opt_clean -purge
hilomap -hicell $HI -locell $LO
setundef -zero
splitnets
opt_clean -purge
tee -o $O/area.txt stat $LIBARGS
write_verilog -noattr $O/netlist.v
YS
yosys -q -l $O/yosys.log $O/synth.ys > /dev/null
cat > $O/sta.tcl <<TCL
foreach f {$LIBS} { read_liberty \$f }
read_verilog $O/netlist.v
link_design $TOP
create_clock -name clk -period $CLKU [get_ports clk]
# module boundary: inputs from flops of the neighbouring logic, outputs into flops (50 ps each)
set_input_delay $IOU -clock clk [delete_from_list [all_inputs] [get_ports clk]]
set_output_delay $IOU -clock clk [all_outputs]
set_load 1.0 [all_outputs]
report_checks -path_delay max -digits 1 > $O/timing.txt
report_checks -path_delay max -digits 1 -from [all_registers -clock_pins] -to [all_registers -data_pins] > $O/timing_reg2reg.txt
report_wns > $O/wns.txt
report_tns >> $O/wns.txt
report_power > $O/power.txt
exit
TCL
sta -no_init -exit $O/sta.tcl > $O/sta.log 2>&1 || true
# REPAIR=1: OpenROAD repair_design (buffering / resizing for fanout, slew, capacitance) + repair_timing -setup on the
# unplaced netlist (no wire parasitics), as the floorplan stage of a P&R flow would; reports -> $O/repair_*.txt
if [ "${REPAIR:-0}" = 1 ]; then
  cat > $O/repair.tcl <<TCL
read_lef ${ORFS:-$HOME/work/OpenROAD-flow-scripts}/flow/platforms/asap7/lef/asap7_tech_1x_201209.lef
read_lef ${ORFS:-$HOME/work/OpenROAD-flow-scripts}/flow/platforms/asap7/lef/asap7sc7p5t_28_R_1x_220121a.lef
foreach f {$LIBS} { read_liberty \$f }
read_verilog $O/netlist.v
link_design $TOP
create_clock -name clk -period $CLK [get_ports clk]
set_input_delay 50 -clock clk [delete_from_list [all_inputs] [get_ports clk]]
set_output_delay 50 -clock clk [all_outputs]
set_load 1.0 [all_outputs]
repair_design
repair_timing -setup
report_design_area_metrics
set fa [open $O/repair_area.txt w]; puts \$fa "design area [sta::format_area [rsz::design_area] 3] um^2"; close \$fa
report_checks -path_delay max -digits 1 > $O/repair_timing.txt
report_checks -path_delay max -digits 1 -from [all_registers -clock_pins] -to [all_registers -data_pins] > $O/repair_timing_reg2reg.txt
report_wns > $O/repair_wns.txt
report_tns >> $O/repair_wns.txt
report_power > $O/repair_power.txt
write_verilog $O/netlist_repaired.v
exit
TCL
  openroad -no_init -exit $O/repair.tcl > $O/repair.log 2>&1 || true
  cat $O/repair_area.txt $O/repair_wns.txt; grep -m1 "slack" $O/repair_timing_reg2reg.txt || true
fi
grep -i "chip area" $O/area.txt; cat $O/wns.txt; grep -m1 "slack" $O/timing_reg2reg.txt; grep -E "^Total" $O/power.txt || true
