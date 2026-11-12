# Out-of-context synthesis + place + route of the 32-lane engine on the ZCU102 part (XCZU9EG-2) to find its clock
#   cd build/fpga/zcu_ooc; vivado -mode batch -source ../../../hw/huff/fpga/ooc_zu.tcl
set R [file normalize [file dirname [info script]]/../rtl]
set P xczu9eg-ffvb1156-2-e
create_project -in_memory -part $P -force
read_verilog -sv [list $R/huff_lane.sv $R/huff_arb.sv $R/huff_engine.sv]
synth_design -top huff_engine -part $P -mode out_of_context -flatten_hierarchy rebuilt \
  -generic LANES=32 -generic BUF=64 -generic FIFO=4 -generic PORT=8 -generic LOOKUP=1
create_clock -period 5.000 -name clk [get_ports clk]
report_utilization -file util_synth.rpt
opt_design
place_design
phys_opt_design
route_design
report_utilization -file util.rpt
report_timing_summary -max_paths 5 -file timing.rpt
