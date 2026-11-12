# Vivado batch build of the Nexys 4 DDR bring-up design -> build/fpga/nexys/{huff_nexys.bit, reports}
#   vivado -mode batch -source hw/huff/fpga/build.tcl
set H [file normalize [file dirname [info script]]]
set O [file normalize $H/../../../build/fpga/nexys]
file mkdir $O
read_verilog -sv [list $H/../rtl/huff_lane.sv $H/../rtl/huff_arb.sv $H/../rtl/huff_engine.sv $H/uart.sv $H/nexys_top.sv]
read_xdc $H/nexys4ddr.xdc
synth_design -top nexys_top -part xc7a100tcsg324-1
report_utilization -file $O/util_synth.rpt
opt_design
place_design
phys_opt_design
route_design
report_utilization -file $O/util.rpt
report_timing_summary -file $O/timing.rpt
write_bitstream -force $O/huff_nexys.bit
