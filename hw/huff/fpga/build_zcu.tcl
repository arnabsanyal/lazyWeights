# Vivado batch build of the ZCU102 design (zcu_top.sv: NE Huffman engines + the PS block design) ->
# build/fpga/zcu_<tag>/{zcu.bit, reports}. Arguments: NE MULT DIV tag (fabric clock = 300 MHz x MULT / DIV).
#   vivado -mode batch -source hw/huff/fpga/build_zcu.tcl -tclargs 4 4 12 f100
set H [file normalize [file dirname [info script]]]
lassign $argv NE MULT DIV TAG
set KHZ [expr {int(round(300000.0 * $MULT / $DIV))}]
set O [file normalize $H/../../../build/fpga/zcu_$TAG]
file delete -force $O/proj
file mkdir $O
create_project zcu $O/proj -part xczu9eg-ffvb1156-2-e -force
set_property board_part xilinx.com:zcu102:part0:3.4 [current_project]

# PS block design: M_AXI_HPM0_LPD only, through a SmartConnect to an external AXI4-Lite port
create_bd_design zps
create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e ps
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} [get_bd_cells ps]
set_property -dict [list CONFIG.PSU__USE__M_AXI_GP0 {0} CONFIG.PSU__USE__M_AXI_GP1 {0} CONFIG.PSU__USE__M_AXI_GP2 {1}] [get_bd_cells ps]
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect sc
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] [get_bd_cells sc]
create_bd_port -dir I -type clk -freq_hz [expr {$KHZ * 1000}] clk
create_bd_port -dir I -type rst resetn
set_property CONFIG.POLARITY ACTIVE_LOW [get_bd_ports resetn]
create_bd_intf_port -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI
set_property -dict [list CONFIG.PROTOCOL {AXI4LITE} CONFIG.ADDR_WIDTH {32} CONFIG.DATA_WIDTH {32}] [get_bd_intf_ports M_AXI]
set_property CONFIG.ASSOCIATED_BUSIF {M_AXI} [get_bd_ports clk]
set_property CONFIG.ASSOCIATED_RESET {resetn} [get_bd_ports clk]
connect_bd_intf_net [get_bd_intf_pins ps/M_AXI_HPM0_LPD] [get_bd_intf_pins sc/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins sc/M00_AXI] [get_bd_intf_ports M_AXI]
connect_bd_net [get_bd_ports clk] [get_bd_pins ps/maxihpm0_lpd_aclk] [get_bd_pins sc/aclk]
connect_bd_net [get_bd_ports resetn] [get_bd_pins sc/aresetn]
assign_bd_address -target_address_space /ps/Data [get_bd_addr_segs M_AXI/Reg] -range 16M -offset 0x80000000
puts "LPD width: [get_property CONFIG.PSU__MAXIGP2__DATA_WIDTH [get_bd_cells ps]]"
validate_bd_design
save_bd_design
generate_target all [get_files zps.bd]
add_files -norecurse [make_wrapper -files [get_files zps.bd] -top]

add_files -norecurse [list $H/../rtl/huff_lane.sv $H/../rtl/huff_arb.sv $H/../rtl/huff_engine.sv $H/zcu_core.sv $H/zcu_top.sv]
add_files -fileset constrs_1 -norecurse $H/zcu102.xdc
set_property top zcu_top [current_fileset]
set_property generic "NE=$NE MULT=$MULT DIV=$DIV CLK_KHZ=$KHZ" [current_fileset]
set_property STEPS.SYNTH_DESIGN.ARGS.FLATTEN_HIERARCHY rebuilt [get_runs synth_1]
set_property STRATEGY Performance_ExplorePostRoutePhysOpt [get_runs impl_1]
launch_runs synth_1 -jobs 16
wait_on_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 16
wait_on_run impl_1
open_run impl_1
report_utilization -file $O/util.rpt
report_utilization -hierarchical -hierarchical_depth 3 -file $O/util_hier.rpt
report_timing_summary -max_paths 10 -file $O/timing.rpt
report_power -file $O/power.rpt
file copy -force [glob $O/proj/zcu.runs/impl_1/*.bit] $O/zcu.bit
puts "BUILD_DONE $TAG NE=$NE clk=${KHZ}kHz"
