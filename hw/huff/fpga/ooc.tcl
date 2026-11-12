# Out-of-context synthesis of the decode lane and the 32-lane engine on the Nexys 4 DDR part (XC7A100T-1)
set R [file normalize [file dirname [info script]]/../rtl]   ;# run from build/fpga/ooc
foreach {top name params} {
  huff_lane   lane   {BUF 64 FIFO 4 LOOKUP 1}
  huff_engine eng32  {LANES 32 BUF 64 FIFO 4 PORT 8 LOOKUP 1}
} {
  create_project -in_memory -part xc7a100tcsg324-1 -force
  read_verilog -sv [list $R/huff_lane.sv $R/huff_arb.sv $R/huff_engine.sv]
  set g {}; foreach {k v} $params { lappend g -generic $k=$v }
  synth_design -top $top -part xc7a100tcsg324-1 -mode out_of_context -flatten_hierarchy rebuilt {*}$g
  create_clock -period 10.000 -name clk [get_ports clk]
  report_utilization -file util_$name.rpt
  report_timing_summary -file timing_$name.rpt
  close_project
}
