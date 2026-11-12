# 1 GHz (ASAP7 time unit ps); I/O budget 20% of the period
current_design $::env(DESIGN_NAME)
set clk_period [expr {[info exists ::env(CLK_PERIOD_PS)] ? $::env(CLK_PERIOD_PS) : 1000}]
create_clock -name core_clock -period $clk_period [get_ports clk]
set_input_delay [expr $clk_period * 0.2] -clock core_clock [delete_from_list [all_inputs] [get_ports clk]]
set_output_delay [expr $clk_period * 0.2] -clock core_clock [all_outputs]
