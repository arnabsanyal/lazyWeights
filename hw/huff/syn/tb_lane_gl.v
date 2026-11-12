// Gate-level replay of one lane's inputs (dumped by hw/huff/tb/cosim_main.cpp --stim) on the synthesized lane netlist;
// VCD of the decode phase (after the table load) for OpenROAD read_power_activities.
`timescale 1ps/1ps
module tb;
`include "stim_params.vh"
  localparam integer NL = `NLINES;
  reg clk = 0, rst = 1;
  reg [163:0] mem [0:NL-1];
  reg [163:0] f;
  huff_lane dut (.clk(clk), .rst(rst), .start(f[160]), .ntiles(NTILES), .lut_we(f[24]), .lut_addr(f[22:12]),
    .lut_wdata(f[11:0]), .srt_we(f[44]), .srt_addr(f[43:36]), .srt_wdata(f[35:28]), .limit(LIMIT), .off(OFF),
    .desc_start(f[111:80]), .desc_end(f[79:48]), .rtile(), .want(), .grant(f[156]), .req_addr(), .req_meta(),
    .resp_valid(f[152]), .resp_data(f[143:112]), .resp_meta(f[151:144]), .out_valid(), .out_sym(), .out_idx(),
    .out_tile(), .busy(), .stall());
  integer i;
  always #500 clk = ~clk;
  initial begin
    $readmemh(`STIM_HEX, mem);
    f = 0;
    @(negedge clk); @(negedge clk); rst = 0;
    for (i = 0; i < NL; i = i + 1) begin
      f = mem[i];
      if (i == `DECODE_START) begin $dumpfile(`VCD); $dumpvars(0, dut); end
      @(negedge clk);
    end
    $finish;
  end
endmodule
