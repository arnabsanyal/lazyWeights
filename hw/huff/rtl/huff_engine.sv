// A4: Huffman tile engine = 32 lanes (huff_lane.sv) + the memory port arbiter (huff_arb.sv): up to PORT word grants
// per cycle, round-robin over the lanes that want a word (pointer advances by one each cycle with a grant), as
// hw/huff/huff_engine.h decode_mat<LOOKUP, BUF, FIFO, PORT, LAT, 1>. Memory and the descriptor fetch (per-lane
// stream bounds from lane_start) are outside: the test bench / SoC returns each granted word LAT cycles later.
// Per-lane / per-port buses are packed ([index][bits]) so that Yosys reads the design.
module huff_engine #(
  parameter int LANES = 32,
  parameter int BUF = 64,
  parameter int FIFO = 4,
  parameter int PORT = 8,
  parameter int LOOKUP = 1,
  parameter int LMAX = 11
) (
  input  logic                     clk,
  input  logic                     rst,
  input  logic                     start,
  input  logic [15:0]              ntiles,
  input  logic                     lut_we,
  input  logic [10:0]              lut_addr,
  input  logic [11:0]              lut_wdata,
  input  logic                     srt_we,
  input  logic [7:0]               srt_addr,
  input  logic [7:0]               srt_wdata,
  input  logic                     cmp_we,     // comparator table registers: limit / first / base of length cmp_l
  input  logic [3:0]               cmp_l,
  input  logic [11:0]              cmp_limit,
  input  logic [10:0]              cmp_first,
  input  logic [8:0]               cmp_base,
  input  logic [LANES-1:0][31:0]   desc_start,
  input  logic [LANES-1:0][31:0]   desc_end,
  output logic [LANES-1:0][15:0]   rtile,
  output logic [PORT-1:0]          req_valid,
  output logic [PORT-1:0][$clog2(LANES)-1:0] req_lane,
  output logic [PORT-1:0][31:0]    req_addr,
  output logic [PORT-1:0][7:0]     req_meta,
  input  logic [LANES-1:0]         resp_valid,
  input  logic [LANES-1:0][31:0]   resp_data,
  input  logic [LANES-1:0][7:0]    resp_meta,
  output logic [LANES-1:0]         out_valid,
  output logic [LANES-1:0][7:0]    out_sym,
  output logic [LANES-1:0][8:0]    out_idx,
  output logic [LANES-1:0][15:0]   out_tile,
  output logic                     busy,
  output logic [5:0]               n_stall
);
  logic [LMAX-1:1][11:0] limit;
  logic [LMAX:1][7:0]    off;
  always_ff @(posedge clk)
    if (cmp_we) begin
      if (cmp_l < 4'(LMAX)) limit[cmp_l] <= cmp_limit;
      off[cmp_l] <= cmp_base[7:0] - cmp_first[7:0];
    end

  logic [LANES-1:0]       want, grant, lbusy, lstall;
  logic [LANES-1:0][31:0] laddr;
  logic [LANES-1:0][7:0]  lmeta;

  for (genvar l = 0; l < LANES; l++) begin : g_lane
    huff_lane #(.BUF(BUF), .FIFO(FIFO), .LOOKUP(LOOKUP), .LMAX(LMAX)) u (
      .clk, .rst, .start, .ntiles, .lut_we, .lut_addr, .lut_wdata, .srt_we, .srt_addr, .srt_wdata,
      .limit, .off, .desc_start(desc_start[l]), .desc_end(desc_end[l]), .rtile(rtile[l]),
      .want(want[l]), .grant(grant[l]), .req_addr(laddr[l]), .req_meta(lmeta[l]),
      .resp_valid(resp_valid[l]), .resp_data(resp_data[l]), .resp_meta(resp_meta[l]),
      .out_valid(out_valid[l]), .out_sym(out_sym[l]), .out_idx(out_idx[l]), .out_tile(out_tile[l]),
      .busy(lbusy[l]), .stall(lstall[l]));
  end

  huff_arb #(.LANES(LANES), .PORT(PORT)) u_arb (
    .clk, .rst, .start, .want, .laddr, .lmeta, .grant, .req_valid, .req_lane, .req_addr, .req_meta);

  always_comb begin
    busy = |lbusy; n_stall = '0;
    for (int l = 0; l < LANES; l++) n_stall = n_stall + 6'(lstall[l]);
  end
endmodule
