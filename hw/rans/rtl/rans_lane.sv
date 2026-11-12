// A4 coder ablation: one rANS decode lane, the same plumbing as hw/huff/rtl/huff_lane.sv (request side walking the
// lane's streams tile after tile, prefetch FIFO of tagged words, 64-bit bit buffer, a tile's remaining bits dropped
// at its end); only the decode stage differs. Static rANS, 12-bit probabilities, 32-bit state, 16-bit renormalization
// (state x in [2^16, 2^32): at most one 16-bit word per symbol), one table per matrix broadcast by the engine (cum /
// freq registers, as the Huffman comparator limits). Per cycle:
//   - a tile's first cycle loads the state: x = the stream's first 32 bits (one cycle per tile without a symbol);
//   - then one symbol: slot = x mod 2^12; s = #{k >= 1 : slot >= cum[k]} (A - 1 comparators, thermometer, one-hot
//     select of freq[s] and cum[s]); x' = freq[s] * (x >> 12) + slot - cum[s]; x' < 2^16: x = x' << 16 | next 16 bits.
// A = 256: pair symbols (two 4-bit codes, as huf4p; NSYM = 512 symbols per tile lane); A = 16: single 4-bit codes
// (NSYM = 1024: half the weights per cycle).
module rans_lane #(
  parameter int A = 256,
  parameter int BUF = 64,
  parameter int FIFO = 4,
  parameter int NSYM = A == 256 ? 512 : 1024
) (
  input  logic        clk,
  input  logic        rst,
  input  logic        start,
  input  logic [15:0] ntiles,
  input  logic [A-1:1][11:0] cum,     // cumulative frequency of symbol k (cum[0] = 0), increasing
  input  logic [A-1:0][12:0] freq,    // frequency (1..4096), sum 4096
  input  logic [31:0] desc_start,
  input  logic [31:0] desc_end,
  output logic [15:0] rtile,
  output logic        want,
  input  logic        grant,
  output logic [31:0] req_addr,
  output logic [7:0]  req_meta,       // tag[7:6] first[5] skip[4:0]
  input  logic        resp_valid,
  input  logic [31:0] resp_data,
  input  logic [7:0]  resp_meta,
  output logic        out_valid,
  output logic [7:0]  out_sym,
  output logic [10:0] out_idx,
  output logic [15:0] out_tile,
  output logic        busy,
  output logic        stall
);
  localparam int CW = $clog2(FIFO + 1);
  localparam int SB = $clog2(A);

  logic [63:0] bb;
  logic [6:0]  nb;
  logic [10:0] done;
  logic [15:0] tile;
  logic [31:0] x;
  logic        ld;                     // state loaded for the current tile
  logic [31:0] wa;
  logic        wa_valid;
  logic [CW-1:0] cnt, narr;
  logic [$clog2(FIFO)-1:0] head, tail;
  logic [39:0] fifo [0:FIFO-1];

  // ---- symbol lookup and state update
  logic [11:0] slot, c;
  logic [12:0] f;
  logic [A-1:1] t;
  logic [A-1:0] oh;
  logic [SB-1:0] s;
  logic [32:0] xn;
  logic need;
  assign slot = x[11:0];
  always_comb begin
    for (int k = 1; k < A; k++) t[k] = slot >= cum[k];
    oh[0] = !t[1];
    for (int k = 1; k < A - 1; k++) oh[k] = t[k] && !t[k + 1];
    oh[A - 1] = t[A - 1];
    s = '0; f = '0; c = '0;
    for (int k = 0; k < A; k++) begin
      s = s | (oh[k] ? SB'(k) : '0);
      f = f | (oh[k] ? freq[k] : '0);
      if (k > 0) c = c | (oh[k] ? cum[k] : '0);
    end
  end
  assign xn = 33'(f) * 33'(x[31:12]) + 33'(slot) - 33'(c);
  assign need = xn[32:16] == '0;

  // ---- decode side
  logic active, got, load, last, hv, stale, refill, pop;
  logic [39:0] hw;
  logic [6:0]  cons, nb_after;
  logic [63:0] bb_sh;
  logic [4:0]  skip;
  assign active = tile < ntiles;
  assign load = active && !ld && nb >= 7'd32;
  assign got = active && ld && (!need || nb >= 7'd16);
  assign last = got && done == 11'(NSYM - 1);
  assign cons = load ? 7'd32 : (got && need) ? 7'd16 : 7'd0;
  assign hv = narr != 0;
  assign hw = fifo[head];
  assign stale = hv && hw[39:38] == tile[1:0] - 2'd1;
  assign refill = !last && hv && hw[39:38] == tile[1:0] && nb_after <= 7'(BUF - 32);
  assign pop = refill || (hv && stale);
  assign nb_after = nb - cons;
  assign bb_sh = bb >> cons;
  assign skip = hw[37] ? hw[36:32] : 5'd0;

  // ---- request side (as huff_lane)
  logic [31:0] wcur, wend;
  assign wcur = wa_valid ? wa : {5'd0, desc_start[31:5]};
  assign wend = desc_end > desc_start ? (desc_end - 32'd1) >> 5 : desc_start >> 5;
  assign want = rtile < ntiles && (cnt - CW'(pop)) < CW'(FIFO);
  assign req_addr = wcur;
  assign req_meta = {rtile[1:0], wcur == {5'd0, desc_start[31:5]}, desc_start[4:0]};

  always_ff @(posedge clk) begin
    if (rst || start) begin
      bb <= '0; nb <= '0; done <= '0; tile <= '0; rtile <= '0; wa_valid <= 1'b0; x <= '0; ld <= 1'b0;
      cnt <= '0; narr <= '0; head <= '0; tail <= '0;
    end else begin
      if (last) begin
        bb <= '0; nb <= '0; done <= '0; tile <= tile + 16'd1; ld <= 1'b0;
      end else begin
        bb <= refill ? (bb_sh | ((64'(hw[31:0]) >> skip) << nb_after)) : bb_sh;
        nb <= refill ? nb_after + 7'd32 - {2'd0, skip} : nb_after;
        done <= done + {10'd0, got};
        if (load) begin x <= bb[31:0]; ld <= 1'b1; end
        else if (got) x <= need ? {xn[15:0], bb[15:0]} : xn[31:0];
      end
      if (pop) head <= head + 1'b1;
      if (resp_valid) begin
        fifo[tail] <= {resp_meta, resp_data};
        tail <= tail + 1'b1;
      end
      narr <= narr + CW'(resp_valid) - CW'(pop);
      cnt <= cnt + CW'(grant) - CW'(pop);
      if (grant) begin
        if (wcur == wend) begin rtile <= rtile + 16'd1; wa_valid <= 1'b0; end
        else begin wa <= wcur + 32'd1; wa_valid <= 1'b1; end
      end
    end
  end

  assign out_valid = got;
  assign out_sym = 8'(s);
  assign out_idx = done;
  assign out_tile = tile;
  assign busy = active;
  assign stall = active && !got;
endmodule
