// A4: one lane datapath of the Huffman tile engine (cycle-equivalent to hw/huff/huff_engine.h, PRE = 1).
// Decode side: lookup on the low 11 bits of the bit buffer (LOOKUP 0: 2048-entry table, 1: canonical first-code
// comparators + a 256-entry symbol RAM), shifter, refill from the prefetch FIFO. Request side: walks the lane's
// streams (tile after tile, [desc_start, desc_end) bits from the descriptor inputs) one word per grant; words are
// tagged with their tile (2 bits) and the first word of a stream carries the bits to drop.
// Re-centering (q = r + add) is outside the lane (output writer): out_sym is the decoded symbol r.
module huff_lane #(
  parameter int BUF = 64,           // bit buffer (42..64)
  parameter int FIFO = 4,           // words incl. in flight
  parameter int LOOKUP = 1,
  parameter int LMAX = 11
) (
  input  logic        clk,
  input  logic        rst,
  input  logic        start,        // begin a matrix of ntiles tiles (this lane's streams)
  input  logic [15:0] ntiles,
  // table load (broadcast to all lanes)
  input  logic        lut_we,
  input  logic [10:0] lut_addr,
  input  logic [11:0] lut_wdata,    // sym | len << 8
  input  logic        srt_we,
  input  logic [7:0]  srt_addr,
  input  logic [7:0]  srt_wdata,
  input  logic [LMAX-1:1][11:0] limit,   // comparator tables (broadcast registers of the engine), by length; limit
                                        // = 2^11 when no code is longer (12 bits)
  input  logic [LMAX:1][7:0]    off,      // base[l] - first[l] mod 256: sorted index = code + off[l]
  // stream descriptor of the request side's current tile (rtile)
  input  logic [31:0] desc_start,
  input  logic [31:0] desc_end,
  output logic [15:0] rtile,
  // memory port
  output logic        want,
  input  logic        grant,
  output logic [31:0] req_addr,
  output logic [7:0]  req_meta,     // tag[7:6] first[5] skip[4:0]
  input  logic        resp_valid,
  input  logic [31:0] resp_data,
  input  logic [7:0]  resp_meta,
  // output
  output logic        out_valid,
  output logic [7:0]  out_sym,
  output logic [8:0]  out_idx,      // symbol index in the tile's lane stream (rows l, l + 32: 0..511)
  output logic [15:0] out_tile,
  output logic        busy,
  output logic        stall
);
  localparam int CW = $clog2(FIFO + 1);

  logic [63:0] bb;
  logic [6:0]  nb;
  logic [8:0]  done;
  logic [15:0] tile;
  logic [31:0] wa;
  logic        wa_valid;
  logic [CW-1:0] cnt;                // words granted and not yet popped (credits)
  logic [CW-1:0] narr;               // words arrived in the FIFO
  logic [$clog2(FIFO)-1:0] head, tail;
  logic [39:0] fifo [0:FIFO-1];

  // ---- lookup
  logic [10:0] win;
  logic [7:0]  sym;
  logic [3:0]  len;
  assign win = bb[10:0];
  generate
    if (LOOKUP == 0) begin : g_table
      logic [11:0] lut [0:2047];
      always_ff @(posedge clk) if (lut_we) lut[lut_addr] <= lut_wdata;
      assign sym = lut[win][7:0];
      assign len = lut[win][11:8];
    end else begin : g_cmp
      // thermometer t[k] = (v >= limit[k]) (limits are monotone), one-hot length, symbol index = code + off[len]
      logic [7:0] sorted [0:255];
      logic [10:0] v;
      logic [LMAX-1:1] t;
      logic [LMAX:1] oh;
      logic [7:0] code, ofs;
      logic [3:0] l;
      always_ff @(posedge clk) if (srt_we) sorted[srt_addr] <= srt_wdata;
      always_comb begin
        for (int i = 0; i < LMAX; i++) v[LMAX - 1 - i] = win[i];
        for (int k = 1; k < LMAX; k++) t[k] = {1'b0, v} >= limit[k];
        oh[1] = !t[1];
        for (int k = 2; k < LMAX; k++) oh[k] = t[k - 1] && !t[k];
        oh[LMAX] = t[LMAX - 1];
        l = '0; code = '0; ofs = '0;
        for (int k = 1; k <= LMAX; k++) begin
          l = l | (oh[k] ? 4'(k) : 4'd0);
          code = code | (oh[k] ? 8'(v >> (LMAX - k)) : 8'd0);
          ofs = ofs | (oh[k] ? off[k] : 8'd0);
        end
      end
      assign sym = sorted[code + ofs];
      assign len = l;
    end
  endgenerate

  // ---- decode side
  logic active, got, last, hv, stale, refill, pop;
  logic [39:0] hw;
  logic [6:0]  nb_after;
  logic [63:0] bb_sh;
  logic [4:0]  skip;
  assign active = tile < ntiles;
  assign got = active && nb != 0 && {3'd0, len} <= nb;
  assign last = got && done == 9'd511;
  assign hv = narr != 0;
  assign hw = fifo[head];
  assign stale = hv && hw[39:38] == tile[1:0] - 2'd1;      // a word of the previous tile (the next tile's words wait)
  assign refill = !last && hv && hw[39:38] == tile[1:0] && nb_after <= 7'(BUF - 32);
  assign pop = refill || (hv && stale);
  assign nb_after = got ? nb - {3'd0, len} : nb;
  assign bb_sh = got ? bb >> len : bb;
  assign skip = hw[37] ? hw[36:32] : 5'd0;

  // ---- request side
  logic [31:0] wcur, wend;
  assign wcur = wa_valid ? wa : {5'd0, desc_start[31:5]};
  assign wend = desc_end > desc_start ? (desc_end - 32'd1) >> 5 : desc_start >> 5;
  assign want = rtile < ntiles && (cnt - CW'(pop)) < CW'(FIFO);
  assign req_addr = wcur;
  assign req_meta = {rtile[1:0], wcur == {5'd0, desc_start[31:5]}, desc_start[4:0]};

  always_ff @(posedge clk) begin
    if (rst || start) begin
      bb <= '0; nb <= '0; done <= '0; tile <= '0; rtile <= '0; wa_valid <= 1'b0;
      cnt <= '0; narr <= '0; head <= '0; tail <= '0;
    end else begin
      if (last) begin
        bb <= '0; nb <= '0; done <= '0; tile <= tile + 16'd1;
      end else begin
        bb <= refill ? (bb_sh | ((64'(hw[31:0]) >> skip) << nb_after)) : bb_sh;
        nb <= refill ? nb_after + 7'd32 - {2'd0, skip} : nb_after;
        done <= done + {8'd0, got};
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
  assign out_sym = sym;
  assign out_idx = done;
  assign out_tile = tile;
  assign busy = active;
  assign stall = active && !got;
endmodule
