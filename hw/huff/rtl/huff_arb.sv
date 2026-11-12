// A4: memory port arbiter of the Huffman tile engine: up to PORT word grants per cycle, round-robin over the lanes
// that want a word (the pointer rr advances by one each cycle with a grant), as hw/huff/huff_engine.h. Masked
// round robin, log depth, AND-OR logic only (no writes through variable indices, which Yosys's proc passes do not
// scale to); counts are saturating thermometer codes (no carry chains). The pointer is kept as the thermometer
// mask m[l] = (l >= rr); a lane's rank in the scan rr, rr + 1, ... is its inclusive prefix count among the wanting
// lanes >= rr (masked), or that group's total plus its prefix count among all wanting lanes (lanes < rr: the prefix
// only covers lanes < rr). Lanes of rank <= PORT are granted; port p takes the lane of rank p + 1 (one-hot, AND-OR
// select of its address).
module huff_arb #(
  parameter int LANES = 32,
  parameter int PORT = 8
) (
  input  logic                   clk,
  input  logic                   rst,
  input  logic                   start,
  input  logic [LANES-1:0]       want,
  input  logic [LANES-1:0][31:0] laddr,
  input  logic [LANES-1:0][7:0]  lmeta,
  output logic [LANES-1:0]       grant,
  output logic [PORT-1:0]        req_valid,
  output logic [PORT-1:0][$clog2(LANES)-1:0] req_lane,   // 5 bits at 32 lanes
  output logic [PORT-1:0][31:0]  req_addr,
  output logic [PORT-1:0][7:0]   req_meta
);
  localparam int LW = $clog2(LANES);
  // counts as saturating thermometer codes: bit i = (count >= i + 1), i = 0..PORT (bit PORT: count > PORT)
  localparam int TB = PORT + 1;
  function automatic logic [TB-1:0] tadd(input logic [TB-1:0] a, input logic [TB-1:0] b);
    for (int i = 0; i < TB; i++) begin
      tadd[i] = a[i] | b[i];
      for (int j = 0; j < i; j++) tadd[i] = tadd[i] | (a[j] & b[i - 1 - j]);
    end
  endfunction
  logic [LANES-1:0] mask, wm;
  logic [LANES-1:0][TB-1:0] pm, pa, rank;
  logic [PORT-1:0][LANES-1:0] sel;

  assign wm = want & mask;
  always_comb begin
    for (int k = 0; k < LANES; k++) begin
      pm[k] = TB'(wm[k]);
      pa[k] = TB'(want[k]);
    end
    for (int lv = 1; lv < LANES; lv = lv * 2)   // Kogge-Stone prefix
      for (int k = LANES - 1; k >= lv; k--) begin
        pm[k] = tadd(pm[k], pm[k - lv]);
        pa[k] = tadd(pa[k], pa[k - lv]);
      end
    for (int k = 0; k < LANES; k++) begin
      rank[k] = mask[k] ? pm[k] : tadd(pm[LANES - 1], pa[k]);
      grant[k] = want[k] && !rank[k][PORT];
    end
  end

  for (genvar p = 0; p < PORT; p++) begin : g_port
    for (genvar k = 0; k < LANES; k++) begin : g_k
      assign sel[p][k] = want[k] && rank[k][p] && !rank[k][p + 1];   // rank == p + 1
    end
    always_comb begin
      req_valid[p] = |sel[p];
      req_lane[p] = '0; req_addr[p] = '0; req_meta[p] = '0;
      for (int l = 0; l < LANES; l++) begin
        req_lane[p] = req_lane[p] | ({LW{sel[p][l]}} & LW'(l));
        req_addr[p] = req_addr[p] | ({32{sel[p][l]}} & laddr[l]);
        req_meta[p] = req_meta[p] | ({8{sel[p][l]}} & lmeta[l]);
      end
    end
  end

  // rr <- rr + 1 when some lane is granted (= some lane wants, PORT >= 1); rr = LANES - 1 wraps to 0 (all ones)
  always_ff @(posedge clk)
    if (rst || start) mask <= '1;
    else if (|want) mask <= mask[LANES-2:0] == '0 ? '1 : mask << 1;
endmodule
