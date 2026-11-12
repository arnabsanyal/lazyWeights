// ZCU102 prototype of the Huffman tile engine (hw/huff/rtl/huff_engine.sv, 32 lanes, comparator lookup, buffer 64,
// FIFO 4, port 8): NE engines in the fabric of the XCZU9EG, driven by the Cortex-A53s over AXI (zcu_top.sv,
// scripts/zcu_host.py). Each engine decodes a chunk of a matrix (hw/huff/tb/chunks_main.cpp) from its own block RAM:
// payload words in 4 copies x 2 ports = the engine's 8 read ports, every granted word returned LAT = 8 cycles after
// its grant (as the C model / co-simulation, and the Nexys 4 DDR bring-up, nexys_top.sv); per-lane descriptors in
// LUT RAM. Decoded symbols are not stored: each lane folds its symbol stream into a CRC-32 (zlib), checked by the
// host against the C model's. Counters: cycles while busy, stall lane-cycles, words granted (vs the C model).
//
// Register bus (byte addresses, 32-bit words; writes from the host only while the engine is idle):
//   engine e window: e << 21 (2 MB), e < NE
//     0x000000 + 4 i  payload word i (write)              0x080000 + 8 (lane NTM + tile) + 4 s  descriptor start / end
//     0x0C0000 + 4 i  lut entry i (write, 12 bits)        0x0D0000 + 4 i  sorted symbol i (write)
//     0x0E0000 + 16 l + 4 f  comparator length l (1..11): f = 0 limit, 1 first, 2 base (writing base commits)
//     0x100000 + 4 r  r = 0 ntiles (r/w), 1 status {tmo, run, busy}, 2 cycles, 3 stall lane-cycles, 4 words,
//                     64 + lane: CRC-32 register (host: value ^ 0xFFFFFFFF = zlib.crc32), 128 + lane: symbols
//   global window: 7 << 21
//     0xE00000 id "HUF2" | 0xE00004 {LAT, NTM, AW, NE} | 0xE00008 start mask (write: engines in the mask start in
//     the same cycle; read: run mask) | 0xE0000C clock (kHz, from the top) | 0xE00010 free-running cycle counter
module huff_zunit #(
  parameter int AW = 15,          // payload words per copy: 2^AW
  parameter int NTM = 16,         // tiles per chunk (descriptor depth)
  parameter int LAT = 8,
  parameter bit AREG = 1          // register the RAM addresses (same LAT)
) (
  input  logic        clk,
  input  logic        rst,
  input  logic        go,         // start pulse
  input  logic        we,
  input  logic [20:0] wa,
  input  logic [31:0] wd,
  input  logic [20:0] ra,
  output logic [31:0] rd,
  output logic        running
);
  localparam int L = 32, P = 8, TB = $clog2(NTM);

  // ---- engine
  logic        e_start, srt_we, cmp_we;
  logic [15:0] e_ntiles;
  logic [7:0]  srt_addr, srt_wdata;
  logic [3:0]  cmp_l;
  logic [11:0] cmp_limit;
  logic [10:0] cmp_first;
  logic [8:0]  cmp_base;
  logic [L-1:0][31:0] desc_start, desc_end;
  logic [L-1:0][15:0] rtile, out_tile;
  logic [P-1:0]       req_valid;
  logic [P-1:0][4:0]  req_lane;
  logic [P-1:0][31:0] req_addr;
  logic [P-1:0][7:0]  req_meta;
  logic [L-1:0]       resp_valid, out_valid;
  logic [L-1:0][31:0] resp_data;
  logic [L-1:0][7:0]  resp_meta, out_sym;
  logic [L-1:0][8:0]  out_idx;
  logic               busy;
  logic [5:0]         n_stall;
  huff_engine #(.LANES(L), .BUF(64), .FIFO(4), .PORT(P), .LOOKUP(1)) u_eng (
    .clk, .rst, .start(e_start), .ntiles(e_ntiles), .lut_we(1'b0), .lut_addr('0), .lut_wdata('0), .srt_we,
    .srt_addr, .srt_wdata, .cmp_we, .cmp_l, .cmp_limit, .cmp_first, .cmp_base, .desc_start, .desc_end, .rtile,
    .req_valid, .req_lane, .req_addr, .req_meta, .resp_valid, .resp_data, .resp_meta, .out_valid, .out_sym, .out_idx,
    .out_tile, .busy, .n_stall);

  // ---- host writes (registered once more here)
  logic        w1;
  logic [20:0] a1;
  logic [31:0] d1;
  always_ff @(posedge clk) begin w1 <= we && !running; a1 <= wa; d1 <= wd; end
  wire w_pay  = w1 && a1[20:19] == 2'b00;
  wire w_desc = w1 && a1[20:18] == 3'b010;
  wire w_srt  = w1 && a1[20:16] == 5'b01101;
  wire w_cmp  = w1 && a1[20:16] == 5'b01110;
  wire w_ctl  = w1 && a1[20];

  // ---- descriptors: per lane, NTM x (start, end) in LUT RAM, read by the lane's request-side tile
  for (genvar l = 0; l < L; l++) begin : g_desc
    (* ram_style = "distributed" *) logic [31:0] ms [NTM];
    (* ram_style = "distributed" *) logic [31:0] me [NTM];
    wire [TB-1:0] wt = a1[3 +: TB];
    wire hit = w_desc && a1[3 + TB +: 5] == 5'(l);
    always_ff @(posedge clk) begin
      if (hit && !a1[2]) ms[wt] <= d1;
      if (hit && a1[2]) me[wt] <= d1;
    end
    assign desc_start[l] = rtile[l] < e_ntiles ? ms[rtile[l][TB-1:0]] : '0;
    assign desc_end[l]   = rtile[l] < e_ntiles ? me[rtile[l][TB-1:0]] : '0;
  end

  // ---- tables
  logic [31:0] cmp_save;
  always_ff @(posedge clk) begin
    srt_we <= w_srt; srt_addr <= a1[9:2]; srt_wdata <= d1[7:0];
    cmp_we <= 1'b0;
    if (w_cmp) begin
      case (a1[3:2])
        2'd0: cmp_save[11:0] <= d1[11:0];
        2'd1: cmp_save[26:16] <= d1[10:0];
        2'd2: begin
          cmp_we <= 1'b1; cmp_l <= a1[7:4];
          cmp_limit <= cmp_save[11:0]; cmp_first <= cmp_save[26:16]; cmp_base <= d1[8:0];
        end
        default: ;
      endcase
    end
  end

  // ---- payload memory: 4 copies, true dual port; host writes go to all copies through port A
  logic run;
  logic [P-1:0][31:0] prd;
  // AREG = 1: the granted addresses are registered before the RAMs (cuts the lane decode -> arbiter -> far-away BRAM
  // path), and the response pipeline is one register shorter, so words still arrive LAT cycles after their grant.
  logic [P-1:0][AW-1:0] qa;
  for (genvar p = 0; p < P; p++)
    if (AREG) begin : g_qa
      always_ff @(posedge clk) qa[p] <= req_addr[p][AW-1:0];
    end else begin : g_qa
      assign qa[p] = req_addr[p][AW-1:0];
    end
  for (genvar c = 0; c < P / 2; c++) begin : g_mem
    (* ram_style = "block" *) logic [31:0] m [2**AW];
    wire [AW-1:0] aa = run ? qa[2*c] : a1[AW+1:2];
    always_ff @(posedge clk) begin
      if (w_pay) m[aa] <= d1;
      prd[2*c] <= m[aa];
    end
    always_ff @(posedge clk) prd[2*c+1] <= m[qa[2*c+1]];
  end
  // response pipeline: request in cycle c -> (AREG: address register in c + 1) -> RAM data in c + 1 + AREG ->
  // more registers -> presented in c + LAT - 1
  logic [P-1:0]      pv [LAT-1];
  logic [P-1:0][4:0] pl [LAT-1];
  logic [P-1:0][7:0] pm [LAT-1];
  logic [P-1:0][31:0] pd [LAT-1];
  always_ff @(posedge clk) begin
    pv[0] <= run && !e_start ? req_valid : '0; pl[0] <= req_lane; pm[0] <= req_meta;   // start-cycle requests dropped
    for (int s = 1; s < LAT - 1; s++) begin
      pv[s] <= pv[s-1]; pl[s] <= pl[s-1]; pm[s] <= pm[s-1]; pd[s] <= s == 1 + AREG ? prd : pd[s-1];
    end
  end
  always_comb begin
    resp_valid = '0; resp_data = '0; resp_meta = '0;
    for (int p = 0; p < P; p++)
      if (pv[LAT-2][p]) begin
        resp_valid[pl[LAT-2][p]] = 1'b1;
        resp_data[pl[LAT-2][p]] = pd[LAT-2][p];
        resp_meta[pl[LAT-2][p]] = pm[LAT-2][p];
      end
  end

  // ---- per-lane CRC-32 (reflected 0xEDB88320, zlib) and symbol counts
  function automatic logic [31:0] crc8(input logic [31:0] c, input logic [7:0] d);
    logic [31:0] x;
    x = c ^ {24'd0, d};
    for (int k = 0; k < 8; k++) x = x[0] ? (x >> 1) ^ 32'hEDB88320 : x >> 1;
    return x;
  endfunction
  logic [L-1:0][31:0] crc, nsym;
  // outputs registered once (timing), folded the next cycle
  logic [L-1:0]      ov;
  logic [L-1:0][7:0] os;
  always_ff @(posedge clk) begin
    ov <= out_valid; os <= out_sym;
    for (int l = 0; l < L; l++)
      if (e_start) begin crc[l] <= '1; nsym[l] <= '0; end
      else if (ov[l]) begin crc[l] <= crc8(crc[l], os[l]); nsym[l] <= nsym[l] + 1; end
  end

  // ---- run control and counters (as nexys_top.sv: start edge, then count while busy)
  logic [31:0] cyc, stl, wrd;
  logic        tmo;
  function automatic logic [3:0] pop8(input logic [7:0] v);
    pop8 = '0; for (int i = 0; i < 8; i++) pop8 = pop8 + 4'(v[i]);
  endfunction
  always_ff @(posedge clk) begin
    e_start <= 1'b0;
    if (rst) begin
      run <= 1'b0; e_ntiles <= '0; tmo <= 1'b0;
    end else begin
      if (w_ctl && a1[11:2] == 10'd0) e_ntiles <= d1[15:0];
      if (go && !run) begin
        e_start <= 1'b1; run <= 1'b1; cyc <= 0; stl <= 0; wrd <= 0; tmo <= 1'b0;
      end else if (run && !e_start) begin
        if (!busy || cyc[30]) begin run <= 1'b0; tmo <= cyc[30]; end
        else begin cyc <= cyc + 1; stl <= stl + 32'(n_stall); wrd <= wrd + 32'(pop8(req_valid)); end
      end
    end
  end
  // the CRC fold lags the decode by two cycles: report running until it has caught up
  logic [1:0] tail;
  always_ff @(posedge clk) tail <= {tail[0], run};
  assign running = run || |tail;

  // ---- register reads (registered)
  always_ff @(posedge clk) begin
    if (ra[20]) begin
      case (ra[11:2]) inside
        10'd0: rd <= {16'd0, e_ntiles};
        10'd1: rd <= {29'd0, tmo, running, busy};
        10'd2: rd <= cyc;
        10'd3: rd <= stl;
        10'd4: rd <= wrd;
        [10'd64:10'd95]:   rd <= crc[ra[6:2]];
        [10'd128:10'd159]: rd <= nsym[ra[6:2]];
        default: rd <= 32'hDEAD_0000 | {22'd0, ra[11:2]};
      endcase
    end else rd <= '0;
  end
endmodule

// NE engines + the global registers on one register bus (requests registered here, read data valid 3 cycles after ra)
module huff_zcu_core #(
  parameter int NE = 4,
  parameter int AW = 15,
  parameter int NTM = 16,
  parameter int LAT = 8,
  parameter bit AREG = 1,
  parameter int CLK_KHZ = 100000
) (
  input  logic        clk,
  input  logic        rst,
  input  logic        we,
  input  logic [23:0] wa,
  input  logic [31:0] wd,
  input  logic [23:0] ra,
  output logic [31:0] rd
);
  logic        we1;
  logic [23:0] wa1, ra1;
  logic [31:0] wd1;
  logic [NE-1:0] go, run;
  logic [NE-1:0][31:0] urd;
  logic [31:0] fc;
  always_ff @(posedge clk) begin
    we1 <= we; wa1 <= wa; wd1 <= wd; ra1 <= ra;
    fc <= fc + 1;
    go <= '0;
    if (we && wa[23:21] == 3'd7 && wa[4:2] == 3'd2) go <= wd[NE-1:0];
  end
  for (genvar e = 0; e < NE; e++) begin : g_u
    huff_zunit #(.AW(AW), .NTM(NTM), .LAT(LAT), .AREG(AREG)) u (
      .clk, .rst, .go(go[e]), .we(we1 && wa1[23:21] == 3'(e)), .wa(wa1[20:0]), .wd(wd1), .ra(ra1[20:0]), .rd(urd[e]),
      .running(run[e]));
  end
  logic [23:0] ra2;
  logic [31:0] grd;
  always_ff @(posedge clk) begin
    ra2 <= ra1;
    case (ra1[4:2])
      3'd0: grd <= 32'h3246_5548;   // "HUF2"
      3'd1: grd <= {8'(LAT), 8'(NTM), 8'(AW), 8'(NE)};
      3'd2: grd <= 32'(run);
      3'd3: grd <= 32'(CLK_KHZ);
      3'd4: grd <= fc;
      default: grd <= '0;
    endcase
    rd <= ra2[23:21] == 3'd7 ? grd : (ra2[23:21] < 3'(NE) ? urd[ra2[23:21]] : 32'hBAD0_0000);
  end
endmodule

// AXI4-Lite slave -> register bus (one transaction at a time; read data 4 cycles after the address)
module axil_regs #(
  parameter int AWID = 24
) (
  input  logic            clk,
  input  logic            rstn,
  input  logic [31:0]     awaddr,
  input  logic            awvalid,
  output logic            awready,
  input  logic [31:0]     wdata,
  input  logic [3:0]      wstrb,
  input  logic            wvalid,
  output logic            wready,
  output logic [1:0]      bresp,
  output logic            bvalid,
  input  logic            bready,
  input  logic [31:0]     araddr,
  input  logic            arvalid,
  output logic            arready,
  output logic [31:0]     rdata,
  output logic [1:0]      rresp,
  output logic            rvalid,
  input  logic            rready,
  output logic            we,
  output logic [AWID-1:0] wa,
  output logic [31:0]     wd,
  output logic [AWID-1:0] ra,
  input  logic [31:0]     rd
);
  assign bresp = 2'b00;
  assign rresp = 2'b00;
  // write: take AW and W together
  always_ff @(posedge clk) begin
    we <= 1'b0; awready <= 1'b0; wready <= 1'b0;
    if (!rstn) bvalid <= 1'b0;
    else begin
      if (bvalid && bready) bvalid <= 1'b0;
      if (awvalid && wvalid && !awready && !bvalid) begin
        awready <= 1'b1; wready <= 1'b1; bvalid <= 1'b1;
        we <= 1'b1; wa <= awaddr[AWID-1:0]; wd <= wdata;
      end
    end
  end
  // read
  logic [3:0] rw;
  always_ff @(posedge clk) begin
    arready <= 1'b0;
    if (!rstn) begin rvalid <= 1'b0; rw <= '0; end
    else begin
      rw <= {rw[2:0], 1'b0};
      if (rvalid && rready) rvalid <= 1'b0;
      if (arvalid && !arready && !rvalid && rw == 0) begin arready <= 1'b1; ra <= araddr[AWID-1:0]; rw <= 4'd1; end
      if (rw[3]) begin rvalid <= 1'b1; rdata <= rd; end
    end
  end
endmodule
