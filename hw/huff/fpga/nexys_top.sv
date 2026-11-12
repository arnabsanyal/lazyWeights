// FPGA bring-up of the Huffman tile engine (hw/huff/rtl/huff_engine.sv, 32 lanes, comparator lookup, buffer 64,
// FIFO 4, port 8) on a Digilent Nexys 4 DDR (XC7A100T), driven over the USB-UART by scripts/fpga_host.py.
// Memory: the slice's payload words in on-chip block RAM, 4 copies x 2 ports = the engine's 8 read ports; every
// granted word comes back LAT = 8 cycles after its grant, as the C model / co-simulation (hw/huff/tb/cosim_main.cpp).
// Descriptors (per lane and tile: stream start / end bits) in registers; decoded symbols into one RAM per lane.
// Counters: cycles while busy, stall lane-cycles (n_stall), words granted -> compared with the C model.
//
// Commands (bytes, little endian), each load answered with 0xAC:
//   01 lut: 2048 x u16 | 02 srt: 256 x u8 | 03 cmp: 11 x (u16 limit, u16 first, u16 base), lengths 1..11
//   04 desc: u8 nt, nt x 32 x (u32 start, u32 end) | 05 words: u32 n, n x u32
//   06 run: u8 nt -> A5, u32 cycles, u32 stalls, u32 words, u8 timeout
//   07 read: u8 nt -> 32 x nt x 512 bytes (lane, tile, symbol index)  | 08 ping -> 5A, 01 (version)
module huff_fpga_core #(
  parameter int CPB = 25,         // UART clocks per bit (25 MHz, 1 Mbaud)
  parameter int AW = 14,          // payload words per copy: 2^AW (16384 = 64 KB)
  parameter int NTM = 8,          // tiles per slice (max)
  parameter int LAT = 8
) (
  input  logic        clk,
  input  logic        rst_in,     // asynchronous request (button / MMCM not locked), synchronized here
  input  logic        uart_rx_i,
  output logic        uart_tx_o,
  output logic [15:0] led
);
  localparam int L = 32, P = 8, TB = $clog2(NTM);
  logic [3:0] rs;
  always_ff @(posedge clk) rs <= {rs[2:0], rst_in};
  wire rst = rs[3];
  wire locked = 1'b1;

  // ---- UART
  logic rxv, txs, txb;
  logic [7:0] rxd, txd;
  uart_rx #(.CPB(CPB)) u_rx (.clk, .rst, .rx(uart_rx_i), .valid(rxv), .data(rxd));
  uart_tx #(.CPB(CPB)) u_tx (.clk, .rst, .start(txs), .data(txd), .busy(txb), .tx(uart_tx_o));

  // ---- engine
  logic        e_start, lut_we, srt_we, cmp_we;
  logic [15:0] e_ntiles;
  logic [10:0] lut_addr;
  logic [11:0] lut_wdata;
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
    .clk, .rst, .start(e_start), .ntiles(e_ntiles), .lut_we, .lut_addr, .lut_wdata, .srt_we, .srt_addr, .srt_wdata,
    .cmp_we, .cmp_l, .cmp_limit, .cmp_first, .cmp_base, .desc_start, .desc_end, .rtile, .req_valid, .req_lane,
    .req_addr, .req_meta, .resp_valid, .resp_data, .resp_meta, .out_valid, .out_sym, .out_idx, .out_tile, .busy,
    .n_stall);

  // ---- descriptors: per lane, per tile (registers), selected by the lane's request-side tile
  logic [31:0] ds [L][NTM], de [L][NTM];
  logic        d_we;
  logic [4:0]  d_lane;
  logic [TB-1:0] d_tile;
  logic [31:0] d_s, d_e;
  always_ff @(posedge clk) if (d_we) begin ds[d_lane][d_tile] <= d_s; de[d_lane][d_tile] <= d_e; end
  always_comb
    for (int l = 0; l < L; l++) begin
      desc_start[l] = rtile[l] < e_ntiles ? ds[l][rtile[l][TB-1:0]] : '0;
      desc_end[l]   = rtile[l] < e_ntiles ? de[l][rtile[l][TB-1:0]] : '0;
    end

  // ---- payload memory: 4 copies, true dual port; load writes all copies through port A
  logic          w_we;
  logic [AW-1:0] w_addr;
  logic [31:0]   w_data;
  logic          run;
  logic [P-1:0][31:0] rd;
  for (genvar c = 0; c < P / 2; c++) begin : g_mem
    (* ram_style = "block" *) logic [31:0] m [2**AW];
    wire [AW-1:0] aa = run ? req_addr[2*c][AW-1:0] : w_addr;
    always_ff @(posedge clk) begin
      if (w_we) m[aa] <= w_data;
      rd[2*c] <= m[aa];
    end
    always_ff @(posedge clk) rd[2*c+1] <= m[req_addr[2*c+1][AW-1:0]];
  end
  // response pipeline: request in cycle c -> RAM data in c + 1 -> LAT - 2 more registers -> presented in c + LAT - 1
  // (the co-simulation's due = c + LAT - 1, applied before that cycle's clock edge)
  logic [P-1:0]      pv [LAT-1];
  logic [P-1:0][4:0] pl [LAT-1];
  logic [P-1:0][7:0] pm [LAT-1];
  logic [P-1:0][31:0] pd [LAT-1];   // pd[0] unused (data comes from rd at stage 0)
  always_ff @(posedge clk) begin
    // requests of the start cycle are dropped (start clears the lanes' credits at that edge; the co-simulation and
    // the C model do not serve them either)
    pv[0] <= run && !e_start ? req_valid : '0; pl[0] <= req_lane; pm[0] <= req_meta;
    for (int s = 1; s < LAT - 1; s++) begin
      pv[s] <= pv[s-1]; pl[s] <= pl[s-1]; pm[s] <= pm[s-1]; pd[s] <= s == 1 ? rd : pd[s-1];
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

  // ---- outputs: one RAM per lane, address {tile, symbol index}
  logic [TB+8:0] o_raddr;
  logic [L-1:0][7:0] o_rd;
  for (genvar l = 0; l < L; l++) begin : g_out
    (* ram_style = "block" *) logic [7:0] o [2**(TB+9)];
    always_ff @(posedge clk) begin
      if (out_valid[l]) o[{out_tile[l][TB-1:0], out_idx[l]}] <= out_sym[l];
      o_rd[l] <= o[o_raddr];
    end
  end

  // ---- counters
  logic [31:0] cyc, stl, wrd;
  logic        tmo;
  function automatic logic [3:0] pop8(input logic [7:0] v);
    pop8 = '0; for (int i = 0; i < 8; i++) pop8 = pop8 + 4'(v[i]);
  endfunction

  // ---- command FSM
  typedef enum logic [3:0] {IDLE, LUT, SRT, CMP, DNT, DESC, WN, WORDS, RUNNT, RUN0, RUNNING, REPLY, RDNT, READ, ACK} st_t;
  st_t st;
  logic [31:0] sh;           // little-endian byte assembly
  logic [3:0]  nb;           // bytes assembled in the current field
  logic [31:0] cnt, total;
  logic [7:0]  nt;
  logic [31:0] dsave;
  logic [7:0]  rep [17];     // reply bytes
  logic [4:0]  rn, ri;       // reply length, index
  logic [4:0]  rl;           // readback lane
  logic [TB+8:0] rpos;
  logic [1:0]  rph;          // readback phase: 0 set address, 1 wait, 2 send
  wire [31:0] shn = {rxd, sh[31:8]};

  always_ff @(posedge clk) begin
    lut_we <= 1'b0; srt_we <= 1'b0; cmp_we <= 1'b0; d_we <= 1'b0; w_we <= 1'b0; e_start <= 1'b0; txs <= 1'b0;
    if (rst) begin
      st <= IDLE; run <= 1'b0; e_ntiles <= '0;
    end else case (st)
      IDLE: if (rxv) begin
        cnt <= 0; nb <= 0;
        case (rxd)
          8'h01: st <= LUT;
          8'h02: st <= SRT;
          8'h03: st <= CMP;
          8'h04: st <= DNT;
          8'h05: st <= WN;
          8'h06: st <= RUNNT;
          8'h07: st <= RDNT;
          8'h08: begin rep[0] <= 8'h5A; rep[1] <= 8'h01; rn <= 2; ri <= 0; st <= REPLY; end
          default: ;
        endcase
      end
      LUT: if (rxv) begin
        sh <= shn;
        if (nb == 1) begin
          lut_we <= 1'b1; lut_addr <= cnt[10:0]; lut_wdata <= shn[27:16]; nb <= 0; cnt <= cnt + 1;
          if (cnt == 2047) st <= ACK;
        end else nb <= nb + 1;
      end
      SRT: if (rxv) begin
        srt_we <= 1'b1; srt_addr <= cnt[7:0]; srt_wdata <= rxd; cnt <= cnt + 1;
        if (cnt == 255) st <= ACK;
      end
      CMP: if (rxv) begin       // 6 bytes per length: limit, first, base
        sh <= shn;
        if (nb == 5) begin
          cmp_we <= 1'b1; cmp_l <= 4'(cnt + 1);
          cmp_limit <= dsave[11:0]; cmp_first <= dsave[26:16]; cmp_base <= shn[24:16];
          nb <= 0; cnt <= cnt + 1;
          if (cnt == 10) st <= ACK;
        end else begin
          nb <= nb + 1;
          if (nb == 3) dsave <= shn;   // bytes 0..3: limit (lo 16) | first (hi 16)
        end
      end
      DNT: if (rxv) begin nt <= rxd; total <= {rxd, 5'd0}; st <= DESC; end   // nt x 32 records
      DESC: if (rxv) begin
        sh <= shn;
        if (nb == 3) dsave <= shn;
        if (nb == 7) begin
          d_we <= 1'b1; d_lane <= cnt[4:0]; d_tile <= cnt[5+TB-1:5]; d_s <= dsave; d_e <= shn;
          nb <= 0; cnt <= cnt + 1;
          if (cnt + 1 == total) st <= ACK;
        end else nb <= nb + 1;
      end
      WN: if (rxv) begin
        sh <= shn;
        if (nb == 3) begin total <= shn; nb <= 0; st <= shn == 0 ? ACK : WORDS; end else nb <= nb + 1;
      end
      WORDS: if (rxv) begin
        sh <= shn;
        if (nb == 3) begin
          w_we <= 1'b1; w_addr <= cnt[AW-1:0]; w_data <= shn; nb <= 0; cnt <= cnt + 1;
          if (cnt + 1 == total) st <= ACK;
        end else nb <= nb + 1;
      end
      RUNNT: if (rxv) begin e_ntiles <= {8'd0, rxd}; st <= RUN0; end
      RUN0: begin           // start edge (co-simulation: start = 1, clock, then the loop)
        e_start <= 1'b1; run <= 1'b1; cyc <= 0; stl <= 0; wrd <= 0; tmo <= 1'b0; st <= RUNNING;
      end
      RUNNING: if (!e_start) begin
        if (!busy || cyc[27]) begin
          run <= 1'b0; tmo <= cyc[27];
          rep[0] <= 8'hA5;
          {rep[4], rep[3], rep[2], rep[1]} <= cyc;
          {rep[8], rep[7], rep[6], rep[5]} <= stl;
          {rep[12], rep[11], rep[10], rep[9]} <= wrd;
          rep[13] <= {7'd0, cyc[27]};
          rn <= 14; ri <= 0; st <= REPLY;
        end else begin
          cyc <= cyc + 1; stl <= stl + 32'(n_stall); wrd <= wrd + 32'(pop8(req_valid));
        end
      end
      RDNT: if (rxv) begin nt <= rxd; rl <= 0; rpos <= 0; rph <= 0; st <= READ; end
      READ: case (rph)
        2'd0: begin o_raddr <= rpos; rph <= 2'd1; end
        2'd1: rph <= 2'd2;
        2'd2: if (!txb && !txs) begin
          txs <= 1'b1; txd <= o_rd[rl]; rph <= 2'd3;
        end
        2'd3: if (txb) begin   // byte accepted by the transmitter
          rph <= 2'd0;
          if (rpos == {nt[TB:0], 9'd0} - 1) begin
            rpos <= 0;
            if (rl == 31) st <= IDLE; else rl <= rl + 1;
          end else rpos <= rpos + 1;
        end
      endcase
      REPLY: if (!txb && !txs) begin
        txs <= 1'b1; txd <= rep[ri]; ri <= ri + 1;
        if (ri + 1 == rn) st <= IDLE;
      end
      ACK: begin rep[0] <= 8'hAC; rn <= 1; ri <= 0; st <= REPLY; end
      default: st <= IDLE;
    endcase
  end

  // ---- LEDs: 0 heartbeat, 1 run, 2 timeout, 15:8 state
  logic [25:0] hb;
  always_ff @(posedge clk) hb <= hb + 1;
  assign led = {4'd0, st, 4'd0, locked, tmo, run, hb[25]};
endmodule

// Board top: 100 MHz -> 25 MHz (MMCM: VCO 1000 MHz / 40; the engine reaches ~50 MHz out of context), CPU_RESET button (active low).
module nexys_top (
  input  logic        CLK100MHZ,
  input  logic        CPU_RESETN,
  input  logic        UART_TXD_IN,
  output logic        UART_RXD_OUT,
  output logic [15:0] LED
);
  logic clk, fb, locked;
  MMCME2_BASE #(.CLKIN1_PERIOD(10.0), .CLKFBOUT_MULT_F(10.0), .CLKOUT0_DIVIDE_F(40.0)) u_mmcm (
    .CLKIN1(CLK100MHZ), .CLKFBIN(fb), .CLKFBOUT(fb), .CLKOUT0(clk), .LOCKED(locked), .PWRDWN(1'b0), .RST(1'b0),
    .CLKOUT0B(), .CLKOUT1(), .CLKOUT1B(), .CLKOUT2(), .CLKOUT2B(), .CLKOUT3(), .CLKOUT3B(), .CLKOUT4(), .CLKOUT5(),
    .CLKOUT6(), .CLKFBOUTB());
  logic [15:0] led;
  huff_fpga_core u_core (.clk, .rst_in(~(locked & CPU_RESETN)), .uart_rx_i(UART_TXD_IN), .uart_tx_o(UART_RXD_OUT),
                         .led);
  assign LED = {led[15:4], locked, led[2:0]};
endmodule
