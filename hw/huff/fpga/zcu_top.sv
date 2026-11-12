// ZCU102 board top: the 300 MHz user clock (Si570 USER_SI570_SYSCLK, AL8/AL7, HP bank 64; CLK_125 sits on an HD
// bank, which cannot drive an MMCM) -> MMCM -> fabric clock (300 MHz x MULT / DIV);
// the PS block design (zps: Zynq UltraScale+ PS, M_AXI_HPM0_LPD at 0x8000_0000 -> SmartConnect -> AXI4-Lite) clocked
// by the same clock; axil_regs -> huff_zcu_core (NE engines). The engines' clock does not depend on the PS's clock
// configuration (a bitstream loaded from Linux does not change it).
module zcu_top #(
  parameter int NE = 4,
  parameter real MULT = 4.0,      // VCO = 300 MHz x MULT (800..1600 MHz)
  parameter real DIV = 12.0,       // fabric clock = VCO / DIV
  parameter int CLK_KHZ = 100000
) (
  input  logic SYSCLK_P,
  input  logic SYSCLK_N
);
  logic clk_in, clk_mmcm, clk, fb, locked;
  IBUFDS u_ibuf (.I(SYSCLK_P), .IB(SYSCLK_N), .O(clk_in));
  MMCME4_BASE #(.CLKIN1_PERIOD(3.333), .CLKFBOUT_MULT_F(MULT), .DIVCLK_DIVIDE(1), .CLKOUT0_DIVIDE_F(DIV)) u_mmcm (
    .CLKIN1(clk_in), .CLKFBIN(fb), .CLKFBOUT(fb), .CLKOUT0(clk_mmcm), .LOCKED(locked), .PWRDWN(1'b0), .RST(1'b0),
    .CLKOUT0B(), .CLKOUT1(), .CLKOUT1B(), .CLKOUT2(), .CLKOUT2B(), .CLKOUT3(), .CLKOUT3B(), .CLKOUT4(), .CLKOUT5(),
    .CLKOUT6(), .CLKFBOUTB());
  BUFG u_bufg (.I(clk_mmcm), .O(clk));

  logic [7:0] rs;
  always_ff @(posedge clk or negedge locked)
    if (!locked) rs <= '0; else rs <= {rs[6:0], 1'b1};
  wire rstn = rs[7];

  logic [31:0] awaddr, wdata, araddr, rdata;
  logic [3:0]  wstrb;
  logic [2:0]  awprot, arprot;
  logic        awvalid, awready, wvalid, wready, bvalid, bready, arvalid, arready, rvalid, rready;
  logic [1:0]  bresp, rresp;
  zps_wrapper u_ps (
    .clk(clk), .resetn(rstn),
    .M_AXI_araddr(araddr), .M_AXI_arprot(arprot), .M_AXI_arready(arready), .M_AXI_arvalid(arvalid),
    .M_AXI_awaddr(awaddr), .M_AXI_awprot(awprot), .M_AXI_awready(awready), .M_AXI_awvalid(awvalid),
    .M_AXI_bready(bready), .M_AXI_bresp(bresp), .M_AXI_bvalid(bvalid), .M_AXI_rdata(rdata), .M_AXI_rready(rready),
    .M_AXI_rresp(rresp), .M_AXI_rvalid(rvalid), .M_AXI_wdata(wdata), .M_AXI_wready(wready), .M_AXI_wstrb(wstrb),
    .M_AXI_wvalid(wvalid));

  logic        we;
  logic [23:0] wa, ra;
  logic [31:0] wd, rd;
  axil_regs #(.AWID(24)) u_axil (
    .clk, .rstn, .awaddr, .awvalid, .awready, .wdata, .wstrb, .wvalid, .wready, .bresp, .bvalid, .bready, .araddr,
    .arvalid, .arready, .rdata, .rresp, .rvalid, .rready, .we, .wa, .wd, .ra, .rd);
  huff_zcu_core #(.NE(NE), .AW(15), .NTM(16), .LAT(8), .CLK_KHZ(CLK_KHZ)) u_core (
    .clk, .rst(!rstn), .we, .wa, .wd, .ra, .rd);
endmodule
