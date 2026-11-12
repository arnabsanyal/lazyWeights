// 8N1 UART, CPB clock cycles per bit (25 MHz / 1 Mbaud = 25).
module uart_rx #(parameter int CPB = 50) (
  input  logic clk, rst, rx,
  output logic       valid,
  output logic [7:0] data
);
  logic [1:0] s;                       // 2-flop synchronizer
  always_ff @(posedge clk) s <= {s[0], rx};
  wire r = s[1];
  logic [$clog2(CPB*3/2+1)-1:0] cnt;
  logic [3:0] bitn;
  logic busy;
  always_ff @(posedge clk) begin
    valid <= 1'b0;
    if (rst) begin busy <= 1'b0; end
    else if (!busy) begin
      if (!r) begin busy <= 1'b1; cnt <= CPB * 3 / 2 - 1; bitn <= 0; end   // to the middle of bit 0
    end else if (cnt != 0) cnt <= cnt - 1;
    else if (bitn < 8) begin data <= {r, data[7:1]}; bitn <= bitn + 1; cnt <= CPB - 1; end
    else begin busy <= 1'b0; valid <= r; end                                // stop bit must be 1
  end
endmodule

module uart_tx #(parameter int CPB = 50) (
  input  logic clk, rst,
  input  logic       start,
  input  logic [7:0] data,
  output logic       busy,
  output logic       tx
);
  logic [9:0] sh;
  logic [$clog2(CPB)-1:0] cnt;
  logic [3:0] n;
  initial begin tx = 1'b1; busy = 1'b0; end   // line idle from configuration (FPGA register init)
  always_ff @(posedge clk) begin
    if (rst) begin busy <= 1'b0; tx <= 1'b1; end
    else if (!busy) begin
      if (start) begin sh <= {1'b1, data, 1'b0}; busy <= 1'b1; cnt <= CPB - 1; n <= 0; tx <= 1'b0; end
    end else if (cnt != 0) cnt <= cnt - 1;
    else if (n < 9) begin n <= n + 1; cnt <= CPB - 1; tx <= sh[n + 1]; end
    else begin busy <= 1'b0; tx <= 1'b1; end
  end
endmodule
