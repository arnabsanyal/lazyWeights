# FPGA bring-up of the Huffman tile engine (Nexys 4 DDR, XC7A100T)

32-lane engine (hw/huff/rtl, comparator lookup, buffer 64, FIFO 4, port 8) at 25 MHz; payload in block RAM (4 copies = 8 read ports), every word returned 8 cycles after its grant; driven over the USB-UART from the ZCU102 (scripts/fpga_host.py). Jobs: tile slices (first / middle / last) of the co-simulation's matrices (scripts/fpga_jobs.py); reference: the C model on the same slice.

- jobs: 87, ok: 87; matrices: 29 (huf4p, huf8)
- symbols: 7127040, mismatches: 0
- cycles: 225334 on the board, 225334 in the C model; stalls and words equal on every job: True
- utilization (whole design): 46823 LUTs (74%), 23038 FFs, 96 BRAM36, 0 DSP; timing at 25 MHz: WNS +2.204 ns, WHS +0.035 ns
- engine alone (out of context, hw/huff/fpga ooc): 36,145 LUTs, 5,211 FFs, 0 BRAM; critical path ~19.6 ns (~50 MHz)
