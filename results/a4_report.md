# A4: hardware Huffman decoder

## C simulation (hw/huff/huff_engine.h) on every Huffman-coded matrix

Engine = 32 lanes, 1 symbol per lane per cycle target (huf4p: 2 weights per symbol). Reference config: 64-bit buffer, 4-word FIFO, 8-word (256-bit) read port, 8-cycle memory latency, stream-end prefetch.

| bundle | matrices | mismatches (table, comparator) | symbols / cycle | weights / cycle | stall | words / cycle | no prefetch | port 6 | port 7 | latency 32 (FIFO 8) |
|---|---|---|---|---|---|---|---|---|---|---|
| phi3_kq_IQ4_XS_ref | 161 | 0, 0 | 31.91 | 62.2 | 0.25% | 7.79 | 31.25 | 24.56 | 28.46 | 31.45 |
| phi3_kq_IQ4_XS_g1_lam0_c1.25 | 161 | 0, 0 | 31.92 | 62.2 | 0.24% | 6.87 | 31.47 | 27.66 | 31.89 | 31.91 |
| mistral_kq_IQ4_XS_g1_lam0 | 225 | 0, 0 | 31.92 | 61.5 | 0.24% | 7.23 | 31.46 | 26.18 | 30.24 | 31.86 |
| mistral_kq_IQ4_XS_g1_lam0_c1.5 | 225 | 0, 0 | 31.92 | 61.6 | 0.23% | 6.34 | 31.48 | 29.62 | 31.92 | 31.92 |
| smollm_kq_Q6_K_g1_lam0_c1.5 | 121 | 0, 0 | 31.93 | 31.9 | 0.22% | 4.90 | 31.49 | 31.93 | 31.93 | 31.93 |
| smollm_kq_IQ4_XS_g1_lam0 | 121 | 0, 0 | 31.92 | 60.3 | 0.24% | 7.23 | 31.44 | 26.24 | 30.15 | 31.85 |
| phi3_q4_g128_lam0.1 | 161 | 0, 0 | 31.92 | 63.8 | 0.24% | 7.17 | 31.46 | 26.63 | 31.00 | 31.90 |
| phi3_q4_channel_lam0.1 | 161 | 0, 0 | 31.92 | 63.8 | 0.23% | 6.20 | 31.48 | 30.48 | 31.92 | 31.92 |
| q8_gptq_huf8 | 161 | 0, 0 | 31.93 | 31.9 | 0.22% | 4.69 | 31.49 | 31.92 | 31.93 | 31.93 |
| mistral_q4_g128_lam0_ao | 161 | 0, 0 | 31.92 | 63.8 | 0.24% | 7.20 | 31.47 | 26.56 | 30.98 | 31.90 |
| mistral_q4_channel_lam0.1 | 161 | 0, 0 | 31.93 | 63.9 | 0.23% | 6.06 | 31.49 | 31.20 | 31.93 | 31.92 |
| mistral_q8_tensor_lam0.3 | 161 | 0, 0 | 31.93 | 31.9 | 0.22% | 4.39 | 31.50 | 31.89 | 31.93 | 31.93 |
| smollm_q4_g128_lam0 | 121 | 0, 0 | 31.92 | 63.8 | 0.24% | 7.23 | 31.45 | 26.47 | 30.88 | 31.89 |
| smollm_q4_channel_lam0.1 | 121 | 0, 0 | 31.92 | 63.8 | 0.23% | 6.32 | 31.47 | 30.16 | 31.92 | 31.92 |
| smollm_q8_tensor_lam0.3 | 121 | 0, 0 | 31.93 | 31.9 | 0.22% | 5.06 | 31.49 | 31.62 | 31.90 | 31.92 |
| smollm_kq_Q5_K_M_g1_lam0 | 145 | 0, 0 | 31.93 | 31.9 | 0.22% | 4.88 | 31.49 | 31.93 | 31.93 | 31.92 |
| phi3_kq_IQ4_XS_g0_lam0_c1.25 | 161 | 0, 0 | 31.92 | 62.2 | 0.24% | 7.40 | 31.45 | 25.80 | 29.88 | 31.86 |

## Co-simulation (Verilator, hw/huff/rtl) against the C model

- lookup table, buffer 64, FIFO 4, port 8, latency 8: 29 / 29 matrices bit-exact and cycle-exact (huf4p, huf8)
- lookup comparator, buffer 42, FIFO 2, port 8, latency 1: 29 / 29 matrices bit-exact and cycle-exact (huf4p, huf8)
- lookup comparator, buffer 48, FIFO 8, port 8, latency 32: 29 / 29 matrices bit-exact and cycle-exact (huf4p, huf8)
- lookup comparator, buffer 64, FIFO 4, port 6, latency 8: 29 / 29 matrices bit-exact and cycle-exact (huf4p, huf8)
- lookup comparator, buffer 64, FIFO 4, port 8, latency 8: 29 / 29 matrices bit-exact and cycle-exact (huf4p, huf8)

## Post-synthesis (ASAP7 RVT TT at 1 GHz; Nangate45 typical as a sanity check; no place and route)

| block | params | corner, clock | flow | area (um^2) | flops | reg-to-reg slack (ps) | WNS incl. I/O (ps) | power, default activity (mW) | power, co-sim activity (mW) |
|---|---|---|---|---|---|---|---|---|---|
| huff_lane | BUF=64 FIFO=4 LOOKUP=1 | ASAP7 RVT TT, 1000 MHz | yosys 0.38 + abc (map -D 600), OpenSTA 2.5 | 1566 | 2365 | 117 | 0 | 4.94 | 3.59 (smollm_iq4_l0o) |
| huff_lane | BUF=64 FIFO=4 LOOKUP=0 | ASAP7 RVT TT, 1000 MHz | yosys 0.38 + abc, OpenSTA 2.5 | 14076 | 24893 | -31318 | -31318 | 94.40 | - |
| huff_arb | LANES=32 PORT=8 | ASAP7 RVT TT, 1000 MHz | yosys 0.38 + abc (map -D 350), OpenROAD repair_design / repair_timing (no placement), OpenSTA 2.5 | 1443 | 32 | 669 | 0 | 0.83 | - |
| rans_lane | A=256 BUF=64 FIFO=4 | ASAP7 RVT TT, 1000 MHz | yosys 0.38 + abc (map -D 600), OpenSTA 2.5 | 2604 | 352 | -2654 | -2672 | 9.69 | - |
| rans_lane | A=16 BUF=64 FIFO=4 | ASAP7 RVT TT, 1000 MHz | yosys 0.38 + abc (map -D 600), OpenSTA 2.5 | 705 | 352 | -506 | -506 | 8.85 | - |
| rans_lane | A=256 BUF=64 FIFO=4 | ASAP7 RVT TT, 270 MHz | yosys 0.38 + abc (map -D 2220), OpenSTA 2.5 | 1692 | 352 | 427 | 0 | 1.29 | - |
| rans_lane | A=16 BUF=64 FIFO=4 | ASAP7 RVT TT, 645 MHz | yosys 0.38 + abc (map -D 930), OpenSTA 2.5 | 580 | 352 | 110 | 0 | 5.02 | - |
| huff_lane | BUF=64 FIFO=4 LOOKUP=1 | Nangate45 typical, 500 MHz | yosys 0.38 + abc (map -D 1200), OpenSTA 2.5 | 23462 | 2365 | 500 | 0 | 25.20 | - |
| huff_arb | LANES=32 PORT=8 | Nangate45 typical, 500 MHz | yosys 0.38 + abc (map -D 1200), OpenSTA 2.5 | 18639 | 32 | 1500 | 0 | 4.25 | - |

Engine (comparator lookup, PORT 8) = 32 x lane_cmp + arb8_d350: 51550 um^2 = 0.0515 mm^2 (blocks synthesized separately; both meet 1 GHz: lane WNS 0 ps, arbiter WNS 0 ps incl. 50 ps I/O budgets).

## Hardware per headline point (Orin Nano Super, beta = 102 GB/s)

| model | budget | b_c (b/w) | theta* (G weights/s) | engines | area (mm^2) |
|---|---|---|---|---|---|
| phi3 | <=1% | 4.27 | 191 | 4 | 0.206 |
| phi3 | <=3.5% | 3.79 | 215 | 4 | 0.206 |
| mistral | <=1% | 4.02 | 203 | 4 | 0.206 |
| mistral | <=3.5% | 3.56 | 229 | 4 | 0.206 |
| smollm | <=1% | 5.48 | 149 | 5 | 0.258 |
| smollm | <=3.5% | 4.11 | 199 | 4 | 0.206 |
