# ZCU102 prototype (XCZU9EG-2, Vivado 2025.2), 2026-11-10

Design: hw/huff/fpga/zcu_top.sv + zcu_core.sv: 4 Huffman engines (32 lanes, comparator lookup, BUF 64, FIFO 4,
8 ports; payload in BRAM, 4 copies x 2 ports, words back 8 cycles after grant), each with per-lane CRC-32 and symbol
counters; clock = Si570 300 MHz -> MMCM -> 100 MHz. Host: the board's Cortex-A53 (Ubuntu 22.04) over
M_AXI_HPM0_LPD -> AXI4-Lite (scripts/zcu_host.py). Jobs: the 29 co-simulation matrices, each cut into chunks of
<= 16 tiles / 32K payload words (hw/huff/tb/chunks_main.cpp, scripts/zcu_jobs.py); each chunk is written into BRAM,
the 4 engines start in the same cycle, and every lane's CRC-32 / symbol count and every chunk's cycles / stalls /
words are compared with the C model.

## Implementation (100 MHz build, build_zcu.tcl 4 4 12 f100)

| | total | per engine | device % |
|---|---|---|---|
| LUTs | 189,632 | 47.2k (3.3k LUTRAM) | 69.2 |
| FFs | 36,942 | 9.0k | 6.7 |
| BRAM36 | 464 | 116 | 50.9 |
| DSP | 0 | 0 | 0 |

WNS +0.039 ns, WHS +0.010 ns at 100 MHz (Performance_ExplorePostRoutePhysOpt).
Vivado vectorless power: 6.09 W on-chip (PS8 2.73 W), VCCINT 3.17 W.

## On the board (fabric clock measured 100.00 MHz)

- 29 / 29 matrices, 2,723 chunks, 484,442,112 symbols: **0 mismatches**; cycles, stalls and words equal to the C
  model on every chunk.
- 31.71-31.85 symbols / cycle / engine (31.79 overall) -> **12.7 G weights/s** for 4 engines at 100 MHz
  (12.4 G/s counting the slowest engine of each group of 4 chunks).
- Power (INA226, 20 s, smollm lm_head, 253,631 back-to-back runs, engine duty 78.4%; host loop between runs):
  PL VCCINT (u79) 0.79 W idle (bitstream loaded) -> 3.86 W busy: +3.08 W, i.e. ~3.9 W dynamic at full duty,
  0.31 nJ per weight. VCCBRAM (u81) 0.025 W and VCCAUX (u80) 0.387 W unchanged (INA226 LSB 25 mW).

Host note: numpy block writes to /dev/mem must be multiples of 16 B (glibc's aarch64 memcpy ends with an
overlapping unaligned 16 B store -> SIGBUS on Device memory); zcu_host.wblk writes the tail per word.

Raw: results/logs/a4/zcu/ (results_zcu_f100.jsonl, power_f100.json, f100_*.rpt).

## 125 MHz attempt (build_zcu.tcl 4 5 12 f125: VCO 1500 / 12)

Fails timing: WNS -2.159 ns, 19,640 failing endpoints. Worst path 9.90 ns, 23 logic levels, 77% routing delay (the
100 MHz build's worst path: 9.42 ns, 23 levels). The design tops out near 100 MHz without extra pipelining, so the
paper reports 100 MHz. Report: results/logs/a4/zcu/f125_timing.rpt. (A first try with DIV 9.6 was dropped: the MMCM
divider steps by 0.125, so it would have run at 124.7 MHz.)

## Address register (AREG) and faster clocks

zcu_core.sv `AREG=1` (default since this change): each port's BRAM read address is registered before the payload
RAMs and one response stage is dropped, so words still come back 8 cycles after grant (LAT 8, cycle-exact with the C
model; Verilator: 4 matrices / 77 chunks, 0 bad). It removes the long route to the RAMs; the remaining worst path is
the engine's own loop (limit_reg -> lane decode -> arbiter -> address register / word counter), ~8.7-9.0 ns.

| build (VCO / DIV) | clock | WNS | failing endpoints | worst path | LUTs |
|---|---|---|---|---|---|
| f125r (1500 / 12) | 125 MHz | -0.714 ns | 3,334 | 8.78 ns, 29 levels (-> wrd_reg) | ~184k |
| f120r (1200 / 10) | 120 MHz | -0.523 ns | 2,571 | 8.66 ns, 23 levels, 75% route (-> qa_reg) | ~184k |
| **f112r (1125 / 10)** | **112.5 MHz** | **+0.058 ns** (WHS +0.010) | 0 | 8.99 ns, 28 levels (-> wrd_reg) | 183,005 |

f112r: 183,005 LUTs (66.8%), 37,178 FFs (6.8%), 464 BRAM36 (50.9%), 0 DSP. Vivado vectorless: 6.19 W on-chip,
VCCINT 3.29 W.

On the board (fabric clock measured 112.50 MHz):
- 29 / 29 matrices, 2,723 chunks, 484,442,112 symbols: **0 mismatches**; cycles, stalls and words equal to the C model.
- 31.79 symbols / cycle / engine (cycle counts identical to the 100 MHz build) -> **14.3 G weights/s** for 4 engines
  (14.0 G/s counting the slowest engine of each group).
- Power (same protocol as above, 3 runs of 20 s): VCCINT 0.88 W idle -> 3.63-3.74 W busy at 77-78% duty, i.e.
  **3.60-3.68 W dynamic at full duty, 0.25-0.26 nJ per weight**. Lower than the 100 MHz build's 3.9 W despite the higher
  clock: a different netlist (AREG, 6.6k fewer LUTs, shorter RAM routes); idle is 0.09 W higher (likely leakage; die
  temperature was not logged in the 100 MHz session). Vivado's vectorless estimate goes the other way (+0.13 W VCCINT).

The ceiling (~115 MHz) is the engine's decode -> arbiter recurrence; pipelining cannot cut it without changing
the cycle behaviour (see the notes on speculative lookahead / more lanes).

Raw: results/logs/a4/zcu/{results_zcu_f112r.jsonl, power_f112r.json, f112r_*.rpt, f120r_timing.rpt, f125r_timing.rpt}.
