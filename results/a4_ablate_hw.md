# Hardware ablations (ASAP7 RVT TT, 1 GHz, pre-layout; results/logs/a4/ablate_syn.sh, reports in results/logs/a4/syn)

## FIFO for a 1024-cycle read latency (latency/4 = 256 words per lane)
- Flip-flop FIFO (lane_cmp_f256): lane 7,290 um^2 vs 1,566 um^2 with FIFO 4, and fails 1 GHz by 13.6 ns (WNS -13643 ps): not viable.
- SRAM macro estimate: ASAP7 fakeram7_256x32 is 8.36 x 42.0 um = 351 um^2; a 40-bit entry (32 data + 8 meta),
  scaled by width = 439 um^2 per lane, x32 lanes = 0.0140 mm^2 per engine = +27% over the 0.0516 mm^2 engine.

## Lanes per engine (ports scale with lanes, 4 lanes per port; ABC_D=350, OpenROAD repair)
| lanes | ports | arbiter area (um^2) | WNS at 1 GHz (ps) | worst path |
|---|---|---|---|---|
| 32 | 8 | 1,443 | +6.8 (met) | flop -> req_valid |
| 64 | 16 | 9,174 (6.4x) | -292 (~1.29 ns, ~775 MHz) | flop -> req_valid |
| 128 | 32 | -- | -- | stopped: Yosys still in its first PROC pass after 8 h 50 min at 20 GB |
