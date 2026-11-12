# A5: per-token decode time with the hardware decoder (Orin Nano Super)

Our runtime rows: results/a2_orin/bench11 (int8 activations in the decode GEMV).

- diagnostic, phi3: bytes-only fit t = 5.22 ms + S / 62.4 GB/s over 8 llama.cpp GGUFs, leave-one-out error max 47% (time depends on the quant type)
- diagnostic, mistral: bytes-only fit t = 37.40 ms + S / 127.4 GB/s over 7 llama.cpp GGUFs, leave-one-out error max 37% (time depends on the quant type)
- diagnostic, smollm: bytes-only fit t = 15.28 ms + S / 125.1 GB/s over 9 llama.cpp GGUFs, leave-one-out error max 12% (time depends on the quant type)

llama.cpp kernels: measured per-token time of the comparator and of the same GGUF without entropy coding (DRAM use of the latter: S_fixed / t / 102 GB/s); with the hardware decoder, bounds [bandwidth-bound, compute-bound]. Engines: smallest number of 32-lane engines at 1 GHz reaching theta* = 8 beta / b_c (A4 C simulation rate of the comparator-lookup engine on the bundle's matrices).

| model | budget | llama.cpp comparator | same GGUF fixed-width (DRAM use) | + hardware decoder | vs comparator | engines (theta*) | ours: software coded / twin = hardware | ours + hardware vs comparator |
|---|---|---|---|---|---|---|---|---|
| phi3 | <=1% | 33.3 ms | 33.3 ms (59%) | 31.4 - 33.3 ms | -6% to +0% | 4 (201 G/s) | 74.4 / 28.1 ms | -16% |
| phi3 | <=3.5% | 42.2 ms | 33.3 ms (59%) | 29.3 - 33.3 ms | -31% to -21% | 4 (215 G/s) | 73.8 / 28.1 ms | -34% |
| mistral | <=1% | 83.2 ms | 59.0 ms (64%) | 55.0 - 59.0 ms | -34% to -29% | 4 (203 G/s) | 144.6 / 49.2 ms | -41% |
| mistral | <=3.5% | 77.3 ms | 58.9 ms (64%) | 48.6 - 58.9 ms | -37% to -24% | 4 (229 G/s) | 140.3 / 49.3 ms | -36% |
| smollm | <=1% | 25.8 ms | 25.8 ms (47%) | 24.4 - 25.8 ms | -5% to -0% | 5 (151 G/s) | 59.0 / 17.4 ms | -33% |
| smollm | <=3.5% | 23.1 ms | 23.1 ms (40%) | 21.6 - 23.1 ms | -6% to -0% | 4 (199 G/s) | 43.8 / 14.0 ms | -39% |

Our runtime: software Huffman decode costs 13.29 ns per 1000 coded weights per token on top of the packed twin (fit over 6 points; prediction error of the coded rows max 32.0%). The hardware decoder removes it: t_hw = t_twin.

## Jetson Nano (MAXN; measured read bandwidth 22.5 GB/s)

Our runtime on the Nano is compute-bound (fp32 activations, no dp4a on sm_53): the twin reads its weights at a small fraction of the bandwidth, so t_hw = t_twin and a single engine covers theta*. llama.cpp on the Nano runs on the CPU only (CUDA 10.2 cannot build its CUDA backend): a different class of baseline.

| model | budget | ours: software coded / twin = hardware | twin DRAM use | engines (theta*) | llama.cpp CPU comparator | ours + hardware vs CPU comparator |
|---|---|---|---|---|---|---|
| phi3 | <=3.5% | 559.3 / 540.0 ms | 17% | 1 (47 G/s) | 735.6 ms | -27% |
| smollm | <=3.5% | 272.0 / 253.2 ms | 16% | 1 (44 G/s) | 317.3 ms | -20% |

Software Huffman decode on the Nano: 6.20 ns per 1000 coded weights per token on top of the twin (fit over 2 points; prediction error max 3.0%).
