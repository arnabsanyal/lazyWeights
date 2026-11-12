# A5 energy model: the hardware decode unit on the Jetsons

Decode engine power (ASAP7, 1 GHz, pre-layout): 32 lanes x 3.59 mW (co-simulation activity) + arbiter 0.83 mW (default activity) = 115.7 mW per engine.

E_hw = E_twin (measured) - e_dram x bytes saved + E_dec, per weight pass x passes per token (1 for plain greedy; prompt-lookup speculation, draft 7, no drafter model: measured passes of our runtime). t_hw = t_twin (conservative); on the Orin also the bandwidth-bound end t_twin x S_coded / S_fixed (the GPU reads the coded bytes from DRAM). E_dec: clock-gated (engines on while decoding) / always-on (all engines for the whole token); last column: always-on with decoder power x2. Baseline = llama.cpp comparator in the same mode (CUDA on the Orin, CPU on the Nano).

## Orin Nano Super

DRAM read energy at the module input (VDD_IN, kernels/membw_energy.cu): slope 106.3 pJ/B (1 GiB buffer) - 15.8 pJ/B (L2-resident) = **90.5 pJ/byte (11.31 pJ/bit)**; idle 6.65 W; fit residual max 0.06 / 0.15 W.

### Orin Nano Super, plain greedy

| model | budget | passes / token | llama.cpp baseline ms / J | ours software ms / J | twin = ours + hardware ms | bytes saved / token | DRAM energy saved | E_dec gated / always-on | ours + hardware J | time vs baseline [bandwidth-bound, conservative] | energy vs baseline | energy (always-on x2) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| phi3 | <=1% | 1.00 | 35.0 / 0.691 | 77.7 / 1.344 | 31.3 | 113 MB | 10.3 mJ | 6.9 / 14.5 mJ | 0.626 | -16% .. -10% | -9% | -6% |
| phi3 | <=3.5% | 1.00 | 44.0 / 0.840 | 77.2 / 1.339 | 31.4 | 241 MB | 21.8 mJ | 6.9 / 14.6 mJ | 0.616 | -37% .. -29% | -27% | -24% |
| mistral | <=1% | 1.00 | 83.3 / 1.784 | 148.4 / 2.570 | 52.9 | 260 MB | 23.5 mJ | 13.4 / 24.5 mJ | 1.106 | -41% .. -36% | -38% | -36% |
| mistral | <=3.5% | 1.00 | 77.7 / 1.611 | 143.9 / 2.469 | 53.0 | 672 MB | 60.8 mJ | 13.4 / 24.5 mJ | 1.068 | -44% .. -32% | -34% | -32% |
| smollm | <=1% | 1.00 | 26.4 / 0.481 | 60.2 / 0.955 | 18.6 | 67 MB | 6.0 mJ | 6.2 / 10.8 mJ | 0.356 | -33% .. -29% | -26% | -23% |
| smollm | <=3.5% | 1.00 | 23.7 / 0.375 | 45.0 / 0.706 | 15.2 | 60 MB | 5.4 mJ | 3.3 / 7.1 mJ | 0.287 | -40% .. -36% | -24% | -21% |

### Orin Nano Super, prompt-lookup speculation

| model | budget | passes / token | llama.cpp baseline ms / J | ours software ms / J | twin = ours + hardware ms | bytes saved / token | DRAM energy saved | E_dec gated / always-on | ours + hardware J | time vs baseline [bandwidth-bound, conservative] | energy vs baseline | energy (always-on x2) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| phi3 | <=1% | 0.48 | 27.4 / 0.495 | 46.1 / 0.771 | 17.9 | 54 MB | 4.9 mJ | 3.3 / 8.3 mJ | 0.366 | -39% .. -35% | -26% | -23% |
| phi3 | <=3.5% | 0.48 | 27.6 / 0.512 | 46.7 / 0.783 | 18.3 | 115 MB | 10.4 mJ | 3.3 / 8.5 mJ | 0.370 | -42% .. -34% | -28% | -25% |
| mistral | <=1% | 0.58 | 61.6 / 1.276 | 103.6 / 1.786 | 41.5 | 151 MB | 13.6 mJ | 7.7 / 19.2 mJ | 0.871 | -37% .. -33% | -32% | -29% |
| mistral | <=3.5% | 0.55 | 71.6 / 1.397 | 94.4 / 1.613 | 38.5 | 367 MB | 33.2 mJ | 7.3 / 17.8 mJ | 0.786 | -56% .. -46% | -44% | -42% |
| smollm | <=1% | 0.43 | 14.3 / 0.246 | 31.5 / 0.491 | 10.1 | 28 MB | 2.6 mJ | 2.6 / 5.9 mJ | 0.194 | -33% .. -29% | -21% | -17% |
| smollm | <=3.5% | 0.41 | 16.5 / 0.254 | 20.7 / 0.330 | 7.6 | 25 MB | 2.2 mJ | 1.4 / 3.5 mJ | 0.143 | -57% .. -54% | -44% | -41% |

## Jetson Nano

DRAM read energy at the module input (POM_5V_IN, kernels/membw_energy.cu): slope 57.2 pJ/B (1 GiB buffer) - 22.3 pJ/B (L2-resident) = **34.9 pJ/byte (4.36 pJ/bit)**; idle 1.00 W; fit residual max 0.46 / 0.09 W.

### Jetson Nano, plain greedy

| model | budget | passes / token | llama.cpp baseline ms / J | ours software ms / J | twin = ours + hardware ms | bytes saved / token | DRAM energy saved | E_dec gated / always-on | ours + hardware J | time vs baseline [bandwidth-bound, conservative] | energy vs baseline | energy (always-on x2) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| phi3 | <=3.5% | 1.00 | 735.6 / 4.799 | 579.6 / 4.425 | 543.3 | 241 MB | 8.4 mJ | 6.9 / 62.9 mJ | 3.409 | -26% .. -26% | -29% | -26% |
| smollm | <=3.5% | 1.00 | 317.3 / 2.123 | 273.1 / 2.007 | 256.7 | 60 MB | 2.1 mJ | 3.3 / 29.7 mJ | 1.560 | -19% .. -19% | -27% | -24% |

### Jetson Nano, prompt-lookup speculation

| model | budget | passes / token | llama.cpp baseline ms / J | ours software ms / J | twin = ours + hardware ms | bytes saved / token | DRAM energy saved | E_dec gated / always-on | ours + hardware J | time vs baseline [bandwidth-bound, conservative] | energy vs baseline | energy (always-on x2) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| phi3 | <=3.5% | 0.48 | 955.9 / 6.228 | 451.7 / 3.251 | 500.4 | 117 MB | 4.1 mJ | 3.4 / 57.9 mJ | 2.993 | -48% .. -48% | -52% | -50% |
| smollm | <=3.5% | 0.41 | 419.3 / 2.729 | 202.8 / 1.318 | 217.2 | 25 MB | 0.9 mJ | 1.4 / 25.1 mJ | 1.226 | -48% .. -48% | -55% | -53% |
