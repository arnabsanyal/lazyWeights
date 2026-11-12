# A5 energy model: the hardware decode unit on the Jetsons

Decode engine power (ASAP7, 1 GHz, pre-layout): 32 lanes x 3.59 mW (co-simulation activity) + arbiter 0.83 mW (default activity) = 115.7 mW per engine.

E_hw = E_twin (measured) - e_dram x bytes saved + E_dec, per weight pass x passes per token (1 for plain greedy; prompt-lookup speculation, draft 7, no drafter model: measured passes of our runtime). t_hw = t_twin (conservative); on the Orin also the bandwidth-bound end t_twin x S_coded / S_fixed (the GPU reads the coded bytes from DRAM). E_dec: clock-gated (engines on while decoding) / always-on (all engines for the whole token); last column: always-on with decoder power x2. Baseline = llama.cpp comparator in the same mode (CUDA on the Orin, CPU on the Nano).

## Orin Nano Super

DRAM read energy at the module input (VDD_IN, kernels/membw_energy.cu): slope 106.3 pJ/B (1 GiB buffer) - 15.8 pJ/B (L2-resident) = **90.5 pJ/byte (11.31 pJ/bit)**; idle 6.65 W; fit residual max 0.06 / 0.15 W.

### Orin Nano Super, plain greedy

| model | budget | passes / token | llama.cpp baseline ms / J | ours software ms / J | twin = ours + hardware ms | bytes saved / token | DRAM energy saved | E_dec gated / always-on | ours + hardware J | time vs baseline [bandwidth-bound, conservative] | energy vs baseline | energy (always-on x2) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| phi3 | <=1% | 1.00 | 35.0 / 0.691 | 77.7 / 1.344 | 31.4 | 113 MB | 10.3 mJ | 6.9 / 14.5 mJ | 0.627 | -15% .. -10% | -9% | -6% |
| phi3 | <=3.5% | 1.00 | 44.0 / 0.840 | 77.2 / 1.340 | 31.4 | 241 MB | 21.8 mJ | 6.9 / 14.5 mJ | 0.616 | -37% .. -29% | -27% | -24% |
| mistral | <=1% | 1.00 | 83.3 / 1.784 | 148.4 / 2.572 | 53.0 | 260 MB | 23.5 mJ | 13.4 / 24.5 mJ | 1.107 | -41% .. -36% | -38% | -36% |
| mistral | <=3.5% | 1.00 | 77.7 / 1.611 | 144.0 / 2.468 | 52.9 | 672 MB | 60.8 mJ | 13.4 / 24.5 mJ | 1.068 | -44% .. -32% | -34% | -32% |
| smollm | <=1% | 1.00 | 26.4 / 0.481 | 60.3 / 0.956 | 18.6 | 67 MB | 6.0 mJ | 6.2 / 10.8 mJ | 0.356 | -33% .. -29% | -26% | -23% |
| smollm | <=3.5% | 1.00 | 23.7 / 0.375 | 45.0 / 0.706 | 15.3 | 60 MB | 5.4 mJ | 3.3 / 7.1 mJ | 0.287 | -40% .. -36% | -23% | -21% |

### Orin Nano Super, prompt-lookup speculation

| model | budget | passes / token | llama.cpp baseline ms / J | ours software ms / J | twin = ours + hardware ms | bytes saved / token | DRAM energy saved | E_dec gated / always-on | ours + hardware J | time vs baseline [bandwidth-bound, conservative] | energy vs baseline | energy (always-on x2) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| phi3 | <=1% | 0.55 | 27.4 / 0.495 | 46.4 / 0.791 | 18.3 | 62 MB | 5.6 mJ | 3.8 / 8.5 mJ | 0.368 | -37% .. -33% | -26% | -23% |
| phi3 | <=3.5% | 0.55 | 27.6 / 0.512 | 46.5 / 0.794 | 18.5 | 132 MB | 12.0 mJ | 3.8 / 8.6 mJ | 0.366 | -41% .. -33% | -29% | -26% |
| mistral | <=1% | 0.67 | 61.6 / 1.276 | 105.7 / 1.830 | 39.5 | 174 MB | 15.8 mJ | 9.0 / 18.3 mJ | 0.825 | -40% .. -36% | -35% | -33% |
| mistral | <=3.5% | 0.63 | 71.6 / 1.397 | 97.3 / 1.667 | 37.3 | 425 MB | 38.5 mJ | 8.5 / 17.3 mJ | 0.757 | -57% .. -48% | -46% | -44% |
| smollm | <=1% | 0.48 | 14.3 / 0.246 | 32.4 / 0.508 | 10.2 | 32 MB | 2.9 mJ | 3.0 / 5.9 mJ | 0.193 | -32% .. -29% | -22% | -18% |
| smollm | <=3.5% | 0.45 | 16.5 / 0.254 | 21.3 / 0.336 | 7.5 | 27 MB | 2.4 mJ | 1.5 / 3.5 mJ | 0.138 | -57% .. -55% | -46% | -44% |

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
