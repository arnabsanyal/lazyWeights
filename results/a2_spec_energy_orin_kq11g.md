# Decode speed and energy, orin (our runtime: kq11g)

Per point: ms per generated token / J per token (module input rail, idle included), plain greedy and with prompt-lookup speculation (draft 7), over the 6 prompts (128 new tokens each, stops at EOS). Ours = kernels/llmrt.cu on the Huffman-coded bundle; twin = the same codes packed (no entropy coding); llama.cpp = llama-lookup (CUDA) on the same GGUF and on the llama.cpp comparator.

| model | budget | ours plain | ours spec | twin plain | twin spec | llama.cpp same GGUF plain | spec | llama.cpp comparator plain | spec |
|---|---|---|---|---|---|---|---|---|---|
| phi3 | <=1% | 78.4 / 1.36 | 48.8 / 0.83 | 31.4 / 0.63 | 19.1 / 0.39 | 35.0 / 0.69 | 27.4 / 0.49 | 35.0 / 0.69 | 27.4 / 0.49 |
| phi3 | <=3.5% | 77.2 / 1.34 | 46.5 / 0.79 | 31.4 / 0.63 | 18.5 / 0.37 | 35.0 / 0.69 | 27.5 / 0.49 | 44.0 / 0.84 | 27.6 / 0.51 |
| mistral | <=1% | 148.4 / 2.57 | 105.7 / 1.83 | 53.0 / 1.12 | 39.5 / 0.83 | 58.9 / 1.29 | 54.4 / 1.11 | 83.3 / 1.78 | 61.6 / 1.28 |
| mistral | <=3.5% | 144.0 / 2.47 | 97.3 / 1.67 | 52.9 / 1.12 | 37.3 / 0.79 | 57.8 / 1.27 | 50.7 / 1.03 | 77.7 / 1.61 | 71.6 / 1.40 |
| smollm | <=1% | 60.3 / 0.96 | 32.4 / 0.51 | 18.6 / 0.36 | 10.2 / 0.19 | - | - | 26.4 / 0.48 | 14.3 / 0.25 |
| smollm | <=3.5% | 45.0 / 0.71 | 21.3 / 0.34 | 15.3 / 0.29 | 7.5 / 0.14 | 23.7 / 0.38 | 13.6 / 0.21 | 23.7 / 0.38 | 16.5 / 0.25 |
