# Decode speed and energy, nano (our runtime: usb10)

Per point: ms per generated token / J per token (module input rail, idle included), plain greedy and with prompt-lookup speculation (draft 7), over the 6 prompts (128 new tokens each, stops at EOS). Ours = kernels/llmrt.cu on the Huffman-coded bundle; twin = the same codes packed (no entropy coding); llama.cpp = llama-lookup (CUDA) on the same GGUF and on the llama.cpp comparator.

| model | budget | ours plain | ours spec | twin plain | twin spec | llama.cpp same GGUF plain | spec | llama.cpp comparator plain | spec |
|---|---|---|---|---|---|---|---|---|---|
| mistral | <=1% | 3400.6 / 15.83 | 2296.2 / 11.27 | - | - | 17967.3 / 58.25 | 13309.7 / 46.14 | 22590.9 / 71.21 | 16661.8 / 56.56 |
| mistral | <=3.5% | 2077.0 / 12.03 | 1492.3 / 8.85 | - | - | 15052.2 / 49.03 | 11532.9 / 40.21 | 1928.0 / 12.66 | 2900.1 / 19.10 |
