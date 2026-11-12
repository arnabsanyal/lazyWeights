# Run-to-run variance of the Orin end-to-end table (5 repeats)

Per point and mode: coefficient of variation (sd / mean, %) of ms and mJ per token over the repeats, and the range of Delta t (bandwidth-bound / conservative) and Delta E vs the llama.cpp baseline; [table] = the paper's values (single run). Clocks: MAXN_SUPER + jetson_clocks.

| model | budget | mode | n | baseline ms CV | baseline mJ CV | twin ms CV | twin mJ CV | Delta t bw [table] | Delta t cons [table] | Delta E [table] |
|---|---|---|---|---|---|---|---|---|---|---|
| phi3 | <=1% | plain | 5 | 0.1 | 0.1 | 0.2 | 0.6 | -16..-15% [-16%] | -11..-10% [-10%] | -11..-10% [-9%] |
| phi3 | <=1% | spec | 5 | 0.1 | 0.1 | 0.2 | 0.5 | -39..-38% [-39%] | -35..-35% [-35%] | -27..-26% [-26%] |
| phi3 | <=3.5% | plain | 5 | 0.0 | 0.1 | 0.1 | 0.4 | -37..-37% [-37%] | -29..-29% [-29%] | -28..-27% [-27%] |
| phi3 | <=3.5% | spec | 5 | 0.0 | 0.1 | 0.1 | 0.3 | -42..-42% [-42%] | -34..-34% [-34%] | -28..-28% [-28%] |
| mistral | <=1% | plain | 5 | 0.1 | 0.1 | 0.1 | 0.2 | -41..-41% [-41%] | -37..-36% [-36%] | -38..-38% [-38%] |
| mistral | <=1% | spec | 5 | 0.1 | 0.1 | 0.5 | 0.4 | -38..-37% [-37%] | -33..-32% [-33%] | -32..-32% [-32%] |
| mistral | <=3.5% | plain | 5 | 0.0 | 0.1 | 0.1 | 0.1 | -44..-44% [-44%] | -32..-32% [-32%] | -34..-34% [-34%] |
| mistral | <=3.5% | spec | 5 | 0.1 | 0.1 | 0.3 | 0.2 | -55..-55% [-56%] | -46..-46% [-46%] | -44..-43% [-44%] |
| smollm | <=1% | plain | 5 | 0.0 | 0.1 | 0.1 | 0.2 | -33..-33% [-33%] | -29..-29% [-29%] | -26..-26% [-26%] |
| smollm | <=1% | spec | 5 | 0.0 | 0.1 | 0.0 | 0.1 | -33..-33% [-33%] | -29..-29% [-29%] | -21..-21% [-21%] |
| smollm | <=3.5% | plain | 5 | 0.0 | 0.0 | 0.2 | 0.1 | -40..-40% [-40%] | -36..-35% [-36%] | -24..-24% [-24%] |
| smollm | <=3.5% | spec | 5 | 0.0 | 0.1 | 0.0 | 0.1 | -57..-57% [-57%] | -54..-54% [-54%] | -44..-43% [-44%] |

Max CV over all cells: 0.6%; median 0.1%. Largest |Delta - table| over all repeats: 1.7 points.
