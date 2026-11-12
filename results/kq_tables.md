# Fixed quality budgets

dPPL vs fp16 (WikiText-2 whole test set, 512-token chunks). b/w include scales, Huffman tables, lane offsets and the embedding. llama.cpp = smallest llama.cpp point (with or without imatrix) within the budget; fixed-width = our smallest packed point; coded = our smallest Huffman-coded point; twin = the same codes as the coded point, packed (no entropy coding). Saving vs llama.cpp = 1 - coded / llama.cpp; quantizer part = 1 - twin / llama.cpp; Huffman part = 1 - coded / twin.

| model | budget | llama.cpp | fixed-width (ours) | coded (ours) | twin b/w | saving vs llama.cpp | quantizer part | Huffman part | saving vs best fixed-width |
|---|---|---|---|---|---|---|---|---|---|
| Phi-3-mini | <=1% | IQ4_XS +imatrix, 4.311 (+0.65%) | IQ4_XS GPTQ γ0 ×1.25 λ0, 4.311 (+0.11%) | IQ4_XS GPTQ γ0.25 ×1.25 λ0, 4.045 (+0.77%) | 4.311 | 6.2% | 0.0% | 6.2% | 6.2% |
| Phi-3-mini | <=2% | IQ4_XS +imatrix, 4.311 (+0.65%) | IQ4_XS GPTQ γ0 ×1.25 λ0, 4.311 (+0.11%) | IQ4_XS GPTQ γ0 ×1.25 λ0.03, 4.008 (+1.61%) | 4.311 | 7.0% | 0.0% | 7.0% | 7.0% |
| Phi-3-mini | <=3.5% | Q3_K_M +imatrix, 4.108 (+2.91%) | GPTQ 4b g128 λ0.1, 4.157 (+2.84%) | IQ4_XS GPTQ γ1 ×1.25 λ0, 3.649 (+2.97%) | 4.311 | 11.2% | -5.0% | 15.4% | 11.2% |
| Mistral-7B | <=1% | Q5_K_M, 5.668 (+0.44%) | IQ4_XS GPTQ γ1 λ0, 4.316 (+0.94%) | IQ4_XS GPTQ γ1 λ0 (RTN), 3.886 (-0.04%) | 4.316 | 31.4% | 23.9% | 10.0% | 10.0% |
| Mistral-7B | <=2% | IQ3_M +imatrix, 3.628 (+1.01%) | GPTQ 4b g128 λ0, 4.157 (+1.84%) | IQ4_XS GPTQ γ1 ×1.5 λ0, 3.422 (+1.46%) | 4.316 | 5.7% | -19.0% | 20.7% | 5.7% |
| Mistral-7B | <=3.5% | IQ3_M +imatrix, 3.628 (+1.01%) | GPTQ 4b g128 λ0, 4.157 (+1.84%) | IQ4_XS GPTQ γ1 ×1.5 λ0, 3.422 (+1.46%) | 4.316 | 5.7% | -19.0% | 20.7% | 5.7% |
| SmolLM-1.7B | <=1% | Q5_K_M, 5.721 (+0.65%) | Q5_K_M GPTQ γ0 λ0, 5.721 (+0.31%) | Q6_K GPTQ γ1 ×1.5 λ0, 5.045 (+0.95%) | 6.564 | 11.8% | -14.7% | 23.1% | 11.8% |
| SmolLM-1.7B | <=2% | Q4_K_M +imatrix, 4.927 (+1.70%) | IQ4_XS GPTQ γ0.5 λ0, 4.388 (+1.90%) | IQ4_XS GPTQ γ0.5 λ0, 4.311 (+1.90%) | 4.388 | 12.5% | 10.9% | 1.8% | 1.8% |
| SmolLM-1.7B | <=3.5% | IQ4_XS +imatrix, 4.388 (+3.09%) | IQ4_XS GPTQ γ0 λ0, 4.388 (+2.24%) | IQ4_XS GPTQ γ1 λ0, 3.934 (+3.27%) | 4.388 | 10.3% | 0.0% | 10.3% | 10.3% |

## Task accuracy of the table points

| model | point | family | b/w | dPPL | hellaswag | arc_easy | piqa | gsm8k |
|---|---|---|---|---|---|---|---|---|
| Phi-3-mini | IQ4_XS +imatrix | packed | 4.311 | +0.65% | 0.783 | 0.807 | 0.798 | 0.722 |
| Phi-3-mini | IQ4_XS GPTQ γ0.25 ×1.25 λ0 | coded | 4.045 | +0.77% | pending | pending | pending | pending |
| Phi-3-mini | IQ4_XS GPTQ γ0 ×1.25 λ0.03 | coded | 4.008 | +1.61% | pending | pending | pending | pending |
| Phi-3-mini | Q3_K_M +imatrix | packed | 4.108 | +2.91% | 0.776 | 0.786 | 0.799 | 0.722 |
| Phi-3-mini | IQ4_XS GPTQ γ1 ×1.25 λ0 | coded | 3.649 | +2.97% | 0.774 | 0.811 | 0.796 | 0.672 |
| Mistral-7B | Q5_K_M | packed | 5.668 | +0.44% | 0.842 | 0.716 | 0.768 | 0.434 |
| Mistral-7B | IQ4_XS GPTQ γ1 λ0 (RTN) | coded | 3.886 | -0.04% | pending | pending | pending | pending |
| Mistral-7B | IQ3_M +imatrix | packed | 3.628 | +1.01% | 0.838 | 0.722 | 0.779 | 0.395 |
| Mistral-7B | IQ4_XS GPTQ γ1 ×1.5 λ0 | coded | 3.422 | +1.46% | 0.838 | 0.719 | 0.782 | 0.376 |
| SmolLM-1.7B | Q5_K_M | packed | 5.721 | +0.65% | 0.607 | 0.633 | 0.737 | 0.055 |
| SmolLM-1.7B | Q6_K GPTQ γ1 ×1.5 λ0 | coded | 5.045 | +0.95% | 0.607 | 0.622 | 0.735 | 0.058 |
| SmolLM-1.7B | Q4_K_M +imatrix | packed | 4.927 | +1.70% | 0.604 | 0.629 | 0.739 | 0.042 |
| SmolLM-1.7B | IQ4_XS GPTQ γ0.5 λ0 | coded | 4.311 | +1.90% | 0.603 | 0.622 | 0.734 | 0.036 |
| SmolLM-1.7B | IQ4_XS +imatrix | packed | 4.388 | +3.09% | 0.605 | 0.625 | 0.732 | 0.042 |
| SmolLM-1.7B | IQ4_XS GPTQ γ1 λ0 | coded | 3.934 | +3.27% | 0.601 | 0.610 | 0.736 | 0.040 |

## Headline points: task accuracy

HellaSwag: lm-eval acc_norm (0-shot), all 10042. GSM8K-CoT: 8-shot chain of thought, all 1319, exact match (strict = 'The answer is N', flexible = last number); +- = stderr of the strict score.

| model | budget | point | b/w | dPPL | HellaSwag | GSM8K-CoT strict | GSM8K-CoT flexible |
|---|---|---|---|---|---|---|---|
| Phi-3-mini | - | fp16 | 16 | 0 | 0.785 | 0.791 +- 0.011 | 0.806 |
| Phi-3-mini | <=1% | ours: IQ4_XS GPTQ γ0 ×1.25 λ0 | 4.046 | +0.11% | 0.778 | 0.746 +- 0.012 | 0.771 |
| Phi-3-mini | <=1% | llama.cpp: IQ4_XS +imatrix | 4.311 | +0.65% | 0.783 | 0.774 +- 0.012 | 0.785 |
| Phi-3-mini | <=3.5% | ours: IQ4_XS GPTQ γ1 ×1.25 λ0 | 3.649 | +2.97% | 0.774 | 0.757 +- 0.012 | 0.770 |
| Phi-3-mini | <=3.5% | llama.cpp: Q3_K_M +imatrix | 4.108 | +2.91% | 0.776 | 0.772 +- 0.012 | 0.779 |
| Mistral-7B | - | fp16 | 16 | 0 | 0.844 | 0.403 +- 0.014 | 0.447 |
| Mistral-7B | <=1% | ours: IQ4_XS GPTQ γ1 λ0 | 3.886 | +0.94% | 0.840 | 0.378 +- 0.013 | 0.402 |
| Mistral-7B | <=1% | llama.cpp: Q5_K_M | 5.668 | +0.44% | 0.842 | 0.373 +- 0.013 | 0.425 |
| Mistral-7B | <=3.5% | ours: IQ4_XS GPTQ γ1 ×1.5 λ0 | 3.422 | +1.46% | 0.838 | 0.333 +- 0.013 | 0.368 |
| Mistral-7B | <=3.5% | llama.cpp: IQ3_M +imatrix | 3.628 | +1.01% | 0.838 | 0.373 +- 0.013 | 0.400 |
| SmolLM-1.7B | - | fp16 | 16 | 0 | 0.608 | 0.058 +- 0.006 | 0.066 |
| SmolLM-1.7B | <=1% | ours: Q5_K_M GPTQ γ1 λ0 | 5.162 | +0.88% | 0.607 | 0.063 +- 0.007 | 0.067 |
| SmolLM-1.7B | <=1% | llama.cpp: Q5_K_M | 5.721 | +0.65% | 0.607 | 0.058 +- 0.006 | 0.064 |
| SmolLM-1.7B | <=3.5% | ours: IQ4_XS GPTQ γ1 λ0 | 3.934 | +3.27% | 0.601 | 0.049 +- 0.006 | 0.055 |
| SmolLM-1.7B | <=3.5% | llama.cpp: IQ4_XS +imatrix | 4.388 | +3.09% | 0.605 | 0.049 +- 0.006 | 0.055 |
