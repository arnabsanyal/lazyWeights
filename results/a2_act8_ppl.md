# Decode-path PPL: int8 vs fp32 activations

WikiText-2, independent 512-token chunks fed through the KV cache in steps of 8 (decode / verify GEMV; the paper's PPL protocol uses the prefill GEMM, which is unchanged). Packed twins (coded bundles give identical logits).

| bundle | step | fp32 | int8 | int8 vs fp32 |
|---|---|---|---|---|
| mistral_kq_IQ4_XS_g1_lam0_c1.5_pk | 8 | 8.1424 | 8.1527 | +0.13% |
| mistral_kq_IQ4_XS_g1_lam0_pk | 8 | 8.1013 | 8.1157 | +0.18% |
| phi3_kq_IQ4_XS_g1_lam0_c1.25_pk | 8 | 9.4704 | 9.4686 | -0.02% |
| phi3_kq_IQ4_XS_ref_pk | 8 | 9.2561 | 9.2497 | -0.07% |
| smollm_kq_IQ4_XS_g1_lam0_pk | 8 | 18.5640 | 18.5789 | +0.08% |
| smollm_kq_Q6_K_g1_lam0_c1.5_pk | 8 | 18.1468 | 18.1601 | +0.07% |
