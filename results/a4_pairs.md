# Pair symbols on/off (huf4p vs one 4-bit symbol per weight), 4-bit headline bundles

Size over every huf4p matrix; rate from the C model (comparator lookup, buffer 64, FIFO 4, port 8, latency 8) on the first, middle and last layer's huf4p matrices.

| bundle | matrices | b/w pairs | b/w singles | singles size | sample | w/cycle pairs | w/cycle singles |
|---|---|---|---|---|---|---|---|
| phi3_kq_IQ4_XS_g0_lam0_c1.25 | 160 | 3.728 | 3.758 | +0.8% | 15 | 63.84 | 31.93 |
| phi3_kq_IQ4_XS_g1_lam0_c1.25 | 160 | 3.458 | 3.452 | -0.2% | 15 | 63.85 | 31.93 |
| mistral_kq_IQ4_XS_g1_lam0 | 192 | 3.679 | 3.691 | +0.3% | 18 | 63.84 | 31.93 |
| mistral_kq_IQ4_XS_g1_lam0_c1.5 | 192 | 3.220 | 3.234 | +0.4% | 18 | 63.85 | 31.93 |
| smollm_kq_IQ4_XS_g1_lam0 | 120 | 3.704 | 3.721 | +0.5% | 15 | 63.84 | 31.93 |
