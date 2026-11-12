#!/bin/bash
# Orin (MAXN_SUPER + jetson_clocks): software Huffman decode rate of the 6 headline coded bundles, llmrt_kq7 decode-bench
# (GPU and 6-thread CPU decode of every coded matrix) for the decode-wall figure -> ~/lw/a2_decode7/.
set -u
cd ~/lw_data
X=~/lw/rt/build/llmrt_kq7; O=~/lw/a2_decode7; mkdir -p $O
for b in phi3_kq_IQ4_XS_ref phi3_kq_IQ4_XS_g1_lam0_c1.25 mistral_kq_IQ4_XS_g1_lam0 mistral_kq_IQ4_XS_g1_lam0_c1.5 smollm_kq_Q5_K_M_g1_lam0 smollm_kq_IQ4_XS_g1_lam0; do
  $X rt/$b decode-bench --threads 6 > $O/decode_$b.jsonl 2> $O/decode_$b.err; echo "$(date -Is) $b rc=$?"
done
echo ORIN_DECODE_DONE
