#!/bin/bash
# Orin (MAXN_SUPER + jetson_clocks): the headline points with llmrt_kq11g = kernels/llmrt.cu built with
# --default-stream per-thread, LLMRT_GRAPHS=1 (decode / verify passes replay CUDA graphs; graph-check: bit-exact).
# 1) graph-check on every bundle -> ~/lw/a2_graph/check/; 2) fwd-bench m = 1..8 at 128, graphs off and on
# -> ~/lw/a2_bench11/; 3) spec under tegrastats, llama.cpp's drafter -> ~/lw_data/energy_kq11g/<m>/ and ours
# -> ~/lw_data/energy_kq11glk/<m>/. llama.cpp rows unchanged (it already uses CUDA graphs).
set -u
cd ~/lw_data
X=~/lw/rt/build/llmrt_kq11g; O=~/lw/a2_bench11; C=~/lw/a2_graph/check; mkdir -p $O $C
P=(phi3_kq_IQ4_XS_ref phi3_kq_IQ4_XS_g1_lam0_c1.25)
G=(phi3_kq_IQ4_XS_g0_lam0_c1.25)
M=(mistral_kq_IQ4_XS_g1_lam0 mistral_kq_IQ4_XS_g1_lam0_c1.5)
S=(smollm_kq_Q5_K_M_g1_lam0 smollm_kq_IQ4_XS_g1_lam0)
for b in ${G[@]} ${S[@]} ${P[@]} ${M[@]}; do for v in "" _pk; do
  [ -s $C/$b$v.jsonl ] || LLMRT_GRAPHS=1 $X rt/$b$v graph-check --prompt-len 128 --ctx 1024 > $C/$b$v.jsonl 2> $C/$b$v.err
  echo "$(date -Is) check $b$v rc=$? $(tail -1 $C/$b$v.jsonl)"
  [ -s $O/fwd_$b${v}_nog.jsonl ] || LLMRT_GRAPHS=0 $X rt/$b$v fwd-bench --prompt-len 128 --reps 5 > $O/fwd_$b${v}_nog.jsonl 2> $O/fwd_$b${v}_nog.err
  [ -s $O/fwd_$b$v.jsonl ] || LLMRT_GRAPHS=1 $X rt/$b$v fwd-bench --prompt-len 128 --reps 7 > $O/fwd_$b$v.jsonl 2> $O/fwd_$b$v.err
done; done
echo "$(date -Is) fwd done"
twins() { for b in "$@"; do echo rt/$b rt/${b}_pk; done; }
export LLMRT_GRAPHS=1
for dr in ngramcache lookup; do
  E=~/lw_data/energy_kq11g; [ $dr = lookup ] && E=~/lw_data/energy_kq11glk
  export LLMRT_ARGS="--drafter $dr"
  [ -f $E/phi3g0/done ]  || ~/lw/scripts/device/energy_run.sh $X $E/phi3g0  1024 spec_prompts.txt $(twins ${G[@]})
  [ -f $E/smollm/done ]  || ~/lw/scripts/device/energy_run.sh $X $E/smollm  1024 smollm.txt       $(twins ${S[@]})
  [ -f $E/phi3/done ]    || ~/lw/scripts/device/energy_run.sh $X $E/phi3    1024 spec_prompts.txt $(twins ${P[@]})
  [ -f $E/mistral/done ] || ~/lw/scripts/device/energy_run.sh $X $E/mistral 1024 mistral.txt      $(twins ${M[@]})
  echo "$(date -Is) $dr done"
done
echo ORIN_KQ11G_DONE
