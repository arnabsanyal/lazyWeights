#!/bin/bash
# Run-to-run variance of the Orin end-to-end table (paper Table "e2e"): R repeats of exactly its measured rows, under
# tegrastats (module input rail): our runtime llmrt_kq11g (CUDA graphs, our prompt-lookup drafter) on the six
# fixed-width twins (the "+unit" time), and llama.cpp llama-lookup on the six budget-baseline GGUFs, greedy (draft 0)
# and prompt-lookup speculation (draft 7), same prompts / 128 new tokens / ctx 1024 as the table.
#   ~/lw/scripts/device/orin_variance.sh [R=5]   -> ~/lw_data/var/r<i>/{phi3g0,phi3,mistral,smollm,llama_<m>}
# Needs MAXN_SUPER + jetson_clocks (as the table). Resumable (done markers). Analysis: scripts/a5_variance.py.
set -u
R=${1:-5}
cd ~/lw_data
X=~/lw/rt/build/llmrt_kq11g; L=~/lw/llama.cpp/build/bin/llama-lookup
export LLMRT_GRAPHS=1 LLMRT_ARGS="--drafter lookup"
export LD_LIBRARY_PATH=~/mm/root/envs/lcpp/lib:${LD_LIBRARY_PATH:-}
declare -A CMP=([phi3]="cmp_IQ4_XS_im cmp_Q3_K_M_im" [mistral]="cmp_Q5_K_M_no cmp_IQ3_M_im" [smollm]="cmp_Q5_K_M_no cmp_IQ4_XS_im")
for i in $(seq 1 $R); do
  E=~/lw_data/var/r$i; mkdir -p $E
  [ -f $E/phi3g0/done ]  || ~/lw/scripts/device/energy_run.sh $X $E/phi3g0  1024 spec_prompts.txt rt/phi3_kq_IQ4_XS_g0_lam0_c1.25_pk
  [ -f $E/phi3/done ]    || ~/lw/scripts/device/energy_run.sh $X $E/phi3    1024 spec_prompts.txt rt/phi3_kq_IQ4_XS_g1_lam0_c1.25_pk
  [ -f $E/mistral/done ] || ~/lw/scripts/device/energy_run.sh $X $E/mistral 1024 mistral.txt rt/mistral_kq_IQ4_XS_g1_lam0_pk rt/mistral_kq_IQ4_XS_g1_lam0_c1.5_pk
  [ -f $E/smollm/done ]  || ~/lw/scripts/device/energy_run.sh $X $E/smollm  1024 smollm.txt rt/smollm_kq_Q5_K_M_g1_lam0_pk rt/smollm_kq_IQ4_XS_g1_lam0_pk
  for m in phi3 mistral smollm; do
    D=$E/llama_$m; [ -f $D/done ] && continue; rm -rf $D; mkdir -p $D
    tegrastats --interval 50 | python3 -u ~/lw/scripts/energy_stamp.py > $D/tegrastats.log &
    SP=$!
    sleep 10
    for g in ${CMP[$m]}; do
      for d in 0 7; do for p in ~/lw_data/lookup_prompts/${m}_*.txt; do
        t=$(basename $p .txt); t=${t#${m}_}
        echo "start $g $d $t $(date +%s.%N)" >> $D/marks.txt
        $L -m gguf/$m/$g.gguf -f $p -n 128 -c 1024 -ngl 99 --temp 0 --top-k 1 --spec-draft-n-max $d \
           > $D/${g}_d${d}_$t.out 2> $D/${g}_d${d}_$t.err
        echo "end $g $d $t $(date +%s.%N) rc=$?" >> $D/marks.txt
        sleep 2
      done; done
    done
    sleep 10; kill $SP 2>/dev/null   # python3 exits; tegrastats ends on SIGPIPE
    echo DONE > $D/done
  done
  echo "$(date -Is) repeat $i done"
done
echo ORIN_VARIANCE_DONE
