#!/bin/bash
# Power / energy of token generation on a Jetson: tegrastats every 50 ms (timestamped by energy_stamp.py) around
# llmrt `spec` runs (which print wall-clock phase marks: prefill, plain greedy, prompt-lookup speculation); 10 s of
# idle before the first and after the last run. Analysis: scripts/energy_analyze.py.
#   energy_run.sh <llmrt binary> <out dir> <ctx> <prompts file> <bundle> [<bundle> ...]
# TEGRA="sudo -n tegrastats" on the Nano (power rails are root-only there). LLMRT_ARGS: extra llmrt options.
set -u
X=$1; O=$2; CTX=$3; PR=$4; shift 4
mkdir -p $O
D=$(cd "$(dirname "$0")" && pwd)
${TEGRA:-tegrastats} --interval 50 | python3 -u $D/energy_stamp.py > $O/tegrastats.log &
SP=$!
sleep 10
for b in "$@"; do
  n=$(basename $b)
  DRAFT=7; case $n in *raw8*) DRAFT=0;; esac
  $X $b spec --ctx $CTX --draft $DRAFT --ngram 3 ${LLMRT_ARGS:-} < $PR > $O/spec_$n.jsonl 2> $O/spec_$n.err
  sleep 5
done
sleep 10
kill $SP 2>/dev/null; pkill -P $$ python3 2>/dev/null
echo DONE > $O/done
