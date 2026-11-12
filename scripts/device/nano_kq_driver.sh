#!/bin/bash
# Nano (MAXN): kq headline bundles one at a time (SD card space): copy from the model store ($LW_MODELS), fwd-bench m = 1..8 at position
# 128, then spec (plain greedy vs prompt lookup, draft 7, ngram 3) under tegrastats (energy_run.sh), delete the copy.
# Output on the Nano ~/ecw/spec/kq_out/ (fwd_<b>.jsonl) and ~/ecw/spec/energy_kq/<b>/. Skips bundles that do not fit
# (free space < bundle + 300 MB) or are done.
# BIN (llmrt_kq) names the runtime binary on the Nano and OUT (empty) suffixes the output dirs (kq_out$OUT,
# energy_kq$OUT); LLMRT_ARGS goes to spec. DB=1 also runs `check` (GPU vs CPU decode, bit-exact) and `decode-bench`
# (GPU and 4-thread CPU decode of every coded matrix) per bundle.
set -u
cd "$(dirname "$0")/../.."
N=jetson-nano; R=$LW_MODELS
declare -A DIR=([phi3]=phi3-mini-4k-instruct [smollm]=smollm-1-7b-instruct)
declare -A PR=([phi3]=spec_prompts.txt [smollm]=smollm.txt)
BIN=${BIN:-llmrt_kq}; OUT=${OUT:-}; DB=${DB:-0}
for b in "$@"; do
  m=${b%%_*}; src=$R/${DIR[$m]}/rt_bundles/$b
  if ssh $N "[ -f ~/ecw/spec/energy_kq$OUT/$b/done ]"; then echo "$(date -Is) $b done already"; continue; fi
  if ! ssh $N "[ -f ~/ecw_data/rt/$b/manifest.txt ]"; then   # (a copy left by an interrupted run is reused)
    need=$(( $(du -sm $src | cut -f1) + 300 )); free=$(ssh $N "df -m --output=avail / | tail -1")
    if [ $free -lt $need ]; then echo "$(date -Is) skip $b: need $need MB, free $free MB"; continue; fi
  fi
  [ "$(ssh $N 'dmesg | grep -c "Call trace"')" -le ${DMESG_BASE:-0} ] || { echo "$(date -Is) Call trace in dmesg: stop"; exit 1; }
  rsync -a $src $N:ecw_data/rt/ || { echo "$(date -Is) copy $b failed"; continue; }
  ssh $N "cd ~/ecw/spec; K=kq_out$OUT; mkdir -p \$K
    if [ $DB = 1 ]; then
      ./$BIN ~/ecw_data/rt/$b check > \$K/check_$b.txt 2>&1; echo check rc=\$?
      ./$BIN ~/ecw_data/rt/$b decode-bench --threads 4 > \$K/decode_$b.jsonl 2> \$K/decode_$b.err; echo decode rc=\$?
    fi
    ./$BIN ~/ecw_data/rt/$b fwd-bench --prompt-len 128 --reps 5 > \$K/fwd_$b.jsonl 2> \$K/fwd_$b.err
    echo fwd rc=\$?
    TEGRA='sudo -n tegrastats' LLMRT_ARGS='${LLMRT_ARGS:-}' scripts/device/energy_run.sh ./$BIN ~/ecw/spec/energy_kq$OUT/$b 1024 ${PR[$m]} ~/ecw_data/rt/$b
    echo energy rc=\$?
    rm -rf ~/ecw_data/rt/$b"
  echo "$(date -Is) $b finished"
done
echo NANO_KQ_DONE
