#!/bin/bash
# Nano (MAXN, 4 GB): Mistral-7B headline points by layer streaming. llmrt keeps the first matrices (forward order)
# resident up to a fixed weight budget (BUDGET_MB, the same for every bundle) and streams the rest from storage every
# forward pass (O_DIRECT, two pinned staging buffers, loader thread). The floor alone is not enough: a 3.2 GB bundle
# then loads fully resident with 403 MB left and the forward pass is OOM-killed (2026-10-31). Bundles written by
# `llmrt export-il` (coded matrices stored in the device layout, compact): <point>_ilc, twin <point>_pk_ilc.
#   nano_mistral_driver.sh <tag> <storage dir on the Nano> <bundle> ...   (tag: sd | usb, names the output dirs)
# Per bundle: copy from the model store ($LW_MODELS), fwd-bench (m = 1..8, zero-copy staging, then PHI3RT_NO_ZERO_COPY=1), spec (plain vs
# prompt lookup, 6 prompts, 32 new tokens: mistral32.txt) under tegrastats, delete the copy.
# NOZC=0 skips the PHI3RT_NO_ZERO_COPY run (it needs more host memory: OOM at budget 2560, 2026-11-01). DMESG_BASE:
# Call-trace count already in dmesg (known, not GPU) that does not stop the run.
# RT_ENV: environment for every llmrt run (PHI3RT_NO_DIRECT=1: buffered reads, streamed matrices served from the page
# cache once read). BIN (llmrt_kq5): runtime binary on the Nano; LLMRT_ARGS: extra spec options (with --budget-mb).
# Output on the Nano ~/ecw/spec/stream_<tag>/fwd_<b>[_nozc].{jsonl,err} and ~/ecw/spec/energy_stream_<tag>/<b>/.
set -u
cd "$(dirname "$0")/../.."
N=jetson-nano; R=$LW_MODELS/mistral-7b-instruct-v0.2/rt_bundles
TAG=$1; ST=$2; shift 2
BUDGET_MB=${BUDGET_MB:-2048}; BIN=${BIN:-llmrt_kq5}
until grep -q NANO_LLAMA_DONE results/logs/a2/nano_llama_driver.log 2>/dev/null; do sleep 60; done   # SD / power
for b in "$@"; do
  if ssh $N "[ -f ~/ecw/spec/energy_stream_$TAG/$b/done ]"; then echo "$(date -Is) $b done already"; continue; fi
  [ "$(ssh $N 'dmesg | grep -c "Call trace"')" -le ${DMESG_BASE:-0} ] || { echo "$(date -Is) Call trace in dmesg: stop"; exit 1; }
  need=$(( $(du -sm $R/$b | cut -f1) + 300 )); free=$(ssh $N "mkdir -p $ST; df -m --output=avail $ST | tail -1")
  if ! ssh $N "[ -f $ST/$b/manifest.txt ]" && [ $free -lt $need ]; then echo "$(date -Is) skip $b: need $need MB, free $free MB"; continue; fi
  rsync -a $R/$b $N:$ST/ || { echo "$(date -Is) copy $b failed"; continue; }
  echo "$(date -Is) $b copied"
  ssh $N "cd ~/ecw/spec; O=stream_$TAG; mkdir -p \$O
    ${RT_ENV:-} ./$BIN $ST/$b fwd-bench --prompt-len 32 --reps ${REPS:-2} --budget-mb $BUDGET_MB > \$O/fwd_$b.jsonl 2> \$O/fwd_$b.err; echo fwd rc=\$?
    [ ${NOZC:-1} = 1 ] && PHI3RT_NO_ZERO_COPY=1 ./$BIN $ST/$b fwd-bench --prompt-len 32 --reps 2 --budget-mb $BUDGET_MB > \$O/fwd_${b}_nozc.jsonl 2> \$O/fwd_${b}_nozc.err
    echo fwd nozc rc=\$?
    ${RT_ENV:-} TEGRA='sudo -n tegrastats' LLMRT_ARGS='--budget-mb $BUDGET_MB ${LLMRT_ARGS:-}' scripts/device/energy_run.sh ./$BIN ~/ecw/spec/energy_stream_$TAG/$b 512 mistral32.txt $ST/$b
    echo energy rc=\$?
    rm -rf $ST/$b"
  echo "$(date -Is) $b finished"
done
echo NANO_MISTRAL_DONE
