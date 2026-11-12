#!/bin/bash
# Nano (MAXN): llama.cpp CPU baseline for Mistral-7B (as nano_llama_driver.sh: llama-lookup-cpu, 4 threads, the 6
# lookup prompts under tegrastats, marks.txt), the two comparators and our two GGUFs. Every GGUF (3.3-5.1 GB) is
# larger than the free RAM: llama.cpp mmaps it and pages weights from the SD card each token (--no-repack: no
# anonymous copy of the weights). 32 new tokens (as the GPU streaming runs, mistral32.txt). One GGUF at a time
# (SD card space). A watchdog kills llama-lookup when MemAvailable < 384 MiB (rc=137 in marks.txt).
#   nano_llama_mistral.sh [<draft n_max> ...]   (default 0: plain greedy; 7: prompt lookup)
# GD: GGUF directory on the Nano (~/ecw_data/gguf, SD card; /mnt/usb/gguf: the USB stick, which keeps every GGUF);
# D: output dir under ~/ecw/spec (energy_kq/llama_mistral -> results/a2_nano/energy_kq/llama_mistral/).
set -u
cd "$(dirname "$0")/../.."
while [ -f results/logs/a2/HOLD_nano ]; do sleep 60; done   # queue paused by hand (rm HOLD_nano to go on)
N=jetson-nano; R=$LW_MODELS/mistral-7b-instruct-v0.2/gguf
G="$R/kq/IQ4_XS_g1_lam0.gguf:kq_IQ4_XS_g1_lam0 $R/bench/cmp_Q5_K_M_no.gguf:cmp_Q5_K_M_no
   $R/kq/IQ4_XS_g1_lam0_c1.5.gguf:kq_IQ4_XS_g1_lam0_c1.5 $R/bench/cmp_IQ3_M_im.gguf:cmp_IQ3_M_im"
GD=${GD:-~/ecw_data/gguf}; D=${D:-energy_kq/llama_mistral}
ssh $N "mkdir -p ~/ecw/spec/$D $GD"
for d in ${@:-0}; do for e in $G; do
  g=${e##*:}
  if ssh $N "[ -f ~/ecw/spec/$D/done_${g}_d$d ]"; then echo "$(date -Is) $g d$d done already"; continue; fi
  [ "$(ssh $N 'dmesg | grep -c "Call trace"')" -le ${DMESG_BASE:-0} ] || { echo "$(date -Is) Call trace in dmesg: stop"; exit 1; }
  if ! ssh $N "[ -f $GD/$g.gguf ]"; then
    case $GD in /mnt/usb/*) ;; *) ssh $N "rm -f $GD/*.gguf";; esac
    rsync -a ${e%%:*} $N:$GD/$g.gguf || { echo "$(date -Is) copy $g failed"; exit 1; }
    echo "$(date -Is) $g copied"
  fi
  ssh $N "cd ~/ecw/spec; D=$D; f=$GD/$g.gguf
    sudo -n tegrastats --interval 50 | python3 -u scripts/energy_stamp.py >> \$D/tegrastats.log &
    SP=\$!
    sleep 10
    for p in ~/ecw_data/lookup_prompts/mistral_*.txt; do
      t=\$(basename \$p .txt); t=\${t#mistral_}
      sync; sudo -n /usr/local/sbin/drop-caches   # every prompt starts with a cold page cache
      echo \"start $g $d \$t \$(date +%s.%N)\" >> \$D/marks.txt
      ./llama-lookup-cpu -m \$f -f \$p -n 32 -c 1024 -t 4 --temp 0 --top-k 1 --spec-draft-n-max $d --no-repack \
         > \$D/${g}_d${d}_\$t.out 2> \$D/${g}_d${d}_\$t.err &
      LP=\$!
      while kill -0 \$LP 2>/dev/null; do
        a=\$(awk '/MemAvailable/ {print int(\$2/1024)}' /proc/meminfo)
        if [ \$a -lt 384 ]; then kill -9 \$LP; echo \"floor \$t MemAvailable \$a MB\" >> \$D/marks.txt; fi
        sleep 1
      done
      wait \$LP; rc=\$?
      echo \"end $g $d \$t \$(date +%s.%N) rc=\$rc\" >> \$D/marks.txt
      sleep 2
    done
    sleep 10; kill \$SP 2>/dev/null; echo DONE > \$D/done_${g}_d$d"
  echo "$(date -Is) $g d$d finished"
done; done
case $GD in /mnt/usb/*) ;; *) ssh $N "rm -f $GD/*.gguf";; esac
echo NANO_LLAMA_MISTRAL_DONE
