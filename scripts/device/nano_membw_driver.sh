#!/bin/bash
# Nano DRAM read energy for the A5 energy model (kernels/membw_energy.cu, as on the Orin: results/a5_energy/orin):
# waits for the Mistral streaming driver (the Nano must be idle), builds the benchmark on the Nano (nvcc 10.2, sm_53),
# sweeps a 1 GiB (DRAM) and a 0.125 MiB (L2-resident, 256 KiB L2) buffer over 5 throttle levels under tegrastats
# (sudo -n), fetches to results/a5_energy/nano.
set -u
cd "$(dirname "$0")/../.."
N=jetson-nano
until grep -q NANO_MISTRAL_DONE results/logs/a2/nano_mistral_sd.log 2>/dev/null; do sleep 120; done
[ "$(ssh $N 'dmesg | grep -c "Call trace"')" = 0 ] || { echo "$(date -Is) Call trace in dmesg: stop"; exit 1; }
scp -q kernels/membw_energy.cu $N:~/ecw/spec/
ssh $N 'cd ~/ecw/spec && /usr/local/cuda-10.2/bin/nvcc -O3 -arch=sm_53 membw_energy.cu -o membw_energy && echo built'
ssh $N 'cd ~/ecw/spec; rm -rf membw; mkdir -p membw
  sudo -n tegrastats --interval 50 | python3 -u scripts/energy_stamp.py > membw/tegrastats.log &
  SP=$!; sleep 6; : > membw/runs.jsonl
  for mb in 1024 0.125; do for s in 0 300 1000 3000 10000; do ./membw_energy $mb $s 6 >> membw/runs.jsonl; sleep 3; done; done
  sleep 2; kill $SP; cat membw/runs.jsonl'   # tegrastats dies on its next write (SIGPIPE), as in energy_run.sh
mkdir -p results/a5_energy/nano
scp -q "$N:~/ecw/spec/membw/*" results/a5_energy/nano/
echo NANO_MEMBW_DONE
