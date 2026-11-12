"""FPGA bring-up jobs: tile slices of the co-simulation's matrices (scripts/a4_cosim.py PICK: both symbol kinds, the
longest codes, first / middle / last layers, lm_head), first / middle / last slice of each, as many tiles as fit the
board's payload memory (16384 words) -> build/fpga/jobs/*.job + build/fpga/jobs/index.jsonl (C model stats).

  .venv/bin/python scripts/fpga_jobs.py   (needs build/huff_slice: g++ -O2 -std=c++17 -I hw/huff hw/huff/tb/slice_main.cpp)
"""
import json
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from a4_cosim import PICK  # noqa: E402
from a4_csim import BUNDLES, NAS  # noqa: E402

BIN = os.path.join(REPO, "build", "huff_slice")
OUT = os.path.join(REPO, "build", "fpga", "jobs")


def ntiles(bdir, name):
    for line in open(os.path.join(bdir, "manifest.txt")):
        p = line.split()
        if p and p[0] == "t" and p[1] == name:
            N, K = int(p[3]), int(p[4])
            KS = K if p[2] == "huf8" else K // 2
            return (N // 64) * (KS // 256)
    raise KeyError(name)


def main():
    os.makedirs(OUT, exist_ok=True)
    idx = open(os.path.join(OUT, "index.jsonl"), "w")
    n = 0
    for b, mats in PICK.items():
        bdir = os.path.join(NAS, BUNDLES[b], "rt_bundles", b)
        for m in mats:
            T = ntiles(bdir, m)
            for t0 in sorted({0, T // 2, max(T - 4, 0)}):
                f = os.path.join(OUT, "%s__%s__t%d.job" % (b, m, t0))
                p = subprocess.run([BIN, bdir, m, str(t0), "0", f], capture_output=True, text=True)
                if p.returncode:
                    print("skip", b, m, t0, p.stderr.strip(), flush=True)
                    continue
                r = json.loads(p.stdout)
                r.update(bundle=b, job=os.path.basename(f))
                idx.write(json.dumps(r) + "\n")
                n += 1
    print(n, "jobs ->", OUT)


if __name__ == "__main__":
    main()
