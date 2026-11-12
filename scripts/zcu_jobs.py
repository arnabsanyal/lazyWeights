"""ZCU102 jobs: every co-simulation matrix (scripts/a4_cosim.py PICK: both symbol kinds, the longest codes, first /
middle / last layers, lm_head) cut WHOLE into chunks that fit an engine's payload (32768 words) and descriptor (16
tiles) memories -> build/fpga/zjobs/<bundle>__<matrix>.hmat + index.jsonl (C model cycles / stalls / words per matrix).

  .venv/bin/python scripts/zcu_jobs.py   (needs build/huff_chunks: g++ -O2 -std=c++17 -I hw/huff hw/huff/tb/chunks_main.cpp)
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from a4_cosim import PICK  # noqa: E402
from a4_csim import BUNDLES, NAS  # noqa: E402

BIN = os.path.join(REPO, "build", "huff_chunks")
OUT = os.path.join(REPO, "build", "fpga", "zjobs")
MAX_WORDS, MAX_TILES = 32768, 16


def one(bm):
    b, m = bm
    bdir = os.path.join(NAS, BUNDLES[b], "rt_bundles", b)
    f = os.path.join(OUT, "%s__%s.hmat" % (b, m))
    p = subprocess.run([BIN, bdir, m, f, str(MAX_WORDS), str(MAX_TILES)], capture_output=True, text=True)
    if p.returncode:
        return dict(bundle=b, matrix=m, error=p.stderr.strip())
    r = json.loads(p.stdout)
    r.update(bundle=b, job=os.path.basename(f), bytes=os.path.getsize(f))
    return r


def main():
    os.makedirs(OUT, exist_ok=True)
    todo = [(b, m) for b, ms in PICK.items() for m in ms]
    with ThreadPoolExecutor(8) as ex, open(os.path.join(OUT, "index.jsonl"), "w") as idx:
        for r in ex.map(one, todo):
            print(json.dumps(r), flush=True)
            idx.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
