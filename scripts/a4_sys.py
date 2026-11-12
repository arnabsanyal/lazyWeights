"""A4 / A5 system check of the Huffman tile engines on the headline bundles (hw/huff/tb/sys_main.cpp, comparator
lookup, bit-exact vs the reference decoder on every matrix):

  1. latency / FIFO sweep: one engine, no bandwidth cap, read latency LAT in {8, 128, 256, 512, 1024} cycles
     (1 GHz: up to 1 us) x per-lane FIFO depth; on the first, middle and last layer + lm_head of each bundle.
     Smallest FIFO reaching 99% of the LAT = 8 rate per (bundle, LAT).
  2. shared DRAM: T - 1, T, T + 1 engines (T = A5's engine count, results/a5_system.jsonl) on the whole model, a
     barrier per matrix, the DRAM granting at most cap words / cycle over all engines, cap = the payload share of
     102 GB/s: theta* / f x (payload bits per weight, A4 C sim) / 32, i.e. the rate at which a decoder that never
     stalls delivers theta* = 8 beta / b_c weights / s. Pass: achieved weights / s >= 0.99 theta*.

  .venv/bin/python scripts/a4_sys.py [--bundles a,b] [--jobs 12] [--part 1|2|all]
Rows -> results/a4_sys.jsonl; per-run JSON lines -> results/logs/a4/sys/
"""
import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from lazyweights.log import JsonlLogger  # noqa: E402

from lazyweights.paths import MODELS as NAS  # noqa: E402
BUNDLES = {
    "phi3_kq_IQ4_XS_g0_lam0_c1.25": "phi3-mini-4k-instruct", "phi3_kq_IQ4_XS_ref": "phi3-mini-4k-instruct", "phi3_kq_IQ4_XS_g1_lam0_c1.25": "phi3-mini-4k-instruct",
    "mistral_kq_IQ4_XS_g1_lam0": "mistral-7b-instruct-v0.2", "mistral_kq_IQ4_XS_g1_lam0_c1.5": "mistral-7b-instruct-v0.2",
    "smollm_kq_Q5_K_M_g1_lam0": "smollm-1-7b-instruct", "smollm_kq_IQ4_XS_g1_lam0": "smollm-1-7b-instruct",
}
BIN = os.path.join(REPO, "build", "huff_sys")
LOGD = os.path.join(REPO, "results", "logs", "a4", "sys")
LATS = [8, 128, 256, 512, 1024]
FIFOS = [4, 8, 16, 32, 64, 128, 256, 512]
F_CLK = 1e9


def build():
    src = [os.path.join(REPO, "hw", "huff", f) for f in ("huff_engine.h", os.path.join("tb", "sys_main.cpp"))]
    if not os.path.exists(BIN) or os.path.getmtime(BIN) < max(map(os.path.getmtime, src)):
        subprocess.check_call(["g++", "-O2", "-std=c++17", "-pthread", "-Wall", "-Wno-unknown-pragmas", "-I",
                               os.path.join(REPO, "hw", "huff"), src[1], "-o", BIN])


def bdir(b):
    return os.path.join(NAS, BUNDLES[b], "rt_bundles", b)


def run(b, tag, args, threads=4):
    """One huff_sys run -> its summary line (cached in LOGD/<bundle>_<tag>.jsonl)."""
    p = os.path.join(LOGD, "%s_%s.jsonl" % (b, tag))
    if not (os.path.exists(p) and any('"summary"' in l for l in open(p))):
        with open(p, "w") as f:
            r = subprocess.run([BIN, bdir(b), "--threads", str(threads)] + args, stdout=f, stderr=subprocess.PIPE,
                               text=True)
        if r.returncode:
            raise RuntimeError("%s %s: rc %d %s" % (b, tag, r.returncode, r.stderr.strip()))
    return json.loads(next(l for l in open(p) if '"summary"' in l))


def sample_prefixes(b):
    ls = sorted({int(m.group(1)) for l in open(os.path.join(bdir(b), "manifest.txt"))
                 for m in [re.match(r"t l(\d+)\.", l)] if m})
    return "l%d.,l%d.,l%d.,lm_head" % (ls[0], ls[len(ls) // 2], ls[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", default=",".join(BUNDLES))
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--part", default="all")
    a = ap.parse_args()
    build()
    os.makedirs(LOGD, exist_ok=True)
    bs = a.bundles.split(",")
    pool = ThreadPoolExecutor(a.jobs)
    a5 = {}
    for l in open(os.path.join(REPO, "results", "a5_system.jsonl")):
        r = json.loads(l)
        if r.get("kind") == "point":
            a5[r["bundle"]] = r
    csim = {}
    for l in open(os.path.join(REPO, "results", "a4_csim.jsonl")):
        r = json.loads(l)
        if r["config"] == "comparator":
            csim[r["bundle"]] = r

    # 1. latency / FIFO sweep, one engine, no cap
    jobs = {(b, lat, fi): pool.submit(run, b, "sweep_lat%d_fifo%d" % (lat, fi),
                                      ["--engines", "1", "--lat", str(lat), "--fifo", str(fi), "--prefix",
                                       sample_prefixes(b)])
            for b in bs for lat in LATS for fi in FIFOS if fi <= max(4, 2 * lat)}
    fifo_at = {}
    for b in bs:
        ref = jobs[(b, 8, 4)].result()["weights_per_cycle"]
        for lat in LATS:
            for fi in FIFOS:
                if (b, lat, fi) not in jobs:
                    continue
                s = jobs[(b, lat, fi)].result()
                ok = s["weights_per_cycle"] >= 0.99 * ref
                if ok and (b, lat) not in fifo_at:
                    fifo_at[(b, lat)] = fi
                JsonlLogger("a4_sys", cfg=dict(part="sweep", bundle=b, lat=lat, fifo=fi)).log(dict(
                    part="sweep", bundle=b, engines=1, lat=lat, fifo=fi, port=8, lookup=1, cap=0,
                    prefix=sample_prefixes(b), matrices=s["matrices"], mismatch_matrices=s["mismatch_matrices"],
                    weights_per_cycle=s["weights_per_cycle"], ref_weights_per_cycle=ref,
                    rel=s["weights_per_cycle"] / ref, words_per_cycle=s["words_per_cycle"],
                    fifo_bits_per_engine=32 * fi * 32))
            print("%s lat %d: smallest FIFO at 99%% = %s" % (b, lat, fifo_at.get((b, lat))), flush=True)
    if a.part == "1":
        return

    # 2. shared DRAM, whole model
    jobs = {}
    for b in bs:
        T = a5[b]["engines"]
        theta = a5[b]["theta_star"]
        bpw = csim[b]["bits_per_weight"]
        cap = theta / F_CLK * bpw / 32
        for lat in [8, 256, 512, 1024]:
            fi = fifo_at.get((b, lat))
            if fi is None:
                continue
            for e in (T - 1, T, T + 1):
                jobs[(b, lat, e)] = (fi, cap, theta, pool.submit(
                    run, b, "sys_e%d_lat%d_fifo%d" % (e, lat, fi),
                    ["--engines", str(e), "--lat", str(lat), "--fifo", str(fi), "--cap", "%.6f" % cap], 8))
    for (b, lat, e), (fi, cap, theta, j) in sorted(jobs.items()):
        s = j.result()
        ach = s["weights_per_cycle"] * F_CLK
        JsonlLogger("a4_sys", cfg=dict(part="system", bundle=b, lat=lat, fifo=fi, engines=e)).log(dict(
            part="system", bundle=b, engines=e, a5_engines=a5[b]["engines"], lat=lat, fifo=fi, port=8, lookup=1,
            cap_words_per_cycle=cap, theta_star=theta, achieved_weights_per_s=ach, rel_theta=ach / theta,
            passed=ach >= 0.99 * theta, matrices=s["matrices"], mismatch_matrices=s["mismatch_matrices"],
            words_per_cycle=s["words_per_cycle"], dram_util=s["words_per_cycle"] / cap,
            capped_frac=s["capped_cycles"] / s["cycles"], cycles=s["cycles"],
            fifo_bits_per_engine=32 * fi * 32))
        print("%s lat %4d fifo %3d engines %d: %.1f G w/s = %.3f theta*  DRAM util %.3f  mismatch %d" % (
            b, lat, fi, e, ach / 1e9, ach / theta, s["words_per_cycle"] / cap, s["mismatch_matrices"]), flush=True)


if __name__ == "__main__":
    main()
