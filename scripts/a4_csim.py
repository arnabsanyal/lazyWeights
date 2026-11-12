"""A4 step 1: C simulation of the Huffman tile engine (hw/huff/huff_engine.h) on every Huffman-coded matrix of the
operating-point bundles: bit-exact vs the reference decoder for the table and the comparator lookup, and the
engine's symbols per cycle under the memory configurations of hw/huff/tb/csim_main.cpp (refill stalls on the real
streams). The C++ reference decoder is itself checked against the NumPy one (ecw/rt/bundle.py decode_huf8, plus the
re-centering add) on one matrix per bundle.

  .venv/bin/python scripts/a4_csim.py [--bundles a,b] [--threads 16]
Rows -> results/a4_csim.jsonl (one per bundle and configuration); per-matrix lines -> results/logs/a4/<bundle>.jsonl
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from lazyweights.bundle import decode_huf8  # noqa: E402
from lazyweights.bundle import read_manifest  # noqa: E402
from lazyweights.log import JsonlLogger  # noqa: E402

from lazyweights.paths import MODELS as NAS  # noqa: E402
BUNDLES = {   # headline points (kq) first, then the earlier A1 operating points
    "phi3_kq_IQ4_XS_g0_lam0_c1.25": "phi3-mini-4k-instruct", "phi3_kq_IQ4_XS_ref": "phi3-mini-4k-instruct", "phi3_kq_IQ4_XS_g1_lam0_c1.25": "phi3-mini-4k-instruct",
    "mistral_kq_IQ4_XS_g1_lam0": "mistral-7b-instruct-v0.2", "mistral_kq_IQ4_XS_g1_lam0_c1.5": "mistral-7b-instruct-v0.2",
    "smollm_kq_Q5_K_M_g1_lam0": "smollm-1-7b-instruct", "smollm_kq_IQ4_XS_g1_lam0": "smollm-1-7b-instruct",
    "phi3_q4_g128_lam0.1": "phi3-mini-4k-instruct", "phi3_q4_channel_lam0.1": "phi3-mini-4k-instruct",
    "q8_gptq_huf8": "phi3-mini-4k-instruct", "mistral_q4_g128_lam0_ao": "mistral-7b-instruct-v0.2",
    "mistral_q4_channel_lam0.1": "mistral-7b-instruct-v0.2", "mistral_q8_tensor_lam0.3": "mistral-7b-instruct-v0.2",
    "smollm_q4_g128_lam0": "smollm-1-7b-instruct", "smollm_q4_channel_lam0.1": "smollm-1-7b-instruct",
    "smollm_q8_tensor_lam0.3": "smollm-1-7b-instruct",
}
BIN = os.path.join(REPO, "build", "huff_csim")


def build():
    src = [os.path.join(REPO, "hw", "huff", f) for f in ("huff_engine.h", os.path.join("tb", "csim_main.cpp"))]
    if not os.path.exists(BIN) or os.path.getmtime(BIN) < max(map(os.path.getmtime, src)):
        os.makedirs(os.path.dirname(BIN), exist_ok=True)
        subprocess.check_call(["g++", "-O2", "-std=c++17", "-pthread", "-Wall", "-Wno-unknown-pragmas", "-I",
                               os.path.join(REPO, "hw", "huff"), src[1], "-o", BIN])


def numpy_check(bdir, man):
    """C++ reference decoder == NumPy decode_huf8 (+ add) on the bundle's first Huffman matrix."""
    name = next(n for n, t in man.items() if t["enc"] in ("huf8", "huf4p"))
    t = man[name]
    bits = 8 if t["enc"] == "huf8" else 4
    KS = t["K"] if bits == 8 else t["K"] // 2
    with open(os.path.join(bdir, "weights.bin"), "rb") as f:
        f.seek(t["offset"])
        blob = f.read(t["nbytes"])
        r = decode_huf8(blob, t["N"], KS).astype(np.int64)
        ra = man.get(name + ".radd")
        if ra:
            f.seek(ra["offset"])
            add = np.frombuffer(f.read(ra["nbytes"]), np.float32).astype(np.int64).reshape(t["N"], -1)
            add = np.repeat(add, KS // add.shape[1], axis=1)
            r = ((r + add) & 255) if bits == 8 else (((r & 15) + add) & 15) | (((r >> 4) + add) & 15) << 4
    with tempfile.NamedTemporaryFile() as tf:
        subprocess.check_call([BIN, bdir, "--only", name, "--dump-ref", name, tf.name, "--threads", "1"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        c = np.fromfile(tf.name, np.uint8).reshape(t["N"], KS)
    return name, int((c != r.astype(np.uint8)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", default=",".join(BUNDLES))
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    build()
    os.makedirs(os.path.join(REPO, "results", "logs", "a4"), exist_ok=True)
    for b in a.bundles.split(","):
        bdir = os.path.join(NAS, BUNDLES[b], "rt_bundles", b)
        man = {t["name"]: t for t in read_manifest(bdir)[1]}
        chk, nmis = numpy_check(bdir, man)
        print(b, "numpy check", chk, "mismatches", nmis, flush=True)
        lp = os.path.join(REPO, "results", "logs", "a4", b + ".jsonl")
        with open(lp, "w") as f:
            p = subprocess.run([BIN, bdir, "--threads", str(a.threads)], stdout=f, stderr=subprocess.PIPE, text=True)
        rows = [json.loads(l) for l in open(lp)]
        mats = [r for r in rows if r["kind"] == "matrix"]
        encs = sorted({r["enc"] for r in mats})
        maxlen = max(r["max_len"] for r in mats)
        for s in (r for r in rows if r["kind"] == "summary"):
            cfg = dict(bundle=b, **{k: s[k] for k in ("config", "lookup", "buf", "fifo", "port", "lat", "pre")})
            wps = 2 if encs == ["huf4p"] else None   # weights per symbol (mixed bundles: from the symbol counts)
            w = sum(r["symbols"] * (2 if r["enc"] == "huf4p" else 1) for r in mats if r["config"] == s["config"])
            JsonlLogger("a4_csim", cfg=cfg).log(dict(
                cfg, encs=encs, max_len=maxlen, matrices=s["matrices"], mismatch=s["mismatch"],
                numpy_check=chk, numpy_mismatch=nmis, rc=p.returncode, symbols=s["symbols"], weights=w,
                cycles=s["cycles"], stalls=s["stalls"], words=s["words"], bits=s["bits"],
                sym_per_cycle=s["sym_per_cycle"], weights_per_cycle=w / s["cycles"],
                stall_frac=s["stalls"] / (32.0 * s["cycles"]), bits_per_symbol=s["bits_per_symbol"],
                bits_per_weight=s["bits"] / w, words_per_cycle=s["words_per_cycle"],
                fetch_eff=s["bits"] / 32.0 / s["words"], weights_per_symbol=wps))
            print("  %-14s mis=%d  sym/cyc %.3f  w/cyc %.2f  stall %.4f  words/cyc %.2f" % (
                s["config"], s["mismatch"], s["sym_per_cycle"], w / s["cycles"], s["stalls"] / 32.0 / s["cycles"],
                s["words_per_cycle"]), flush=True)
        print(b, "rc", p.returncode, p.stderr.strip(), flush=True)


if __name__ == "__main__":
    main()
