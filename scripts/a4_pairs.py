"""A4 ablation: pair symbols on/off for the 4-bit (huf4p) headline bundles.

huf4p codes two 4-bit weights per symbol (one joint 256-symbol table per tensor); "off" codes each 4-bit weight as its
own symbol (a 16-symbol table per tensor, length-limited to 11 bits, the same huf8 stream layout). Per bundle:
  - size: coded payload bits per weight, pairs vs singles, over every huf4p matrix (the stored symbols r, before the
    per-row re-centering add, from build/huff_csim --dump-ref minus the .radd tensor). The pair figure is checked
    against the bundle's actual payload (results/logs/a4/<bundle>.jsonl).
  - rate: the C model (hw/huff/huff_engine.h, comparator lookup, buffer 64, FIFO 4, port 8, latency 8) on singles
    re-encodings of a sample of matrices (first, middle, last layer: all their huf4p matrices) vs the pair streams of
    the same matrices; weights per cycle = symbols per cycle x weights per symbol.

  .venv/bin/python scripts/a4_pairs.py [--bundles a,b] [--threads 16]
Rows -> results/a4_pairs.jsonl; table -> results/a4_pairs.md
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
from a4_csim import BIN, BUNDLES, NAS, build  # noqa: E402
from lazyweights.lanecodecs import limited_lengths  # noqa: E402
from lazyweights.bundle import LMAX, Writer, encode_huf8  # noqa: E402
from lazyweights.bundle import read_manifest  # noqa: E402
from lazyweights.log import JsonlLogger  # noqa: E402

HEAD4 = ["phi3_kq_IQ4_XS_g0_lam0_c1.25", "phi3_kq_IQ4_XS_g1_lam0_c1.25", "mistral_kq_IQ4_XS_g1_lam0",
         "mistral_kq_IQ4_XS_g1_lam0_c1.5", "smollm_kq_IQ4_XS_g1_lam0"]
REF_CFG = "comparator"   # buffer 64, FIFO 4, port 8, latency 8 (csim_main.cpp's second configuration)


def huff_bits(hist):
    L = limited_lengths(hist, LMAX).astype(np.int64)
    if (L > 0).sum() == 1:
        L = np.where(hist > 0, 1, 0)
    return int((L * hist).sum())


def symbols(bdir, man, name, tmp):
    """Stored pair symbols r [N, K/2] of a huf4p matrix (C++ reference decoder, re-centering add removed)."""
    t = man[name]
    KS = t["K"] // 2
    f = os.path.join(tmp, "ref.bin")
    subprocess.check_call([BIN, bdir, "--only", name, "--dump-ref", name, f, "--threads", "1"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    q = np.fromfile(f, np.uint8).reshape(t["N"], KS).astype(np.int64)
    os.remove(f)
    ra = man.get(name + ".radd")
    if ra:
        with open(os.path.join(bdir, "weights.bin"), "rb") as fh:
            fh.seek(ra["offset"])
            add = np.frombuffer(fh.read(ra["nbytes"]), np.float32).astype(np.int64).reshape(t["N"], -1)
        add = np.repeat(add, KS // add.shape[1], axis=1)
        q = (((q & 15) - add) & 15) | ((((q >> 4) - add) & 15) << 4)
    return q.astype(np.uint8)


def csim(bdir, threads):
    p = subprocess.run([BIN, bdir, "--threads", str(threads)], capture_output=True, text=True, check=True)
    rows = [json.loads(l) for l in p.stdout.splitlines()]
    return {r["name"]: r for r in rows if r["kind"] == "matrix" and r["config"] == REF_CFG}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", default=",".join(HEAD4))
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    build()
    md = ["# Pair symbols on/off (huf4p vs one 4-bit symbol per weight), 4-bit headline bundles", "",
          "Size over every huf4p matrix; rate from the C model (%s lookup, buffer 64, FIFO 4, port 8, latency 8) on the "
          "first, middle and last layer's huf4p matrices." % REF_CFG, "",
          "| bundle | matrices | b/w pairs | b/w singles | singles size | sample | w/cycle pairs | w/cycle singles |",
          "|" + "---|" * 8]
    for b in a.bundles.split(","):
        bdir = os.path.join(NAS, BUNDLES[b], "rt_bundles", b)
        man = {t["name"]: t for t in read_manifest(bdir)[1]}
        mats = [n for n, t in man.items() if t["enc"] == "huf4p"]
        layers = sorted({int(m.group(1)) for n in mats for m in [re.match(r"l(\d+)\.", n)] if m})
        pick = {layers[0], layers[len(layers) // 2], layers[-1]}
        sample = [n for n in mats if (re.match(r"l(\d+)\.", n) and int(re.match(r"l(\d+)\.", n).group(1)) in pick)]
        logged = {}
        lp = os.path.join(REPO, "results", "logs", "a4", b + ".jsonl")
        if os.path.exists(lp):
            logged = {r["name"]: r["bits"] for r in map(json.loads, open(lp))
                      if r["kind"] == "matrix" and r["config"] == REF_CFG}
        tmp = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
        sdir = os.path.join(tmp, "singles")
        w = Writer(sdir)
        bp = bs = nw = 0
        bad = []
        try:
            for n in mats:
                r = symbols(bdir, man, n, tmp)
                hp = np.bincount(r.ravel(), minlength=256)
                pb = huff_bits(hp)
                lo = np.bincount(r.ravel() & 15, minlength=16) + np.bincount(r.ravel() >> 4, minlength=16)
                sb = huff_bits(lo)
                if n in logged and logged[n] != pb:
                    bad.append((n, logged[n], pb))
                bp, bs, nw = bp + pb, bs + sb, nw + 2 * r.size
                if n in sample:   # singles in weight order: r[n, j] -> weights 2j (low), 2j + 1 (high)
                    s = np.empty((r.shape[0], 2 * r.shape[1]), np.uint8)
                    s[:, 0::2], s[:, 1::2] = r & 15, r >> 4
                    blob, _, _ = encode_huf8(s)
                    w.add(n, "huf8", s.shape[0], s.shape[1], 1.0, 0, blob)
            w.close()
            pr, sr = csim(bdir, a.threads), csim(sdir, a.threads)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        pc = sum(pr[n]["cycles"] for n in sample)
        sc = sum(sr[n]["cycles"] for n in sample)
        sw = sum(sr[n]["symbols"] for n in sample)   # singles: 1 weight per symbol
        pw = sum(2 * pr[n]["symbols"] for n in sample)
        assert pw == sw, (pw, sw)
        mis = sum(sr[n]["mismatch"] for n in sample)
        row = dict(bundle=b, matrices=len(mats), weights=nw, bits_pairs=bp, bits_singles=bs, bpw_pairs=bp / nw,
                   bpw_singles=bs / nw, size_ratio=bs / bp, sample=len(sample), sample_layers=sorted(pick),
                   wpc_pairs=pw / pc, wpc_singles=sw / sc, singles_mismatch=mis, pair_bits_vs_bundle=len(bad),
                   config=REF_CFG)
        JsonlLogger("a4_pairs", cfg=dict(bundle=b, config=REF_CFG)).log(row)
        print(b, "%.3f / %.3f b/w, %.2f / %.2f w/cycle" % (bp / nw, bs / nw, pw / pc, sw / sc),
              "pair-bits mismatches vs bundle:", len(bad), bad[:2], "singles csim mismatches:", mis, flush=True)
    last = {}   # table: latest row per bundle of results/a4_pairs.jsonl (runs may cover a subset of the bundles)
    for r in map(json.loads, open(os.path.join(REPO, "results", "a4_pairs.jsonl"))):
        last[r["bundle"]] = r
    for b in [b for b in HEAD4 if b in last] + [b for b in last if b not in HEAD4]:
        r = last[b]
        md.append("| %s | %d | %.3f | %.3f | %+.1f%% | %d | %.2f | %.2f |" % (
            b, r["matrices"], r["bpw_pairs"], r["bpw_singles"], 100 * (r["size_ratio"] - 1), r["sample"],
            r["wpc_pairs"], r["wpc_singles"]))
    open(os.path.join(REPO, "results", "a4_pairs.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
