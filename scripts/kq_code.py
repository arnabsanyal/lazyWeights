"""Huffman coding of llama.cpp's own codes (no change to the quantizer). Re-quantize the f16 GGUF of
scripts/a1_gguf.py with the same importance matrix (llama-quantize is deterministic: same tensors as the scored A1
row), split every block of a coded type into its codes q and its side bytes (scales / mins / d), check that packing q
back gives the block bytes exactly, and code q with the runtime coder (ecw.rt.bundle.encode_huf8: one length-limited
table per matrix, pair symbols for 4-bit codes, 64-row tiles with 32 lane offsets; LUT and offsets counted). Side
bytes stay as they are. Coded types: Q4_K, IQ4_XS (4-bit, pairs), Q5_K, Q6_K, Q8_0; any other type (Q3_K, IQ3_*, f32
norms) and an untied token_embd stay packed (A1 rule: the embedding is a random-access gather).
Quality: copied from the A1 row of the same model / type (method llama.cpp[+imatrix]); bits from this file.

  .venv/bin/python scripts/kq_code.py --model phi3 --types Q4_K_S,Q4_K_M,IQ4_XS,Q5_K_M,Q6_K [--no-imatrix]
-> results/a1_frontier.jsonl (family coded, method llama.cpp+imatrix+huf)
"""
import argparse
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LLAMA = os.path.abspath(os.environ.get("LW_LLAMA_CPP", os.path.join(REPO, "third_party", "llama.cpp")))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(1, os.path.join(LLAMA, "gguf-py"))
import gguf  # noqa: E402
from lazyweights.lanecodecs import limited_lengths  # noqa: E402
from lazyweights.bundle import encode_huf8  # noqa: E402
from lazyweights.log import JsonlLogger  # noqa: E402
from lazyweights.models import SPECS  # noqa: E402

T = gguf.GGMLQuantizationType


def split(ty, b):
    """Blocks [n, bytes] -> (codes [n, 256 or 32] uint8, side bytes [n, s], code bits). Codes in weight order."""
    n = b.shape[0]
    if ty == T.Q4_K:
        side, qs = b[:, :16], b[:, 16:]
        q = (qs.reshape(n, 4, 1, 32) >> np.array([0, 4], np.uint8).reshape(1, 1, 2, 1)) & 15
        return q.reshape(n, 256), side, 4
    if ty == T.Q5_K:
        side, qh, qs = b[:, :16], b[:, 16:48], b[:, 48:]
        ql = (qs.reshape(n, 4, 1, 32) >> np.array([0, 4], np.uint8).reshape(1, 1, 2, 1)) & 15
        hi = (qh.reshape(n, 1, 32) >> np.arange(8, dtype=np.uint8).reshape(1, 8, 1)) & 1
        return (ql.reshape(n, 8, 32) | (hi << 4)).reshape(n, 256), side, 5
    if ty == T.Q6_K:
        ql, qh, side = b[:, :128], b[:, 128:192], b[:, 192:]
        lo = (ql.reshape(n, 2, 1, 64) >> np.array([0, 4], np.uint8).reshape(1, 1, 2, 1)) & 15
        hi = (qh.reshape(n, 2, 1, 32) >> np.array([0, 2, 4, 6], np.uint8).reshape(1, 1, 4, 1)) & 3
        return (lo.reshape(n, 2, 4, 32) | (hi << 4)).reshape(n, 256), side, 6
    if ty == T.IQ4_XS:
        side, qs = b[:, :8], b[:, 8:]
        q = (qs.reshape(n, 8, 1, 16) >> np.array([0, 4], np.uint8).reshape(1, 1, 2, 1)) & 15
        return q.reshape(n, 256), side, 4
    if ty == T.Q8_0:
        return (b[:, 2:].view(np.int8).astype(np.int16) + 128).astype(np.uint8), b[:, :2], 8
    return None


def join(ty, q, side):
    """Inverse of split (the lossless check)."""
    n = q.shape[0]
    if ty in (T.Q4_K, T.IQ4_XS):
        g = q.reshape(n, 4, 2, 32) if ty == T.Q4_K else q.reshape(n, 8, 2, 16)
        return np.concatenate([side, (g[:, :, 0] | (g[:, :, 1] << 4)).reshape(n, -1)], 1)
    if ty == T.Q5_K:
        g = q.reshape(n, 8, 32)
        qh = np.bitwise_or.reduce((g >> 4) << np.arange(8, dtype=np.uint8).reshape(1, 8, 1), axis=1)
        lo = (g & 15).reshape(n, 4, 2, 32)
        return np.concatenate([side, qh, (lo[:, :, 0] | (lo[:, :, 1] << 4)).reshape(n, -1)], 1)
    if ty == T.Q6_K:
        g = q.reshape(n, 2, 4, 32)
        lo = (g & 15).reshape(n, 2, 2, 64)
        ql = (lo[:, :, 0] | (lo[:, :, 1] << 4)).reshape(n, 128)
        qh = np.bitwise_or.reduce((g >> 4) << np.array([0, 2, 4, 6], np.uint8).reshape(1, 1, 4, 1), axis=2)
        return np.concatenate([ql, qh.reshape(n, 64), side], 1)
    if ty == T.Q8_0:
        return np.concatenate([side, (q.astype(np.int16) - 128).astype(np.int8).view(np.uint8)], 1)


def side_streams(ty, side):
    """Side bytes [n, s] -> (raw bits kept per block: fp16 d / dmin, [symbol streams]): the per-sub-block scale
    (and min) fields as their own symbols (Q4_K / Q5_K: 6-bit sc, m; Q6_K: int8 sc; IQ4_XS: 6-bit ls)."""
    if ty in (T.Q4_K, T.Q5_K):
        b = side[:, 4:].reshape(-1, 3, 4)
        d_, m_, md = b[:, 0], b[:, 1], b[:, 2]
        sc = np.concatenate([d_ & 63, (md & 15) | ((d_ >> 2) & 48)], 1)
        mn = np.concatenate([m_ & 63, (md >> 4) | ((m_ >> 2) & 48)], 1)
        return 32, [sc, mn]
    if ty == T.Q6_K:
        return 16, [side[:, :16]]
    if ty == T.IQ4_XS:
        hi = side[:, 2:4].copy().view(np.uint16)[:, :1]
        lo = side[:, 4:8]
        ls = np.concatenate([(lo & 15)[:, :, None], (lo >> 4)[:, :, None]], 2).reshape(-1, 8)
        ls = ls | (((hi >> (2 * np.arange(8, dtype=np.uint16))) & 3).astype(np.uint8) << 4)
        return 16, [ls]
    return 8 * side.shape[1], []


def huff_bits(x):
    """Length-limited Huffman size of a byte stream (one table: 256 x 4-bit lengths)."""
    h = np.bincount(x.ravel(), minlength=256)
    return int((h * limited_lengths(h, 11)).sum()) + 256 * 4


_R = {}


def job(i):
    t = _R["reader"].tensors[i]
    ty = t.tensor_type
    K, N = (int(x) for x in t.shape) if len(t.shape) == 2 else (int(t.shape[0]), 1)
    nbits = 8 * int(t.n_bytes)
    r = dict(name=t.name, type=ty.name, N=N, K=K, packed_bits=nbits, coded_bits=nbits, coded=False)
    keep = len(t.shape) != 2 or (t.name == "token_embd.weight" and not _R["tied"])
    sp = None if keep else split(ty, np.asarray(t.data).reshape(-1, gguf.GGML_QUANT_SIZES[ty][1]))
    if sp is None:
        return r
    q, side, b = sp
    assert np.array_equal(join(ty, q, side), np.asarray(t.data).reshape(q.shape[0], -1)), t.name
    q = q.reshape(N, K)
    sym = (q[:, 0::2] | (q[:, 1::2] << 4)) if b == 4 else q
    blob, _, hist = encode_huf8(np.ascontiguousarray(sym))
    p = hist[hist > 0] / hist.sum()
    r.update(coded=True, code_bits=b, side_bits=8 * side.size, code_packed_bits=b * N * K,
             huf_bits=8 * len(blob), H=float(-(p * np.log2(p)).sum()) / (2 if b == 4 else 1))
    raw, streams = side_streams(ty, side)
    r["side_coded_bits"] = raw * side.shape[0] + sum(huff_bits(x) for x in streams)
    r["coded_bits"] = min(r["side_bits"], r["side_coded_bits"]) + r["huf_bits"]
    assert r["side_bits"] + r["code_packed_bits"] == nbits, t.name
    return r


def main():
    from transformers import AutoConfig
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(SPECS), required=True)
    ap.add_argument("--types", default="Q4_K_S,Q4_K_M,IQ4_XS,Q5_K_M,Q6_K,Q8_0")
    ap.add_argument("--no-imatrix", action="store_true")
    ap.add_argument("--jobs", type=int, default=24)
    a = ap.parse_args()
    spec = SPECS[a.model]
    gdir = os.path.join(spec["dir"], "gguf")
    tied = AutoConfig.from_pretrained(os.path.join(spec["dir"], "hf_fp16")).tie_word_embeddings
    base = "llama.cpp" if a.no_imatrix else "llama.cpp+imatrix"
    a1 = {r["qtype"]: r for r in map(json.loads, open(os.path.join(REPO, "results", "a1_frontier.jsonl")))
          if r["model"] == a.model and r.get("method") == base}
    log = os.path.join(REPO, "results", "logs", "kq", "code_%s.log" % a.model)
    os.makedirs(os.path.dirname(log), exist_ok=True)
    for ty in a.types.split(","):
        ref = a1[ty]
        cfgd = dict(model=a.model, method=base + "+huf", qtype=ty)
        lg = JsonlLogger("a1_frontier", cfg=cfgd)
        t0 = time.time()
        qf = os.path.join(gdir, "%s_code.gguf" % ty)
        qargs = [] if a.no_imatrix else ["--imatrix", os.path.join(gdir, "imatrix.dat")]
        with open(log, "a") as f:
            subprocess.run([os.path.join(LLAMA, "build", "bin", "llama-quantize")] + qargs +
                           [os.path.join(gdir, "f16.gguf"), qf, ty], check=True, stdout=f, stderr=subprocess.STDOUT)
        _R.update(reader=gguf.GGUFReader(qf), tied=tied)
        with Pool(a.jobs) as pool:
            ts = list(pool.imap(job, range(len(_R["reader"].tensors))))
        _R.clear()
        packed = sum(r["packed_bits"] for r in ts)
        assert packed == ref["tensor_bits"], (ty, packed, ref["tensor_bits"])   # same file as the scored A1 row
        coded = sum(r["coded_bits"] for r in ts)
        nw = ref["n_weights"]
        cd = [r for r in ts if r["coded"]]
        by = {}
        for r in cd:
            k = by.setdefault(r["type"], dict(n=0, codes=0, code_bits=0, huf_bits=0, side_bits=0))
            k["n"] += 1
            k["codes"] += r["N"] * r["K"]
            k["code_bits"] += r["code_packed_bits"]
            k["huf_bits"] += r["huf_bits"]
            k["side_bits"] += r["side_bits"]
        rec = dict(cfgd, family="coded", bpw=coded / nw, packed_bpw=packed / nw, n_weights=nw, tensor_bits=coded,
                   packed_tensor_bits=packed, by_type=by, uncoded_bits=sum(r["packed_bits"] for r in ts if not r["coded"]),
                   quality_from=dict(method=base, qtype=ty, config_hash=ref.get("config_hash"),
                                     git_commit=ref.get("git_commit")), hf=spec["hf"])
        for k in ("ppl_full", "llamacpp_ppl", "hellaswag", "gsm8k", "arc_easy", "piqa", "ppl_protocol"):
            if k in ref:
                rec[k] = ref[k]
        rec["seconds"] = time.time() - t0
        lg.log(rec)
        print("%s %s: packed %.4f -> coded %.4f b/w (%.1f%%); %s" % (
            a.model, ty, packed / nw, coded / nw, 100 * (1 - coded / packed),
            ", ".join("%s x%d: %.3f -> %.3f b/code" % (k, v["n"], v["code_bits"] / v["codes"], v["huf_bits"] / v["codes"])
                      for k, v in by.items())), flush=True)
        os.remove(qf)


if __name__ == "__main__":
    main()
