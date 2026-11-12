"""A4 coder ablation, rate side: Huffman (our bundle format: length-limited 11-bit codes, pair symbols for 4-bit codes,
one table per matrix, lane offsets) vs static rANS with the same symbols, tiles and lanes (hw/rans/rtl/rans_lane.sv:
12-bit probabilities, 32-bit state, 16-bit renormalization) on the coded matrices of the 6 headline points' GGUFs.

rANS size per matrix (exact for the payload model, not an encoder run): sum over symbols of -log2(f_s / 4096) with
f the 12-bit quantized histogram (proportional, at least 1, sum fixed on the largest), + 32 state bits and on average
8 bits of 16-bit word padding per lane stream (64 x 256 tile: 32 lane streams), + the table (256 x 12-bit
frequencies), + lane offsets (32 x 32 bits per tile, as Huffman). Huffman = the bundle size (scripts/kq_code.py
encode_huf8: payload + 2048-entry LUT + lane offsets). Entropy H = the zeroth-order symbol entropy.

  .venv/bin/python scripts/a4_rans_rate.py   -> results/a4_rans_rate.jsonl, printed per point (bits per code)
"""
import os
import sys
from multiprocessing import Pool

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
import kq_code  # noqa: E402
from kq_code import encode_huf8, gguf, split  # noqa: E402
from lazyweights.log import JsonlLogger  # noqa: E402

from lazyweights.paths import MODELS as NAS  # noqa: E402
POINTS = [   # (model, budget, GGUF)
    ("phi3", "1%", "phi3-mini-4k-instruct/gguf/kq/IQ4_XS_g0_lam0_c1.25.gguf"),
    ("phi3", "3.5%", "phi3-mini-4k-instruct/gguf/kq/IQ4_XS_g1_lam0_c1.25.gguf"),
    ("mistral", "1%", "mistral-7b-instruct-v0.2/gguf/kq/IQ4_XS_g1_lam0.gguf"),
    ("mistral", "3.5%", "mistral-7b-instruct-v0.2/gguf/kq/IQ4_XS_g1_lam0_c1.5.gguf"),
    ("smollm", "1%", "smollm-1-7b-instruct/gguf/kq/Q5_K_M_g1_lam0.gguf"),
    ("smollm", "3.5%", "smollm-1-7b-instruct/gguf/kq/IQ4_XS_g1_lam0.gguf"),
]
PB = 12
_R = {}


def qfreq(h):
    f = np.where(h > 0, np.maximum(1, np.round(h / h.sum() * (1 << PB))), 0).astype(np.int64)
    f[np.argmax(f)] += (1 << PB) - f.sum()
    return f


def job(i):
    t = _R["reader"].tensors[i]
    ty = t.tensor_type
    if len(t.shape) != 2 or (t.name == "token_embd.weight" and not _R["tied"]):
        return None
    K, N = (int(x) for x in t.shape)
    sp = split(ty, np.asarray(t.data).reshape(-1, gguf.GGML_QUANT_SIZES[ty][1]))
    if sp is None:
        return None
    q, _, b = sp
    q = q.reshape(N, K)
    sym = (q[:, 0::2] | (q[:, 1::2] << 4)) if b == 4 else q
    blob, _, hist = encode_huf8(np.ascontiguousarray(sym))
    h = np.asarray(hist, np.int64)
    p = h[h > 0] / h.sum()
    f = qfreq(h)
    nz = h > 0
    payload = float((h[nz] * -np.log2(f[nz] / float(1 << PB))).sum())
    T = (N // 64) * (sym.shape[1] // 256)
    rans = payload + T * 32 * (32 + 8) + 256 * PB + T * 32 * 32
    rans1 = rans   # single-code rANS (A = 16, 4-bit codes only): one symbol per code, same tiles and lanes
    if b == 4:
        h1 = np.bincount(q.ravel(), minlength=16).astype(np.int64)
        f1 = qfreq(h1)
        nz1 = h1 > 0
        rans1 = float((h1[nz1] * -np.log2(f1[nz1] / float(1 << PB))).sum()) + T * 32 * (32 + 8) + 16 * PB + T * 32 * 32
    return dict(name=t.name, type=ty.name, N=N, K=K, code_bits=b, codes=N * K, huf_bits=8 * len(blob),
                rans_bits=rans, rans1_bits=rans1, H_bits=float(-(p * np.log2(p)).sum()) * h.sum())


def main():
    lg = JsonlLogger("a4_rans_rate", cfg=dict(prob_bits=PB, state_bits=32, renorm_bits=16))
    print("| model | budget | entropy (b/code) | Huffman (b/code) | rANS pairs (b/code) | vs Huffman | "
          "rANS single 4-bit codes (b/code) | vs Huffman |")
    print("|---|---|---|---|---|---|---|---|")
    for m, bud, g in POINTS:
        r = gguf.GGUFReader(os.path.join(NAS, g))
        names = {t.name for t in r.tensors}
        _R.update(reader=r, tied="output.weight" not in names)
        with Pool(24) as pool:
            ts = [x for x in pool.imap(job, range(len(r.tensors))) if x]
        _R.clear()
        n = sum(x["codes"] for x in ts)
        H, hb, rb, r1 = (sum(x[k] for x in ts) / n for k in ("H_bits", "huf_bits", "rans_bits", "rans1_bits"))
        lg.log(dict(model=m, budget=bud, gguf=g, matrices=len(ts), codes=n, H_bits_per_code=H,
                    huf_bits_per_code=hb, rans_bits_per_code=rb, rans_vs_huf=rb / hb - 1,
                    rans1_bits_per_code=r1, rans1_vs_huf=r1 / hb - 1))
        print("| %s | <=%s | %.4f | %.4f | %.4f | %+.2f%% | %.4f | %+.2f%% |" % (
            m, bud, H, hb, rb, 100 * (rb / hb - 1), r1, 100 * (r1 / hb - 1)), flush=True)


if __name__ == "__main__":
    main()
