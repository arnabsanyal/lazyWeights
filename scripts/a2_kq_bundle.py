"""A2: runtime bundle (kernels/llmrt.cu) of an option-2 point: a GGUF written by scripts/kq_quant.py (or any llama.cpp
GGUF of the same types). Matmul codes in GGUF block order are split from their side bytes (scripts/kq_code.py split),
coded with one Huffman table per matrix (Q4_K / IQ4_XS: huf4p, two codes per symbol; Q5_K / Q6_K: huf8, one code per
symbol; codes are not re-centered, the grid is the format's own), and the native side bytes of every 256-weight
super-block are kept as they are in a side tensor <name>.kqs (enc kq_<type>, [N, K / 256 x side bytes]):
  Q4_K / Q5_K: fp16 d, fp16 dmin, 12 bytes of 6-bit (sc, m) per 32-weight sub-block: w = d sc q - dmin m
  Q6_K: int8 sc per 16-weight sub-block, fp16 d: w = d sc (q - 32)
  IQ4_XS: fp16 d, 6-bit ls per 32-weight sub-block: w = d (ls - 32) kv[q]
llama-arch q / k rows are un-permuted to the HF order (RoPE rotate-half); q, k, v of one type are one matrix l<i>.qkv,
else three (l<i>.q / .k / .v); phi3 ffn_up (gate | up rows) is split into gate and up. Embedding (a host-side gather):
raw8 codes with f32 embed.rs / embed.rsub per sub-block (exact for Q6_K / IQ4_XS / Q8_0: code = q or kv[q] + 128;
Q4_K / Q5_K: rsub = dmin m / (d sc), equal up to f32 rounding). Every matrix is checked: Huffman decode, and the
dequantization from (codes, side bytes) as the kernels compute it against gguf-py's dequantize (bit-exact).
--packed: also the fixed-rate twin <out>_pk (pk4 / pk8 codes, same side tensors) = the GGUF bytes re-laid out.

  .venv/bin/python scripts/a2_kq_bundle.py --model mistral --gguf IQ4_XS_g1_lam0.gguf [--packed]
-> <model dir>/rt_bundles/<model>_kq_<gguf stem>[_pk]/, rows -> results/llm_bundle.jsonl
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
from lazyweights.paths import LLAMA  # noqa: E402
sys.path.insert(1, os.path.join(LLAMA, "gguf-py"))
import gguf  # noqa: E402
from lazyweights.bundle import Writer, decode_huf8, encode_huf8  # noqa: E402
from lazyweights.log import JsonlLogger  # noqa: E402
from kq_code import split  # noqa: E402
from kq_quant import permute  # noqa: E402
from lazyweights.bundle import ROPE_ROWS, eos_ids  # noqa: E402
from lazyweights.models import SPECS  # noqa: E402

T = gguf.GGMLQuantizationType
BITS = {T.Q4_K: 4, T.IQ4_XS: 4, T.Q5_K: 8, T.Q6_K: 8}   # symbol layout: 4 = two codes per symbol
KV = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], np.float32)
PERMUTED = ("mistral", "smollm")


def f16(b, o):
    return b[:, o:o + 2].copy().view(np.float16).astype(np.float32)[:, 0]


def sub_sm(ty, side):
    """side bytes [n, s] of n super-blocks -> (s, o) [n, sub-blocks] f32, as the kernels form them."""
    if ty in (T.Q4_K, T.Q5_K):
        d, dm, q = f16(side, 0), f16(side, 2), side[:, 4:16].astype(np.int32)
        j = np.arange(4)
        sc = np.concatenate([q[:, j] & 63, (q[:, j + 8] & 15) | ((q[:, j] >> 6) << 4)], 1)
        m = np.concatenate([q[:, j + 4] & 63, (q[:, j + 8] >> 4) | ((q[:, j + 4] >> 6) << 4)], 1)
        return d[:, None] * sc.astype(np.float32), dm[:, None] * m.astype(np.float32)
    if ty == T.Q6_K:
        s = f16(side, 16)[:, None] * side[:, :16].view(np.int8).astype(np.float32)
        return s, np.zeros_like(s)
    if ty == T.IQ4_XS:
        hi = side[:, 2:4].copy().view(np.uint16).astype(np.int32)
        lo = side[:, 4:8].astype(np.int32)
        j = np.arange(8)
        ls = ((lo[:, j // 2] >> (4 * (j % 2))) & 15) | (((hi >> (2 * j)) & 3) << 4)
        s = f16(side, 0)[:, None] * (ls - 32).astype(np.float32)
        return s, np.zeros_like(s)
    raise ValueError(ty)


def fq(ty, q):
    """the grid value of a code: Q4_K / Q5_K: q, Q6_K: q - 32, IQ4_XS: kv[q]"""
    return KV[q] if ty == T.IQ4_XS else q.astype(np.float32) - (32 if ty == T.Q6_K else 0)


def dequant(ty, q, side):
    """kernel arithmetic, f32: w = s f(q) - o per sub-block (q [N, K], side [N, K / 256 x s])"""
    N, K = q.shape
    s, o = sub_sm(ty, side.reshape(N * K // 256, -1))
    g = s.shape[1]
    f = fq(ty, q).reshape(N * K // 256, g, -1)
    return (s[:, :, None] * f - o[:, :, None]).reshape(N, K)


def emb8(ty, t):
    """embedding -> raw8 codes [V, D], rs / rsub [V, D / sub-block] f32 (w = (code - rsub) rs)"""
    D, V = (int(x) for x in t.shape)
    b = np.asarray(t.data).reshape(-1, gguf.GGML_QUANT_SIZES[ty][1])
    if ty == T.Q8_0:
        d = f16(b, 0).reshape(V, -1)
        return (b[:, 2:].view(np.int8).astype(np.int16) + 128).astype(np.uint8).reshape(V, D), d, np.full_like(d, 128)
    q, side, _ = split(ty, b)
    s, o = sub_sm(ty, side)
    if ty == T.IQ4_XS:
        return (KV[q] + 128).astype(np.uint8).reshape(V, D), s.reshape(V, -1), np.full((V, s.size // V), 128, np.float32)
    if ty == T.Q6_K:
        return q.reshape(V, D), s.reshape(V, -1), np.full((V, s.size // V), 32, np.float32)
    ok = s != 0
    z = np.where(ok, o / np.where(ok, s, 1), 0)
    assert np.all(ok | (o == 0)), "sub-block with sc = 0 and m != 0"
    return q.reshape(V, D), s.reshape(V, -1), z.astype(np.float32).reshape(V, -1)


def job(args):
    name, ty, q, side, ref, packed = args
    B = BITS[ty]
    assert np.array_equal(dequant(ty, q, side), ref), name + ": dequantization differs from gguf-py"
    sym = q if B == 8 else (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    sym = np.ascontiguousarray(sym)
    if packed:
        return name, sym.tobytes(), None
    blob, nbits, hist = encode_huf8(sym)
    assert np.array_equal(decode_huf8(blob, *sym.shape), sym), name
    return name, blob, nbits


def main():
    from transformers import AutoConfig
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(SPECS), required=True)
    ap.add_argument("--gguf", required=True, help="file name in <model dir>/gguf/kq/ (or a path)")
    ap.add_argument("--out", default="")
    ap.add_argument("--packed", action="store_true")
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()
    spec = SPECS[a.model]
    path = a.gguf if os.path.sep in a.gguf else os.path.join(spec["dir"], "gguf", "kq", a.gguf)
    stem = os.path.basename(path)[:-5]
    out = a.out or os.path.join(spec["dir"], "rt_bundles", "%s_kq_%s" % (a.model, stem))
    t0 = time.time()
    c = AutoConfig.from_pretrained(os.path.join(spec["dir"], "hf_fp16"))
    r = gguf.GGUFReader(path)
    G = {t.name: t for t in r.tensors}
    L, Hq = c.num_hidden_layers, c.num_attention_heads
    Hk = getattr(c, "num_key_value_heads", Hq)
    Dh = getattr(c, "head_dim", None) or c.hidden_size // Hq
    tied = "output.weight" not in G

    def mat(gn, hq=None):
        t = G[gn]
        ty = t.tensor_type
        assert ty in BITS, (gn, ty.name)
        K, N = (int(x) for x in t.shape)
        q, side, _ = split(ty, np.asarray(t.data).reshape(-1, gguf.GGML_QUANT_SIZES[ty][1]))
        q, side = q.reshape(N, K), side.reshape(N, -1)
        ref = gguf.quants.dequantize(np.asarray(t.data), ty).reshape(N, K)
        if hq is not None and a.model in PERMUTED:   # GGUF rows = HF rows [permute]: back to the HF order
            inv = np.argsort(permute(np.arange(N), hq))
            q, side, ref = q[inv], side[inv], ref[inv]
        return ty, q, side, ref

    order = []
    for i in range(L):
        p = "blk.%d." % i
        if p + "attn_qkv.weight" in G:
            order.append(("l%d.qkv" % i, mat(p + "attn_qkv.weight")))
        else:
            qkv = [mat(p + "attn_q.weight", Hq), mat(p + "attn_k.weight", Hk), mat(p + "attn_v.weight")]
            if len({x[0] for x in qkv}) == 1:
                order.append(("l%d.qkv" % i, (qkv[0][0],) + tuple(np.concatenate(x) for x in list(zip(*qkv))[1:])))
            else:
                order += [("l%d.%s" % (i, n), x) for n, x in zip("qkv", qkv)]
        order.append(("l%d.o" % i, mat(p + "attn_output.weight")))
        if p + "ffn_gate.weight" in G:
            order += [("l%d.gate" % i, mat(p + "ffn_gate.weight")), ("l%d.up" % i, mat(p + "ffn_up.weight"))]
        else:   # phi3: ffn_up = [gate; up]
            ty, q, side, ref = mat(p + "ffn_up.weight")
            I = q.shape[0] // 2
            order += [("l%d.gate" % i, (ty, q[:I], side[:I], ref[:I])), ("l%d.up" % i, (ty, q[I:], side[I:], ref[I:]))]
        order.append(("l%d.down" % i, mat(p + "ffn_down.weight")))
    order.append(("lm_head", mat("token_embd.weight" if tied else "output.weight")))
    et = G["token_embd.weight"]
    eq, es, ez = emb8(et.tensor_type, et)
    eref = gguf.quants.dequantize(np.asarray(et.data), et.tensor_type).reshape(eq.shape)
    sub = eq.shape[1] // es.shape[1]
    edq = (eq.astype(np.float32) - np.repeat(ez, sub, 1)) * np.repeat(es, sub, 1)
    emb_err = float(np.abs(edq - eref).max())
    print("embed %s: max |w - gguf| %.3g" % (et.tensor_type.name, emb_err), flush=True)
    norms = {"lnf": np.asarray(G["output_norm.weight"].data, np.float32)}
    for i in range(L):
        norms["l%d.ln1" % i] = np.asarray(G["blk.%d.attn_norm.weight" % i].data, np.float32)
        norms["l%d.ln2" % i] = np.asarray(G["blk.%d.ffn_norm.weight" % i].data, np.float32)
    theta = getattr(c, "rope_theta", None) or c.rope_parameters["rope_theta"]
    inv = 1.0 / (theta ** (np.arange(0, Dh, 2, dtype=np.float64) / Dh))
    ang = np.arange(ROPE_ROWS, dtype=np.float64)[:, None] * inv[None]
    for packed in ([False, True] if a.packed else [False]):
        o = out + ("_pk" if packed else "")
        lg = JsonlLogger("llm_bundle", cfg=dict(model=a.model, gguf=path, out=o, packed=packed))
        w = Writer(o)
        cfg = dict(n_layers=L, hidden=c.hidden_size, n_heads=Hq, n_kv_heads=Hk, head_dim=Dh, inter=c.intermediate_size,
                   vocab=c.vocab_size, eps=c.rms_norm_eps, rope_rows=ROPE_ROWS, rope_half=Dh // 2, attn_scale=Dh ** -0.5,
                   enc="kq_pk" if packed else "kq_huf", tied=int(tied), eos=eos_ids(spec["hf"], c))
        for k, v in cfg.items():
            w.cfg(k, v)

        def f32(name, v):
            v = np.asarray(v, np.float32)
            v = v.reshape(v.shape[0], -1) if v.ndim == 2 else v.reshape(1, -1)
            w.add(name, "f32", v.shape[0], v.shape[1], 1.0, 0, v.tobytes())

        w.add("embed", "raw8", eq.shape[0], eq.shape[1], 1.0, 0, eq.tobytes())
        f32("embed.rs", es)
        f32("embed.rsub", ez)
        f32("cos", np.cos(ang))
        f32("sin", np.sin(ang))
        for i in range(L):
            f32("l%d.ln1" % i, norms["l%d.ln1" % i])
            f32("l%d.ln2" % i, norms["l%d.ln2" % i])
        f32("lnf", norms["lnf"])
        code_b = side_b = n_w = 0
        with Pool(a.jobs) as pool:
            res = pool.imap(job, [(nm, ty, q, side, ref, packed) for nm, (ty, q, side, ref) in order])
            for (nm, (ty, q, side, ref)), (_, blob, nbits) in zip(order, res):
                N, K = q.shape
                B = BITS[ty]
                enc = ("pk8" if B == 8 else "pk4") if packed else ("huf8" if B == 8 else "huf4p")
                w.add(nm, enc, N, K, 1.0, 0, blob)
                w.add(nm + ".kqs", "kq_" + ty.name, N, side.shape[1], 1.0, 0, np.ascontiguousarray(side).tobytes())
                code_b += 8 * len(blob)
                side_b += 8 * side.size
                n_w += N * K
                lg.log(dict(step="tensor", name=nm, N=N, K=K, type=ty.name, enc=enc, blob_bytes=len(blob),
                            side_bytes=int(side.size), bpw=(8 * len(blob) + 8 * side.size) / (N * K)))
        w.close()
        size = os.path.getsize(os.path.join(o, "weights.bin"))
        rec = dict(step="summary", cfg=cfg, gguf=path, matmul_params=n_w, code_bpw=code_b / n_w,
                   side_bpw=side_b / n_w, matmul_bpw=(code_b + side_b) / n_w, bundle_bytes=size,
                   embed_type=et.tensor_type.name, embed_max_abs_err=emb_err, seconds=time.time() - t0)
        lg.log(rec)
        print(rec, flush=True)


if __name__ == "__main__":
    main()
