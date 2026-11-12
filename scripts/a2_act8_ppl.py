"""Accuracy cost of int8 activations in the decode GEMV (kernels/llmrt.cu k_gemvm_kq8): WikiText-2 PPL with the
paper's protocol (independent 512-token chunks) but each chunk fed through the KV cache in steps of --step tokens
(<= 8: the decode / speculative-verify kernels, not the prefill GEMM), with int8 activations (default) and with
LLMRT_FP_ACT=1 (fp32 activations).

  .venv/bin/python scripts/a2_act8_ppl.py --lib kernels/build/libllmrt_kq4.so --bundle DIR --hf REPO [--fp-act]
-> results/a2_act8_ppl.jsonl
  .venv/bin/python scripts/a2_act8_ppl.py --table   (int8 vs fp32 per bundle -> results/a2_act8_ppl.md)
"""
import argparse
import ctypes
import json
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from lazyweights.log import JsonlLogger  # noqa: E402


def table():
    last = {}
    for l in open(os.path.join(REPO, "results", "a2_act8_ppl.jsonl")):
        r = json.loads(l)
        last[(os.path.basename(r["bundle"]), r["act"], r["step"])] = r["ppl"]
    out = ["# Decode-path PPL: int8 vs fp32 activations", "",
           "WikiText-2, independent 512-token chunks fed through the KV cache in steps of 8 (decode / verify GEMV; the "
           "paper's PPL protocol uses the prefill GEMM, which is unchanged). Packed twins (coded bundles give identical "
           "logits).", "", "| bundle | step | fp32 | int8 | int8 vs fp32 |", "|---|---|---|---|---|"]
    for (b, step) in sorted({(k[0], k[2]) for k in last}):
        f, q = last.get((b, "fp32", step)), last.get((b, "int8", step))
        out.append("| %s | %d | %s | %s | %s |" % (b, step, "%.4f" % f if f else "-", "%.4f" % q if q else "-",
                                                   "%+.2f%%" % (100 * (q / f - 1)) if f and q else "-"))
    open(os.path.join(REPO, "results", "a2_act8_ppl.md"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


def main():
    if "--table" in sys.argv:
        return table()
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=os.path.join(REPO, "kernels", "build", "libllmrt_kq4.so"))
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--step", type=int, default=8)
    ap.add_argument("--fp-act", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=0)
    a = ap.parse_args()
    if a.fp_act:
        os.environ["LLMRT_FP_ACT"] = "1"
    from datasets import load_dataset
    from transformers import AutoTokenizer
    L = ctypes.CDLL(a.lib)
    L.rt_open.restype = ctypes.c_void_p
    L.rt_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_longlong, ctypes.c_int, ctypes.c_int]
    L.rt_forward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    L.rt_reset.argtypes = [ctypes.c_void_p]
    L.rt_vocab.argtypes = [ctypes.c_void_p]
    h = L.rt_open(a.bundle.encode(), 512, 128, -1, 384, 1)
    V = L.rt_vocab(h)
    tok = AutoTokenizer.from_pretrained(a.hf)
    text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = np.asarray(tok(text)["input_ids"], np.int32)
    if a.max_tokens:
        ids = ids[:a.max_tokens]
    t0 = time.time()
    nll, cnt = 0.0, 0
    lg = np.zeros((a.step, V), np.float32)
    for i in range(0, len(ids), 512):
        x = np.ascontiguousarray(ids[i:i + 512])
        L.rt_reset(h)
        for s in range(0, len(x), a.step):
            c = np.ascontiguousarray(x[s:s + a.step])
            L.rt_forward(h, c.ctypes.data, len(c), lg.ctypes.data, 1)
            z = lg[:len(c)].astype(np.float64)
            z -= z.max(1, keepdims=True)
            lp = z - np.log(np.exp(z).sum(1, keepdims=True))
            tgt = x[s + 1:s + 1 + len(c)]
            nll -= float(lp[np.arange(len(tgt)), tgt].sum())
            cnt += len(tgt)
    r = dict(bundle=a.bundle, lib=os.path.basename(a.lib), act="fp32" if a.fp_act else "int8", step=a.step,
             n_tokens=len(ids), n_scored=cnt, ppl=float(np.exp(nll / cnt)), seconds=time.time() - t0)
    print(r, flush=True)
    JsonlLogger("a2_act8_ppl", cfg=dict(bundle=os.path.basename(a.bundle), act=r["act"], step=a.step)).log(r)


if __name__ == "__main__":
    main()
