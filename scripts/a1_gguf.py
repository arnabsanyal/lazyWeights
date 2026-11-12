"""A1: llama.cpp fixed-rate baselines on the same fp16 weights as our points (scripts/a1_hf_export.py checkpoint):
convert_hf_to_gguf.py (f16) -> llama-quantize TYPE -> read the GGUF back (llama.cpp gguf-py), dequantize every tensor
(gguf.quants.dequantize), undo the Q / K row permutation of the llama converter, fill the HF module and score it with
scripts/a1_point.py's protocol (WikiText-2 whole test set in 512-token chunks, lm-eval tasks). Size: the GGUF tensor
bytes (all tensors: quantized matrices, embedding, f32 norms) over the same weight count as a1_point. Also runs
llama-perplexity on the quantized file (its own protocol: -c 512, scores the second half of each chunk) as a sanity
check. Quantized GGUFs are deleted after scoring; the f16 GGUF is kept until --cleanup.
--imatrix: llama-imatrix on the f16 GGUF over our GPTQ calibration set (q4_awq.calib: 128 x 512 tokens of C4
validation, decoded to text), every type quantized with it (method llama.cpp+imatrix).
The f16 GGUF read back must equal the fp16 checkpoint bit for bit (checks the name map and the un-permutation).

  .venv/bin/python scripts/a1_gguf.py --model mistral --types Q4_K_M,Q5_K_M,Q8_0,IQ4_XS,Q4_K_S,Q3_K_M,Q6_K
  .venv/bin/python scripts/a1_gguf.py --model mistral --imatrix --types IQ3_XXS,IQ3_M,Q3_K_M,IQ4_XS,Q4_K_S,Q4_K_M,Q5_K_M,Q6_K
-> results/a1_frontier.jsonl (family packed, method llama.cpp)
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LLAMA = os.path.abspath(os.environ.get("LW_LLAMA_CPP", os.path.join(REPO, "third_party", "llama.cpp")))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(1, os.path.join(LLAMA, "gguf-py"))
import gguf  # noqa: E402
from lazyweights.log import JsonlLogger  # noqa: E402
from lazyweights.evaltasks import tasks  # noqa: E402
from lazyweights.models import SPECS  # noqa: E402
from lazyweights.quant import calib, ppl  # noqa: E402

ARCH = {"mistral": gguf.MODEL_ARCH.LLAMA, "smollm": gguf.MODEL_ARCH.LLAMA, "phi3": gguf.MODEL_ARCH.PHI3}
PERMUTED = {"mistral", "smollm"}   # llama converter permutes q / k rows for its rotary layout


def unpermute(w, n_head):
    """Inverse of conversion/llama.py LlamaModel.permute (n_head already the kv head count for k)."""
    return w.reshape(n_head, w.shape[0] // n_head // 2, 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def read_gguf(path, model_key, cfg, hf_names):
    """GGUF -> ({hf name: fp32 numpy}, total tensor bits, {gguf type name: n tensors})."""
    r = gguf.GGUFReader(path)
    tmap = gguf.get_tensor_name_map(ARCH[model_key], cfg.num_hidden_layers)
    to_hf = {}
    for hn in hf_names:
        g = tmap.get_name(hn, try_suffixes=(".weight", ".bias"))
        if g is not None:
            to_hf[g] = hn
    out, bits, types = {}, 0, {}
    for t in r.tensors:
        bits += 8 * int(t.n_bytes)
        types[t.tensor_type.name] = types.get(t.tensor_type.name, 0) + 1
        if t.name not in to_hf:
            raise KeyError("unmapped GGUF tensor %s" % t.name)
        shape = tuple(int(x) for x in reversed(t.shape))          # ggml ne order -> row-major
        w = gguf.quants.dequantize(t.data, t.tensor_type).reshape(shape).astype(np.float32)
        hn = to_hf[t.name]
        if model_key in PERMUTED and hn.endswith("q_proj.weight"):
            w = unpermute(w, cfg.num_attention_heads)
        elif model_key in PERMUTED and hn.endswith("k_proj.weight"):
            w = unpermute(w, cfg.num_key_value_heads)
        out[hn] = w
    return out, bits, types


def fill(model, W):
    params = dict(model.named_parameters())
    with torch.no_grad():
        for hn, w in W.items():
            p = params[hn]
            assert tuple(p.shape) == w.shape, (hn, p.shape, w.shape)
            p.copy_(torch.from_numpy(w).to(p.dtype))
    missing = [n for n in params if n not in W and not (n == "lm_head.weight" and model.config.tie_word_embeddings)]
    assert not missing, missing[:8]


def n_weights(model):
    """Same denominator as a1_point: MatMul weights (incl. lm_head) + untied embedding."""
    tied = model.config.tie_word_embeddings
    return sum(p.numel() for n, p in model.named_parameters()
               if p.dim() == 2 and not (tied and n == "lm_head.weight"))


def run(cmd, log):
    print("+", " ".join(cmd), flush=True)
    with open(log, "a") as f:
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)


def main():
    from datasets import load_dataset
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(SPECS), required=True)
    ap.add_argument("--types", default="Q4_K_M,Q5_K_M,Q8_0,IQ4_XS,Q4_K_S,Q3_K_M,Q6_K")
    ap.add_argument("--tasks", default="ppl,hellaswag,gsm8k,arc_easy,piqa")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--cleanup", action="store_true", help="delete the f16 GGUF at the end")
    ap.add_argument("--imatrix", action="store_true", help="importance matrix from our calibration set")
    a = ap.parse_args()
    spec = SPECS[a.model]
    hf_dir = os.path.join(spec["dir"], "hf_fp16")
    gdir = os.path.join(spec["dir"], "gguf")
    os.makedirs(gdir, exist_ok=True)
    log = os.path.join(REPO, "results", "logs", "a1", "gguf_%s.log" % a.model)
    os.makedirs(os.path.dirname(log), exist_ok=True)
    f16 = os.path.join(gdir, "f16.gguf")
    py = sys.executable
    if not os.path.exists(f16):
        run([py, os.path.join(LLAMA, "convert_hf_to_gguf.py"), hf_dir, "--outtype", "f16", "--outfile", f16], log)
    cfg = AutoConfig.from_pretrained(hf_dir)
    tok = AutoTokenizer.from_pretrained(hf_dir)
    with torch.device("cuda"):
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float16).eval()
    hf_names = [n for n, _ in model.named_parameters()]
    # round trip: f16 GGUF == the fp16 checkpoint, bit for bit
    from safetensors.numpy import load_file
    ref = load_file(os.path.join(hf_dir, "model.safetensors"))
    W, _, _ = read_gguf(f16, a.model, cfg, hf_names)
    bad = [n for n, w in W.items() if not np.array_equal(w.astype(np.float16), ref[n].astype(np.float16))]
    assert not bad, "f16 GGUF round trip differs: %s" % bad[:6]
    print("f16 GGUF round trip exact (%d tensors)" % len(W), flush=True)
    del ref
    text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    raw = os.path.join(gdir, "wiki.test.raw")
    if not os.path.exists(raw):
        with open(raw, "w") as f:
            f.write(text)
    ids = torch.tensor(tok(text)["input_ids"])
    names = [t for t in a.tasks.split(",") if t]
    method, qargs = "llama.cpp", []
    if a.imatrix:
        method, imat = "llama.cpp+imatrix", os.path.join(gdir, "imatrix.dat")
        if not os.path.exists(imat):
            cal, src = calib(tok, 128, 512)
            ctext = os.path.join(gdir, "calib_c4.txt")
            with open(ctext, "w") as f:
                f.write("\n\n".join(tok.decode(c) for c in cal.tolist()))
            print("calibration text: %s (%s)" % (ctext, src), flush=True)
            run([os.path.join(LLAMA, "build", "bin", "llama-imatrix"), "-m", f16, "-f", ctext, "-o", imat,
                 "-c", "512", "-ngl", "99"], log)
        qargs = ["--imatrix", imat]
    for ty in a.types.split(","):
        cfgd = dict(model=a.model, method=method, qtype=ty, tasks=a.tasks)
        lg = JsonlLogger("a1_frontier", cfg=cfgd)
        t0 = time.time()
        qf = os.path.join(gdir, "%s%s.gguf" % (ty, "_imat" if a.imatrix else ""))
        run([os.path.join(LLAMA, "build", "bin", "llama-quantize")] + qargs + [f16, qf, ty], log)
        W, bits, types = read_gguf(qf, a.model, cfg, hf_names)
        fill(model, W)
        del W
        nw = n_weights(model)
        rec = dict(cfgd, family="packed", bpw=bits / nw, n_weights=nw, tensor_bits=bits, gguf_types=types,
                   file_bytes=os.path.getsize(qf), hf=spec["hf"],
                   ppl_protocol="wikitext2 test, all tokens, 512-token chunks")
        print("%s: %.4f b/w %s" % (ty, rec["bpw"], types), flush=True)
        if "ppl" in names:
            rec["ppl_full"] = ppl(model, ids)
            print("ppl_full %.4f" % rec["ppl_full"], flush=True)
            out = subprocess.run([os.path.join(LLAMA, "build", "bin", "llama-perplexity"), "-m", qf, "-f", raw,
                                  "-c", "512", "-ngl", "99", "-b", "512"], capture_output=True, text=True)
            m = re.findall(r"Final estimate: PPL = ([0-9.]+)", out.stdout + out.stderr)
            rec["llamacpp_ppl"] = float(m[-1]) if m else None
            print("llama-perplexity %s" % rec["llamacpp_ppl"], flush=True)
        rec.update(tasks(model, tok, [t for t in names if t != "ppl"], a.batch))
        rec["seconds"] = time.time() - t0
        lg.log(rec)
        os.remove(qf)
    if a.cleanup:
        os.remove(f16)


if __name__ == "__main__":
    main()
