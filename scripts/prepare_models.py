"""Build the fp16 starting checkpoint of a model: download its official fp16 ONNX release from Hugging Face, fill the
Hugging Face module (built from the model's config) with the ONNX initializers, check the weights against the hash the
paper's runs recorded, and write <LW_MODELS>/<dir>/hf_fp16/ (safetensors + config + tokenizer).

Every quantizer in this repository (ours and llama.cpp's, through convert_hf_to_gguf.py) starts from this checkpoint.

  python scripts/prepare_models.py --model smollm        # also: phi3, mistral
  python scripts/prepare_models.py --model phi3 --onnx /path/to/phi3-mini-4k-instruct-cuda-fp16.onnx   (already downloaded)
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from lazyweights.models import SPECS, hf_dir, weights_sha256  # noqa: E402


def load(inits, name, base):
    import onnx
    from onnx import numpy_helper
    i = inits[name]
    if not i.external_data:
        return numpy_helper.to_array(i)
    e = {x.key: x.value for x in i.external_data}      # read directly: the .data file may be a symlink
    raw = np.fromfile(os.path.join(base, e["location"]), np.uint8, int(e["length"]), offset=int(e.get("offset", 0)))
    return raw.view(onnx.helper.tensor_dtype_to_np_dtype(i.data_type)).reshape(list(i.dims))


def hf_name(node_name):
    """'/model/layers.0/attn/q_proj/MatMul' -> 'model.layers.0.self_attn.q_proj'; '/lm_head/MatMul' -> 'lm_head'."""
    n = node_name.strip("/").rsplit("/MatMul", 1)[0].replace("/", ".")
    return n.replace(".attn.", ".self_attn.")


def onnx_weights(src):
    """{hf parameter name: (onnx initializer name, [N, K] or [V, D] fp16 numpy)} for every quantizable / copied
    tensor; tied lm_head (MatMul on Transpose(embed)) is reported as 'lm_head.weight' -> the embedding initializer."""
    import onnx
    m = onnx.load(src, load_external_data=False)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    base = os.path.dirname(os.path.abspath(src))
    prod = {o: n for n in g.node for o in n.output}
    out, L = {}, 0
    for n in g.node:
        if n.op_type == "MatMul":
            b = n.input[1]
            if b in inits and len(inits[b].dims) == 2:
                out[hf_name(n.name) + ".weight"] = (b, load(inits, b, base).T)
            elif b in prod and prod[b].op_type == "Transpose" and prod[b].input[0] in inits:   # tied lm_head
                out[hf_name(n.name) + ".weight"] = (prod[b].input[0], None)
        elif n.op_type == "Gather" and n.input[0] in inits and len(inits[n.input[0]].dims) == 2:
            out["model.embed_tokens.weight"] = (n.input[0], load(inits, n.input[0], base))
    for name, i in inits.items():
        if name.startswith("model.layers.") and "layernorm.weight" in name:
            li = int(name.split(".")[2])
            L = max(L, li)
            hn = name if "final_norm" not in name else "model.norm.weight"
            out[hn] = (name, load(inits, name, base))
        elif name == "model.norm.weight":
            out[name] = (name, load(inits, name, base))
    return out


def build_from_onnx(spec, onnx_path):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    cfg = AutoConfig.from_pretrained(spec["hf"])
    tok = AutoTokenizer.from_pretrained(spec["hf"])
    with torch.device("cuda"):
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float16).eval()
    W = onnx_weights(onnx_path)
    params = dict(model.named_parameters())
    tied = cfg.tie_word_embeddings
    if spec.get("fused"):
        for i in range(cfg.num_hidden_layers):
            p = "model.layers.%d.mlp." % i
            g, u = W.pop(p + "gate_proj.weight"), W.pop(p + "up_proj.weight")
            W[p + "gate_up_proj.weight"] = (None, np.concatenate([g[1], u[1]]))
    with torch.no_grad():
        for hn, (on, w) in W.items():
            if w is None:
                assert tied and hn == "lm_head.weight" and on == W["model.embed_tokens.weight"][0], hn
                continue
            p = params.pop(hn)
            assert tuple(p.shape) == w.shape, (hn, p.shape, w.shape)
            p.copy_(torch.from_numpy(np.ascontiguousarray(w)))
    assert not params, "not filled from ONNX: %s" % sorted(params)[:8]
    if tied:
        assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    return model, tok, W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(SPECS), required=True)
    ap.add_argument("--onnx", help="local copy of the ONNX graph (its external data file next to it)")
    ap.add_argument("--force", action="store_true", help="write even if the weight hash differs")
    a = ap.parse_args()
    spec = SPECS[a.model]
    src = a.onnx
    if not src:
        from huggingface_hub import hf_hub_download
        sub = spec["onnx_dir"]
        src = hf_hub_download(spec["onnx_repo"], spec["src"], subfolder=sub)
        hf_hub_download(spec["onnx_repo"], spec["src"] + spec["data"], subfolder=sub)
    model, tok, _ = build_from_onnx(spec, src)
    h = weights_sha256(model)
    print("weights sha256 %s (%s)" % (h, "matches the paper" if h == spec["sha256"] else "DIFFERS from the paper"))
    if h != spec["sha256"] and not a.force:
        sys.exit("weight hash differs from the paper's checkpoint; rerun with --force to write it anyway")
    out = hf_dir(spec)
    os.makedirs(out, exist_ok=True)
    from safetensors.torch import save_file
    sd = {n: p.detach().cpu().contiguous() for n, p in model.state_dict().items()}
    if model.config.tie_word_embeddings:
        sd.pop("lm_head.weight", None)
    save_file(sd, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    model.config.save_pretrained(out)
    model.generation_config.save_pretrained(out)
    tok.save_pretrained(out)
    with open(os.path.join(out, "source.json"), "w") as f:
        json.dump(dict(model=a.model, hf=spec["hf"], onnx_repo=spec["onnx_repo"], onnx=spec["src"],
                       weights_sha256=h), f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
