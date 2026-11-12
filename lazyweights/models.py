"""The three models of the paper and how their fp16 starting weights are built.

The starting point of every quantizer (ours and llama.cpp's) is an fp16 Hugging Face checkpoint whose weights are the
fp16 initializers of each model's official ONNX release (`scripts/prepare_models.py` downloads the ONNX graph, fills
the Hugging Face module from it and writes `<LW_MODELS>/<dir>/hf_fp16/`). The ONNX weight files are byte-identical to
the ones the paper used; the official safetensors checkpoints differ from them by up to 1 ulp (bf16 -> fp16 rounding),
so loading those instead gives slightly different numbers.
"""
import os

import torch

from lazyweights.paths import MODELS

SPECS = {
    "mistral": dict(hf="mistralai/Mistral-7B-Instruct-v0.2", dir=os.path.join(MODELS, "mistral-7b-instruct-v0.2"),
                    onnx_repo="microsoft/mistral-7b-instruct-v0.2-ONNX",
                    onnx_dir="onnx/cuda/mistral-7b-instruct-v0.2-cuda-fp16",
                    src="mistral-7b-instruct-v0.2-cuda-fp16.onnx", data=".data",
                    sha256="5ec8b4ae3b29132428961adea54f039f8805cb06535a3c3050bd3633ef150d64"),
    "smollm": dict(hf="HuggingFaceTB/SmolLM-1.7B-Instruct", dir=os.path.join(MODELS, "smollm-1-7b-instruct"),
                   onnx_repo="HuggingFaceTB/SmolLM-1.7B-Instruct", onnx_dir="onnx", src="model_fp16.onnx", data="_data",
                   sha256="55c5540bd70fc453798b6ef84721bce8a9dd8dc3304fb57b526cb3de8f7313c6"),
    # phi3: HF fuses qkv and gate / up (gate_up_proj = [gate; up] rows); the ONNX has qkv fused, gate / up separate.
    "phi3": dict(hf="microsoft/Phi-3-mini-4k-instruct", dir=os.path.join(MODELS, "phi3-mini-4k-instruct"),
                 onnx_repo="microsoft/Phi-3-mini-4k-instruct-onnx", onnx_dir="cuda/cuda-fp16",
                 src="phi3-mini-4k-instruct-cuda-fp16.onnx", data=".data", fused=True,
                 sha256="292f4f035debccf72724ab5129fb1ff1ab127929f68ce3117ae18ca8cfd8bc1a"),
}
LIN = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LIN_FUSED = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]


def hf_dir(spec):
    return os.path.join(spec["dir"], "hf_fp16")


def build(spec):
    """(fp16 HF module on the GPU, tokenizer, None) from <dir>/hf_fp16 (scripts/prepare_models.py).

    The module is built on the GPU from its config and then filled, as in the paper's runs: buffers such as the rotary
    inverse frequencies are computed on the GPU, which differs in the last bit from from_pretrained (CPU, then moved)
    and is enough to change GPTQ's choices."""
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    d = hf_dir(spec)
    if not os.path.exists(os.path.join(d, "model.safetensors")):
        raise FileNotFoundError("%s missing: run scripts/prepare_models.py --model <name> first" % d)
    tok = AutoTokenizer.from_pretrained(d)
    cfg = AutoConfig.from_pretrained(d)
    with torch.device("cuda"):
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float16).eval()
    missing, unexpected = model.load_state_dict(load_file(os.path.join(d, "model.safetensors"), device="cuda"),
                                                strict=False)
    assert not unexpected and set(missing) <= ({"lm_head.weight"} if cfg.tie_word_embeddings else set()), \
        (missing, unexpected)
    if cfg.tie_word_embeddings:
        assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    return model, tok, None


def weights_sha256(model):
    """Hash of the fp16 state dict (names and bytes), as recorded in SPECS[...]['sha256']."""
    import hashlib
    h = hashlib.sha256()
    with torch.no_grad():
        for n, p in sorted(model.state_dict().items()):
            h.update(n.encode())
            h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
