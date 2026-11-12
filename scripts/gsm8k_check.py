"""GSM8K-CoT control: score a GGUF as scripts/kq_quant.py --score-only does (fp16 HF model, lm-eval 8-shot chain of thought) at a
given batch size, keeping every generation, to separate batch-size effects from quality differences between models.

  .venv/bin/python scripts/gsm8k_check.py --gguf PATH --tag NAME --batch 32
Row -> results/gsm8k_check.jsonl, generations -> results/logs/a2/gsm8k_check/<tag>_b<batch>.jsonl (not written to a1_frontier)
"""
import argparse
import json
import os
import sys

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
import a1_gguf  # noqa: E402
from lazyweights.log import JsonlLogger  # noqa: E402
from lazyweights.models import SPECS  # noqa: E402


def main():
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--model", default="phi3")
    a = ap.parse_args()
    hf_dir = os.path.join(SPECS[a.model]["dir"], "hf_fp16")
    hcfg = AutoConfig.from_pretrained(hf_dir)
    tok = AutoTokenizer.from_pretrained(hf_dir)
    with torch.device("cuda"):
        model = AutoModelForCausalLM.from_config(hcfg, dtype=torch.float16).eval()
    W, _, _ = a1_gguf.read_gguf(a.gguf, a.model, hcfg, [n for n, _ in model.named_parameters()])
    a1_gguf.fill(model, W)
    del W
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=a.batch, max_length=2048)
    r = lm_eval.simple_evaluate(model=lm, tasks=["gsm8k_cot"], num_fewshot=8, log_samples=True, random_seed=0,
                                numpy_random_seed=0, torch_random_seed=0, fewshot_random_seed=0)
    m = r["results"]["gsm8k_cot"]
    out = dict(tag=a.tag, batch=a.batch, gguf=a.gguf, strict=float(m["exact_match,strict-match"]),
               flex=float(m["exact_match,flexible-extract"]), stderr=float(m["exact_match_stderr,strict-match"]))
    print(out, flush=True)
    with open(os.path.join(REPO, "results/logs/a2/gsm8k_check/%s_b%d.jsonl" % (a.tag, a.batch)), "w") as f:
        for s in r["samples"]["gsm8k_cot"]:
            f.write(json.dumps(dict(doc_id=s["doc_id"], filter=s.get("filter"), resp=s["resps"], filtered=s["filtered_resps"],
                                    target=s["target"], exact_match=s.get("exact_match")), default=str) + "\n")
    JsonlLogger("gsm8k_check", cfg=vars(a)).log(out)


if __name__ == "__main__":
    main()
