"""Speed and energy of the headline points with and without prompt-lookup speculation, on a Jetson (tegrastats
module input rail): our runtime (scripts/device/orin_spec_energy.sh, nano_kq_driver.sh: kernels/llmrt.cu `spec`,
plain greedy vs prompt lookup draft 7 / ngram 3, coded bundle and its packed twin) and llama.cpp llama-lookup on the
same GGUF (kq_*) and on the llama.cpp comparator (cmp_*), spec n_max 0 (plain) vs 7, same 6 prompts, 128 new tokens.

llama.cpp runs load the model each time, so a run's decode window is located from its log (time stamp of the
"decoded N tokens in S seconds" line, relative to the run start in marks.txt) and power is averaged over it.

  .venv/bin/python scripts/a2_spec_energy.py --device orin results/a2_orin/energy_kq
  [--ours results/a2_orin/energy_kq4 --tag act8]   (our runtime's runs from another root: int8-activation build)
  Nano Mistral from the USB stick: --device nano --models mistral --ours results/a2_nano/energy_stream_usb10
    --llama-suffix _usb --tag usb10 results/a2_nano/energy_kq   (streamed bundles <point>_ilc count as <point>)
Rows -> results/a2_spec_energy.jsonl; table -> results/a2_spec_energy_<device>[_<tag>].md
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
from energy_analyze import GPU, TOTAL, mean_in, samples  # noqa: E402
from lazyweights.log import JsonlLogger  # noqa: E402

HEAD = {   # model -> [(budget, our kq point, llama.cpp comparator GGUF stem)]
    "phi3": [("1%", "phi3_kq_IQ4_XS_ref", "cmp_IQ4_XS_im"), ("3.5%", "phi3_kq_IQ4_XS_g1_lam0_c1.25", "cmp_Q3_K_M_im")],
    "mistral": [("1%", "mistral_kq_IQ4_XS_g1_lam0", "cmp_Q5_K_M_no"),
                ("3.5%", "mistral_kq_IQ4_XS_g1_lam0_c1.5", "cmp_IQ3_M_im")],
    "smollm": [("1%", "smollm_kq_Q5_K_M_g1_lam0", "cmp_Q5_K_M_no"),
               ("3.5%", "smollm_kq_IQ4_XS_g1_lam0", "cmp_IQ4_XS_im")],
}
TS = re.compile(r"^(\d+)\.(\d+)\.(\d+)\.(\d+) ")   # llama.cpp log time: min.s.ms.us


def ours(d):
    """llmrt spec runs of one energy_run.sh directory -> rows (bundle, task index, phase, tokens, s, J)."""
    t, rails = samples(os.path.join(d, "tegrastats.log"))
    tot = next(k for k in TOTAL if k in rails)
    out = []
    files = sorted(glob.glob(os.path.join(d, "spec_*.jsonl")))
    rows = [(os.path.basename(f)[5:-6].replace("_ilc", ""), i, json.loads(l)) for f in files for i, l in enumerate(open(f))]
    if not rows:
        return out, None
    idle = float(np.median(rails[tot][t < min(r["t_prefill"][0] for _, _, r in rows) - 1.0])) / 1e3
    for b, i, r in rows:
        for ph, (s0, s1), n in (("plain", r["t_plain"], r["plain_tokens"]), ("spec", r["t_spec"], r["spec_tokens"])):
            p, ns = mean_in(t, rails[tot], s0, s1)
            out.append(dict(runtime="llmrt", bundle=b, task=i, phase=ph, tokens=n, seconds=s1 - s0, power_w=p / 1e3,
                            n_samples=ns, idle_w=idle, tokens_per_pass=r.get("tokens_per_pass") if ph == "spec" else 1))
    return out, idle


def llama(d, tasks):
    t, rails = samples(os.path.join(d, "tegrastats.log"))
    tot = next(k for k in TOTAL if k in rails)
    marks = {}
    for line in open(os.path.join(d, "marks.txt")):
        p = line.split()
        marks.setdefault((p[1], p[2], p[3]), {})[p[0]] = float(p[4])
    idle = float(np.median(rails[tot][t < min(m["start"] for m in marks.values()) - 1.0])) / 1e3
    out = []
    for (g, dr, task), m in sorted(marks.items()):
        log = open(os.path.join(d, "%s_d%s_%s.err" % (g, dr, task))).read().splitlines()
        dec = next(l for l in log if " decoded " in l)
        ts = TS.match(dec)
        rel = int(ts.group(1)) * 60 + int(ts.group(2)) + int(ts.group(3)) / 1e3 + int(ts.group(4)) / 1e6
        n, s = re.search(r"decoded\s+(\d+) tokens in\s+([\d.]+) seconds", dec).groups()
        n, s = int(n), float(s)
        val = lambda k: int(re.search(k + r"\s*=\s*(\d+)", "\n".join(log)).group(1))
        s1 = m["start"] + rel
        p, ns = mean_in(t, rails[tot], s1 - s, s1)
        out.append(dict(runtime="llama.cpp", bundle=g, task=tasks.index(task), phase="plain" if dr == "0" else "spec",
                        tokens=n, seconds=s, power_w=p / 1e3, n_samples=ns, idle_w=idle, n_drafted=val("n_drafted"),
                        n_accept=val("n_accept")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--device", required=True)
    ap.add_argument("--ours", default=None, help="root of our runtime's runs (default: root)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--models", default=",".join(HEAD), help="comma list of models to include")
    ap.add_argument("--llama-suffix", default="", help="llama.cpp run dirs llama_<model><suffix> (e.g. _usb)")
    a = ap.parse_args()
    ours_root = a.ours or a.root
    tasks = ["code_rename", "code_docstring", "summarize", "extract", "qa", "story"]
    rows = []
    for m in a.models.split(","):
        for d in sorted(glob.glob(os.path.join(ours_root, "%s*" % m))) + sorted(glob.glob(os.path.join(a.root, "llama_%s%s" % (m, a.llama_suffix)))):
            if not os.path.exists(os.path.join(d, "tegrastats.log")):
                continue
            rs = llama(d, tasks) if os.path.basename(d).startswith("llama_") else ours(d)[0]
            for r in rs:
                r.update(model=m, device=a.device, group=os.path.basename(d))
            rows += rs
    lg = JsonlLogger("a2_spec_energy", cfg=dict(device=a.device, root=os.path.relpath(a.root, REPO),
                                                ours=os.path.relpath(ours_root, REPO), tag=a.tag,
                                                **({"llama_suffix": a.llama_suffix} if a.llama_suffix else {})))
    for r in rows:
        lg.log(r)

    def agg(m, rt, b, ph):   # over the 6 prompts: total tokens / total time; energy = sum(power x time) / tokens
        rs = [r for r in rows if r["model"] == m and r["runtime"] == rt and r["bundle"] == b and r["phase"] == ph]
        if len(rs) == 0:
            return None
        n = sum(r["tokens"] for r in rs)
        s = sum(r["seconds"] for r in rs)
        e = sum(r["power_w"] * r["seconds"] for r in rs)
        dyn = sum((r["power_w"] - r["idle_w"]) * r["seconds"] for r in rs)
        return dict(ms=1e3 * s / n, j=e / n, dyn=dyn / n, w=e / s, k=len(rs))

    fmt = lambda x: "%.1f / %.2f" % (x["ms"], x["j"]) if x else "-"
    out = ["# Decode speed and energy, %s%s" % (a.device, " (our runtime: %s)" % a.tag if a.tag else ""), "",
           "Per point: ms per generated token / J per token (module input rail, idle included), plain greedy and with "
           "prompt-lookup speculation (draft 7), over the 6 prompts (128 new tokens each, stops at EOS). Ours = "
           "kernels/llmrt.cu on the Huffman-coded bundle; twin = the same codes packed (no entropy coding); "
           "llama.cpp = llama-lookup (CUDA) on the same GGUF and on the llama.cpp comparator.", "",
           "| model | budget | ours plain | ours spec | twin plain | twin spec | llama.cpp same GGUF plain | spec | "
           "llama.cpp comparator plain | spec |", "|" + "---|" * 10]
    for m, pts in HEAD.items():
        if m not in a.models.split(","):
            continue
        for bud, b, cmp_ in pts:
            stem = "kq_" + b.split("_kq_", 1)[1] if "_kq_" in b else None
            if b.endswith("_ref"):   # the llama.cpp+imatrix codes, Huffman-coded: same GGUF as the comparator
                stem = cmp_
            cells = [agg(m, "llmrt", b, "plain"), agg(m, "llmrt", b, "spec"), agg(m, "llmrt", b + "_pk", "plain"),
                     agg(m, "llmrt", b + "_pk", "spec"), agg(m, "llama.cpp", stem, "plain"),
                     agg(m, "llama.cpp", stem, "spec"), agg(m, "llama.cpp", cmp_, "plain"),
                     agg(m, "llama.cpp", cmp_, "spec")]
            if not any(cells):
                continue
            out.append("| %s | <=%s | %s |" % (m, bud, " | ".join(fmt(c) for c in cells)))
    p = os.path.join(REPO, "results", "a2_spec_energy_%s%s.md" % (a.device, "_" + a.tag if a.tag else ""))
    open(p, "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
