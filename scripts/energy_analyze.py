"""Energy per token from scripts/device/energy_run.sh output: tegrastats samples (timestamped every ~50 ms) integrated over
the wall-clock phase marks that kernels/llmrt.cu `spec` prints (prefill, plain greedy, prompt-lookup speculation).

Power rails: module input (Orin VDD_IN, Nano POM_5V_IN; the whole module, not the carrier board) and the GPU rail
(Orin VDD_CPU_GPU_CV, which includes the CPU and the vision accelerators; Nano POM_5V_GPU). Idle power is the
median module input over the samples more than 1 s before the first run. Per phase: mean power, energy (mean power x
duration), energy per token (prefill: per prompt token), and the dynamic part above idle.

  python scripts/energy_analyze.py results/orin/energy/phi3 [...]   -> results/energy.jsonl (one row per bundle,
  prompt, phase) and a summary table
"""
import argparse
import glob
import json
import os
import re

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAIL = re.compile(r"\b(VDD_[A-Z_]+|POM_[0-9A-Z_]+) (\d+)(?:mW)?/(\d+)(?:mW)?")
TOTAL = ("VDD_IN", "POM_5V_IN")
GPU = ("VDD_CPU_GPU_CV", "POM_5V_GPU")


def samples(path):
    t, rails = [], {}
    for line in open(path):
        parts = line.split(" ", 1)
        try:
            ts = float(parts[0])
        except ValueError:
            continue
        r = {k: float(v) for k, v, _ in RAIL.findall(parts[1])}
        if not r:
            continue
        t.append(ts)
        for k, v in r.items():
            rails.setdefault(k, []).append(v)
    return np.array(t), {k: np.array(v) for k, v in rails.items()}


def mean_in(t, v, a, b):
    m = (t >= a) & (t <= b)
    if m.sum() < 2:          # very short phase: nearest samples
        i = np.searchsorted(t, (a + b) / 2)
        return float(v[min(max(i, 0), len(v) - 1)]), int(m.sum())
    return float(v[m].mean()), int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--device", default="")
    a = ap.parse_args()
    out = []
    for d in a.dirs:
        t, rails = samples(os.path.join(d, "tegrastats.log"))
        tot = next(k for k in TOTAL if k in rails)
        gpu = next(k for k in GPU if k in rails)
        specs = sorted(glob.glob(os.path.join(d, "spec_*.jsonl")))
        rows = [(os.path.basename(f)[5:-6], json.loads(l)) for f in specs for l in open(f)]
        if not rows:
            continue
        first = min(r["t_prefill"][0] for _, r in rows)
        idle = float(np.median(rails[tot][t < first - 1.0]))
        idle_gpu = float(np.median(rails[gpu][t < first - 1.0]))
        for bundle, r in rows:
            for ph, (s0, s1), ntok in (("prefill", r["t_prefill"], r["prompt_tokens"]),
                                       ("plain", r["t_plain"], r["plain_tokens"]),
                                       ("spec", r["t_spec"], r["spec_tokens"])):
                dur = s1 - s0
                p, n = mean_in(t, rails[tot], s0, s1)
                pg, _ = mean_in(t, rails[gpu], s0, s1)
                row = dict(device=a.device or os.path.basename(os.path.dirname(os.path.dirname(d.rstrip("/")))),
                           group=os.path.basename(d.rstrip("/")), bundle=bundle, prompt_tokens=r["prompt_tokens"],
                           draft=r["draft"], phase=ph, tokens=ntok, seconds=dur, n_samples=n, rail=tot,
                           power_w=p / 1e3, gpu_rail=gpu, gpu_power_w=pg / 1e3, idle_w=idle / 1e3,
                           idle_gpu_w=idle_gpu / 1e3, energy_j=p / 1e3 * dur,
                           energy_per_token_j=p / 1e3 * dur / max(1, ntok),
                           dynamic_per_token_j=(p - idle) / 1e3 * dur / max(1, ntok),
                           s_per_token=dur / max(1, ntok), tokens_per_pass=r["tokens_per_pass"] if ph == "spec" else 1)
                out.append(row)
    with open(os.path.join(REPO, "results", "energy.jsonl"), "a") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")
    print("%-8s %-30s %-7s %5s %8s %8s %7s %7s %9s %9s" % ("device", "bundle", "phase", "p_tok", "s/tok", "tok/s",
                                                              "P (W)", "Pidle", "J/token", "dyn J/tk"))
    for row in out:
        print("%-8s %-30s %-7s %5d %8.4f %8.2f %7.2f %7.2f %9.4f %9.4f" % (
            row["device"], row["bundle"], row["phase"], row["prompt_tokens"], row["s_per_token"],
            1 / row["s_per_token"], row["power_w"], row["idle_w"], row["energy_per_token_j"],
            row["dynamic_per_token_j"]))


if __name__ == "__main__":
    main()
