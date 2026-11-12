"""A5: per-token decode time with a hardware Huffman decoder (hw/huff, A4), calibrated on the Orin measurements.

Two kernel classes, each calibrated on measured rows:
  llama.cpp CUDA: with the hardware decoder in the memory path the kernel is the one of the same GGUF (it sees the
    same codes) while DRAM delivers the Huffman-coded stream, S_coded instead of S_fixed bytes per token (S = bytes
    of weights read per generated token: GGUF tensors except an untied token embedding, which is gathered; S_coded =
    the coded bundle's matmul bytes (payload, tables, lane offsets, GGUF side bytes) + the rest as in the GGUF).
    Bounds: t_fixed * S_coded / S_fixed (weight streaming bandwidth-bound) <= t_hw <= t_fixed (compute-bound),
    t_fixed = llama-bench tg64 at depth 128 of the same GGUF. A bytes-only fit t = a_m + S / beta over all GGUFs of a
    model is logged as a diagnostic: its leave-one-out error shows that time depends on the quant type, not bytes
    alone, which is why the bounds use the same-GGUF kernel.
  our runtime (kernels/llmrt.cu, compute-bound on the Orin): the decoder replaces the in-kernel Huffman decode, so
    t_hw = t_twin (the same kernel on the packed twin); software: t_sw = t_twin + c_dec * n_coded_weights, c_dec
    fitted per device over the 6 points (validation: the measured coded rows).
The hardware point is the smallest A4 engine count T (32 lanes at 1 GHz each) reaching theta* = 8 beta / b_c
(results/a4_csim.jsonl symbols / cycle of the comparator-lookup engine on the bundle, beta = 102 GB/s Orin Nano Super spec).

  .venv/bin/python scripts/a5_system.py [--bench bench11]   -> results/a5_system.jsonl, results/a5_system.md
(--bench: results/a2_orin/<dir> of our runtime's fwd-bench rows; bench11 = llmrt_kq11g = bench7 + CUDA graphs (fwd_*_nog: graphs off); bench7 = llmrt_kq7 (bench6 = llmrt_kq6): int8 activations, multi-query
attention, Q5_K / Q6_K twins at the GGUF width; bench4 = int8 activations; bench2 = the earlier fp32-activation kernel)
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
from kq_code import gguf  # noqa: E402,F401  (llama.cpp gguf-py on the path)
from lazyweights.log import JsonlLogger  # noqa: E402

from lazyweights.paths import MODELS as NAS  # noqa: E402
GDIR = {"phi3": "phi3-mini-4k-instruct", "mistral": "mistral-7b-instruct-v0.2", "smollm": "smollm-1-7b-instruct"}
HEAD = {   # (budget, bundle, llama.cpp comparator stem)
    "phi3": [("1%", "phi3_kq_IQ4_XS_g0_lam0_c1.25", "cmp_IQ4_XS_im"), ("3.5%", "phi3_kq_IQ4_XS_g1_lam0_c1.25", "cmp_Q3_K_M_im")],
    "mistral": [("1%", "mistral_kq_IQ4_XS_g1_lam0", "cmp_Q5_K_M_no"),
                ("3.5%", "mistral_kq_IQ4_XS_g1_lam0_c1.5", "cmp_IQ3_M_im")],
    "smollm": [("1%", "smollm_kq_Q5_K_M_g1_lam0", "cmp_Q5_K_M_no"),
               ("3.5%", "smollm_kq_IQ4_XS_g1_lam0", "cmp_IQ4_XS_im")],
}
BETA_ORIN = 102e9
BETA_NANO = 22.5e9   # Jetson Nano, measured GPU read-reduce bandwidth (results/x3_roofline_nano.csv; 25.6 GB/s peak)
F_CLK = 1e9


def gguf_path(m, stem):
    g = os.path.join(NAS, GDIR[m], "gguf")
    for p in (os.path.join(g, "bench", stem + ".gguf"), os.path.join(g, "kq", stem[3:] + ".gguf")):
        if os.path.exists(p):
            return p
    return None


def streamed_bytes(path):
    """GGUF tensor bytes read per token: all but an untied token_embd (gathered)."""
    from gguf import GGUFReader
    r = GGUFReader(path)
    names = {t.name: int(t.n_bytes) for t in r.tensors}
    tot = sum(names.values())
    if "output.weight" in names:
        tot -= names.get("token_embd.weight", 0)
    return tot, names


def llama_tg(path):
    rows = [json.loads(l) for l in open(path) if l.startswith("{")]
    r = [x for x in rows if x["n_gen"] > 0 and x["n_depth"] == 128]
    return r[0]["avg_ns"] / 1e9 / r[0]["n_gen"] if r else None


def nano(lg, orin_rows):
    """Jetson Nano (MAXN, sm_53: fp32 activations): our runtime's m = 1 forward time (results/a2_nano/kq_out), the
    software decode cost fit as on the Orin, t_hw = t_twin, engines for theta* = 8 beta / b_c at the measured read
    bandwidth; the llama.cpp baseline there is CPU only (results/a2_spec_energy.jsonl, plain greedy)."""
    fwd = {}
    for f in glob.glob(os.path.join(REPO, "results", "a2_nano", "kq_out", "fwd_*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            if r.get("m") == 1:
                fwd[os.path.basename(f)[4:-6]] = r["forward_s"]
    cpu = {}   # (model, GGUF stem) -> s per token, llama.cpp CPU plain greedy over the 6 prompts
    p = os.path.join(REPO, "results", "a2_spec_energy.jsonl")
    for l in open(p) if os.path.exists(p) else []:
        r = json.loads(l)
        if r.get("device") == "nano" and r.get("runtime") == "llama.cpp" and r.get("phase") == "plain":
            k = (r["model"], r["bundle"])
            n, s = cpu.get(k, (0, 0.0))
            cpu[k] = (n + r["tokens"], s + r["seconds"])
    rows = []
    for o in orin_rows:
        t_sw, t_tw = fwd.get(o["bundle"]), fwd.get(o["bundle"] + "_pk")
        if not (t_sw and t_tw):
            continue
        theta = 8 * BETA_NANO / o["b_c"]
        cmp_ = dict((b, c) for b, _, c in [(x[1], 0, x[2]) for x in HEAD[o["model"]]])[o["bundle"]]
        c = cpu.get((o["model"], cmp_))
        rows.append(dict(kind="nano_point", device="nano", beta=BETA_NANO, model=o["model"], budget=o["budget"],
                         bundle=o["bundle"], b_c=o["b_c"], n_matmul=o["n_matmul"], t_llmrt_coded_s=t_sw,
                         t_llmrt_twin_s=t_tw, dram_use_twin=o["S_fixed"] / t_tw / BETA_NANO, theta_star=theta,
                         engines=math.ceil(theta / (o["engine_weights_per_cycle"] * F_CLK)),
                         t_llama_cpu_cmp_s=c[1] / c[0] if c else None))
    if not rows:
        return []
    n = np.array([r["n_matmul"] for r in rows], float)
    dt = np.array([r["t_llmrt_coded_s"] - r["t_llmrt_twin_s"] for r in rows])
    cd = float((dt * n).sum() / (n * n).sum())
    err = [(r["t_llmrt_twin_s"] + cd * k) / r["t_llmrt_coded_s"] - 1 for r, k in zip(rows, n)]
    for r in rows:
        lg.log(r)
    lg.log(dict(kind="sw_fit", device="nano", c_dec_s_per_weight=cd, points=[r["bundle"] for r in rows], err=err))
    out = ["", "## Jetson Nano (MAXN; measured read bandwidth %.1f GB/s)" % (BETA_NANO / 1e9), "",
           "Our runtime on the Nano is compute-bound (fp32 activations, no dp4a on sm_53): the twin reads its weights "
           "at a small fraction of the bandwidth, so t_hw = t_twin and a single engine covers theta*. llama.cpp on the "
           "Nano runs on the CPU only (CUDA 10.2 cannot build its CUDA backend): a different class of baseline.", "",
           "| model | budget | ours: software coded / twin = hardware | twin DRAM use | engines (theta*) | "
           "llama.cpp CPU comparator | ours + hardware vs CPU comparator |", "|" + "---|" * 7]
    for r in rows:
        c = r["t_llama_cpu_cmp_s"]
        out.append("| %s | <=%s | %.1f / %.1f ms | %.0f%% | %d (%.0f G/s) | %s | %s |" % (
            r["model"], r["budget"], 1e3 * r["t_llmrt_coded_s"], 1e3 * r["t_llmrt_twin_s"], 100 * r["dram_use_twin"],
            r["engines"], r["theta_star"] / 1e9, "%.1f ms" % (1e3 * c) if c else "pending",
            "%+.0f%%" % (100 * (r["t_llmrt_twin_s"] / c - 1)) if c else "-"))
    out += ["", "Software Huffman decode on the Nano: %.2f ns per 1000 coded weights per token on top of the twin (fit "
            "over %d points; prediction error max %.1f%%)." % (1e12 * cd, len(rows), 100 * max(map(abs, err)))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="bench11")
    args = ap.parse_args()
    lg = JsonlLogger("a5_system", cfg=dict(device="orin", beta=BETA_ORIN, f_clk=F_CLK, bench=args.bench))
    bund = {}
    for l in open(os.path.join(REPO, "results", "llm_bundle.jsonl")):
        r = json.loads(l)
        if "matmul_bpw" in r and "gguf" in r:
            bund.setdefault(r["gguf"], []).append(r)
    csim = [json.loads(l) for l in open(os.path.join(REPO, "results", "a4_csim.jsonl"))]
    fwd = {}
    for f in glob.glob(os.path.join(REPO, "results", "a2_orin", args.bench, "fwd_*.jsonl")):
        if f.endswith("_nog.jsonl"):   # bench11: the graphs-off control
            continue
        for l in open(f):
            r = json.loads(l)
            if r["m"] == 1:
                fwd[os.path.basename(f)[4:-6]] = r["forward_s"]
    out = ["# A5: per-token decode time with the hardware decoder (Orin Nano Super)", "",
           "Our runtime rows: results/a2_orin/%s (%s)." % (args.bench, "int8 activations in the decode GEMV"
                                                          if args.bench in ("bench4", "bench6", "bench7", "bench11") else "fp32 activations"), ""]
    rows_md = []
    # ---- bandwidth-class fit per model
    fits = {}
    for m in GDIR:
        pts = []
        for f in sorted(glob.glob(os.path.join(REPO, "results", "a2_orin", "llama_%s_*.txt" % m))):
            stem = os.path.basename(f)[len("llama_%s_" % m):-4]
            p = gguf_path(m, stem)
            t = llama_tg(f)
            if p and t:
                pts.append((stem, streamed_bytes(p)[0], t))
        S = np.array([p[1] for p in pts], float)
        T = np.array([p[2] for p in pts], float)
        A = np.stack([np.ones_like(S), S], 1)
        (a, b), *_ = np.linalg.lstsq(A, T, rcond=None)
        loo = []
        for i in range(len(pts)):
            k = np.arange(len(pts)) != i
            (ai, bi), *_ = np.linalg.lstsq(A[k], T[k], rcond=None)
            loo.append((ai + bi * S[i]) / T[i] - 1)
        fits[m] = (a, b)
        lg.log(dict(kind="fit", model=m, a_s=a, beta_eff=1 / b, n=len(pts), points=[p[0] for p in pts],
                    loo_err=loo, loo_max_abs=max(map(abs, loo))))
        out.append("- diagnostic, %s: bytes-only fit t = %.2f ms + S / %.1f GB/s over %d llama.cpp GGUFs, leave-one-out "
                   "error max %.0f%% (time depends on the quant type)" % (m, 1e3 * a, 1e-9 / b, len(pts),
                                                                          100 * max(map(abs, loo))))
    out += ["", "llama.cpp kernels: measured per-token time of the comparator and of the same GGUF without entropy "
            "coding (DRAM use of the latter: S_fixed / t / 102 GB/s); with the hardware decoder, bounds "
            "[bandwidth-bound, compute-bound]. Engines: smallest number of 32-lane engines at 1 GHz reaching "
            "theta* = 8 beta / b_c (A4 C simulation rate of the comparator-lookup engine on the bundle's matrices).", "",
            "| model | budget | llama.cpp comparator | same GGUF fixed-width (DRAM use) | + hardware decoder | "
            "vs comparator | engines (theta*) | ours: software coded / twin = hardware | ours + hardware vs "
            "comparator |", "|" + "---|" * 9]
    for m, pts in HEAD.items():
        a, b = fits[m]
        for bud, bdl, cmp_ in pts:
            gstem = cmp_ if bdl.endswith("_ref") else "kq_" + bdl.split("_kq_", 1)[1]   # measured with llama.cpp
            gp = os.path.join(NAS, GDIR[m], "gguf", "kq", bdl.split("_kq_", 1)[1] + ".gguf")   # bundle source
            S_fixed, names = streamed_bytes(gp)
            br = [r for r in bund.get(gp, []) if r.get("cfg", {}).get("enc") == "kq_huf"]
            if not br:
                continue
            br = br[-1]
            coded_matmul = br["matmul_bpw"] * br["matmul_params"] / 8
            mm = {n: v for n, v in names.items() if n.endswith(".weight") and n != "token_embd.weight"
                  and ("blk." in n and any(x in n for x in ("attn_q", "attn_k", "attn_v", "attn_qkv", "attn_output",
                                                             "ffn_gate", "ffn_up", "ffn_down")) or n == "output.weight")}
            fixed_matmul = sum(mm.values())
            if "output.weight" not in names:   # tied: the embedding is the output matrix (in the bundle's matmuls)
                fixed_matmul += names["token_embd.weight"]
            S_coded = S_fixed - fixed_matmul + coded_matmul
            t_cmp = llama_tg(os.path.join(REPO, "results", "a2_orin", "llama_%s_%s.txt" % (m, cmp_)))
            t_fix = llama_tg(os.path.join(REPO, "results", "a2_orin", "llama_%s_%s.txt" % (m, gstem)))
            t_hw_lo, t_hw_hi = t_fix * S_coded / S_fixed, t_fix
            cs = [r for r in csim if r["bundle"] == bdl and r["config"] == "comparator"]
            wpc = cs[-1]["weights_per_cycle"] if cs else float("nan")
            b_c = 8 * coded_matmul / br["matmul_params"]
            theta = 8 * BETA_ORIN / b_c
            T_eng = math.ceil(theta / (wpc * F_CLK))
            t_sw, t_tw = fwd.get(bdl), fwd.get(bdl + "_pk")
            row = dict(kind="point", model=m, budget=bud, bundle=bdl, gguf=os.path.basename(gp), S_fixed=S_fixed,
                       S_coded=S_coded, t_cmp_s=t_cmp, t_fixed_s=t_fix, t_hw_lo_s=t_hw_lo, t_hw_hi_s=t_hw_hi,
                       dram_use_fixed=S_fixed / t_fix / BETA_ORIN, theta_star=theta,
                       engine_weights_per_cycle=wpc, engines=T_eng, b_c=b_c, t_llmrt_coded_s=t_sw,
                       t_llmrt_twin_s=t_tw, n_matmul=br["matmul_params"])
            lg.log(row)
            rows_md.append(row)
            out.append("| %s | <=%s | %.1f ms | %.1f ms (%.0f%%) | %.1f - %.1f ms | %+.0f%% to %+.0f%% | %d (%.0f G/s) | "
                       "%s | %s |" % (
                m, bud, 1e3 * t_cmp, 1e3 * t_fix, 100 * S_fixed / t_fix / BETA_ORIN, 1e3 * t_hw_lo, 1e3 * t_hw_hi,
                100 * (t_hw_lo / t_cmp - 1), 100 * (t_hw_hi / t_cmp - 1), T_eng, theta / 1e9,
                "%.1f / %.1f ms" % (1e3 * t_sw, 1e3 * t_tw) if t_sw and t_tw else "-",
                "%+.0f%%" % (100 * (t_tw / t_cmp - 1)) if t_tw else "-"))
    # ---- our runtime: software decode cost per coded weight, t_sw = t_twin + c_dec * n (validation: per point)
    sw = [r for r in rows_md if r["t_llmrt_coded_s"] and r["t_llmrt_twin_s"]]
    if sw:
        n = np.array([r["n_matmul"] for r in sw], float)
        dt = np.array([r["t_llmrt_coded_s"] - r["t_llmrt_twin_s"] for r in sw])
        c = float((dt * n).sum() / (n * n).sum())
        err = [(r["t_llmrt_twin_s"] + c * k) / r["t_llmrt_coded_s"] - 1 for r, k in zip(sw, n)]
        lg.log(dict(kind="sw_fit", c_dec_s_per_weight=c, points=[r["bundle"] for r in sw], err=err))
        out += ["", "Our runtime: software Huffman decode costs %.2f ns per 1000 coded weights per token on top of the "
                "packed twin (fit over %d points; prediction error of the coded rows max %.1f%%). The hardware "
                "decoder removes it: t_hw = t_twin." % (1e12 * c, len(sw), 100 * max(map(abs, err)))]
    out += nano(lg, rows_md)
    open(os.path.join(REPO, "results", "a5_system.md"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
