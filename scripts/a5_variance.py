"""Run-to-run variance of the Orin end-to-end table (paper Table e2e), from R repeats of its measured rows
(scripts/device/orin_variance.sh -> ~/lw_data/var/r<i>, copied to results/a2_orin/var/r<i>): per repeat, the llama.cpp
budget baseline and our fixed-width twin (= the time with the unit), greedy and prompt-lookup speculation, aggregated
over the 6 prompts as in scripts/a2_spec_energy.py; Delta t (conservative: t_twin; bandwidth-bound: the table's
coded/fixed byte ratio) and Delta E from the same energy model as scripts/a5_energy.py (the table's DRAM-saved and
decoder energies per weight pass, times this repeat's passes per token).

  .venv/bin/python scripts/a5_variance.py results/a2_orin/var
Rows -> results/a5_variance.jsonl; table -> results/a5_variance.md
"""
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from a2_spec_energy import llama, ours  # noqa: E402

TASKS = ["code_rename", "code_docstring", "summarize", "extract", "qa", "story"]
POINTS = {  # (model, budget) -> (twin bundle, llama.cpp comparator)
    ("phi3", "1%"): ("phi3_kq_IQ4_XS_g0_lam0_c1.25_pk", "cmp_IQ4_XS_im"),
    ("phi3", "3.5%"): ("phi3_kq_IQ4_XS_g1_lam0_c1.25_pk", "cmp_Q3_K_M_im"),
    ("mistral", "1%"): ("mistral_kq_IQ4_XS_g1_lam0_pk", "cmp_Q5_K_M_no"),
    ("mistral", "3.5%"): ("mistral_kq_IQ4_XS_g1_lam0_c1.5_pk", "cmp_IQ3_M_im"),
    ("smollm", "1%"): ("smollm_kq_Q5_K_M_g1_lam0_pk", "cmp_Q5_K_M_no"),
    ("smollm", "3.5%"): ("smollm_kq_IQ4_XS_g1_lam0_pk", "cmp_IQ4_XS_im"),
}
TABLE_CFG = "257e8cc962d2"   # the a5_energy run behind the paper's table


def agg(rows):
    n = sum(r["tokens"] for r in rows)
    s = sum(r["seconds"] for r in rows)
    e = sum(r["power_w"] * r["seconds"] for r in rows)
    passes = sum(r["tokens"] / r["tokens_per_pass"] for r in rows) if all(r.get("tokens_per_pass") for r in rows) else n
    return dict(t=s / n, e=e / n, k=passes / n, prompts=len(rows))


def main():
    root = sys.argv[1]
    ref = {}
    for l in open(os.path.join(REPO, "results", "a5_energy.jsonl")):
        r = json.loads(l)
        if r.get("device") == "orin" and r.get("kind") == "point" and r.get("config_hash") == TABLE_CFG:
            ref[(r["model"], r["budget"], r["phase"])] = r
    out = []
    reps = sorted([d for d in glob.glob(os.path.join(root, "r*")) if os.path.isdir(d)], key=lambda d: int(os.path.basename(d)[1:]))
    for d in reps:
        rows = []
        for sub in sorted(glob.glob(os.path.join(d, "*"))):
            if not os.path.exists(os.path.join(sub, "done")):
                continue
            rows += llama(sub, TASKS) if os.path.basename(sub).startswith("llama_") else ours(sub)[0]
            for r in rows:
                r.setdefault("model", os.path.basename(sub).replace("llama_", "").replace("g0", ""))
        for (m, bud), (tw, cmp_) in POINTS.items():
            for ph in ("plain", "spec"):
                rt = [r for r in rows if r["runtime"] == "llmrt" and r["bundle"] == tw and r["phase"] == ph]
                rb = [r for r in rows if r["runtime"] == "llama.cpp" and r["bundle"] == cmp_ and r["phase"] == ph
                      and r["model"] == m]
                if len(rt) != 6 or len(rb) != 6:
                    continue
                a, b, f = agg(rt), agg(rb), ref[(m, bud, ph)]
                k_scale = a["k"] / f["passes_per_token"]
                e_hw = a["e"] - (f["e_twin_j"] - f["e_hw_j"]) * k_scale
                bw = f["t_hw_lo_s"] / f["t_twin_s"]
                out.append(dict(rep=os.path.basename(d), model=m, budget=bud, phase=ph, t_base_ms=1e3 * b["t"],
                                e_base_mj=1e3 * b["e"], t_twin_ms=1e3 * a["t"], e_twin_mj=1e3 * a["e"],
                                e_hw_mj=1e3 * e_hw, passes_per_token=a["k"], dt_cons=a["t"] / b["t"] - 1,
                                dt_bw=a["t"] * bw / b["t"] - 1, de=e_hw / b["e"] - 1,
                                table=dict(t_base_ms=1e3 * f["t_base_s"], e_base_mj=1e3 * f["e_base_j"],
                                           t_twin_ms=1e3 * f["t_twin_s"], dt_cons=f["t_vs_base"],
                                           dt_bw=f["t_lo_vs_base"], de=f["vs_base"])))
    with open(os.path.join(REPO, "results", "a5_variance.jsonl"), "w") as fo:
        for r in out:
            fo.write(json.dumps(r) + "\n")
    cv = lambda x: 100 * np.std(x, ddof=1) / np.mean(x) if len(x) > 1 else float("nan")
    L = ["# Run-to-run variance of the Orin end-to-end table (%d repeats)" % len(reps), "",
         "Per point and mode: coefficient of variation (sd / mean, %) of ms and mJ per token over the repeats, and the "
         "range of Delta t (bandwidth-bound / conservative) and Delta E vs the llama.cpp baseline; [table] = the paper's "
         "values (single run). Clocks: MAXN_SUPER + jetson_clocks.", "",
         "| model | budget | mode | n | baseline ms CV | baseline mJ CV | twin ms CV | twin mJ CV | Delta t bw [table] | "
         "Delta t cons [table] | Delta E [table] |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    allcv, dev = [], []
    for (m, bud) in POINTS:
        for ph in ("plain", "spec"):
            rs = [r for r in out if r["model"] == m and r["budget"] == bud and r["phase"] == ph]
            if not rs:
                continue
            t = rs[0]["table"]
            g = lambda k: [r[k] for r in rs]
            c = [cv(g(k)) for k in ("t_base_ms", "e_base_mj", "t_twin_ms", "e_twin_mj")]
            allcv += c
            rng = lambda k: "%+.0f..%+.0f%% [%+.0f%%]" % (100 * min(g(k)), 100 * max(g(k)), 100 * t[k])
            dev += [abs(x - t[k]) for k in ("dt_cons", "dt_bw", "de") for x in g(k)]
            L.append("| %s | <=%s | %s | %d | %.1f | %.1f | %.1f | %.1f | %s | %s | %s |" % (
                m, bud, ph, len(rs), *c, rng("dt_bw"), rng("dt_cons"), rng("de")))
    if allcv:
        L += ["", "Max CV over all cells: %.1f%%; median %.1f%%. Largest |Delta - table| over all repeats: %.1f points." % (
            max(allcv), float(np.median(allcv)), 100 * max(dev))]
    open(os.path.join(REPO, "results", "a5_variance.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
