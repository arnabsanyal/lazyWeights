"""Paper tables: fixed quality budgets (dPPL vs fp16 <= 1%, 2%, 3.5%). Per model and budget: the smallest llama.cpp
point (with or without imatrix), our smallest fixed-width point, our smallest Huffman-coded point, and the fixed-width
twin of that coded point (same codes, no entropy coding), so saving vs llama.cpp = quantizer part x Huffman part.
Then the 6 headline points (<=1% and <=3.5%, each vs its llama.cpp comparator) with
HellaSwag and GSM8K chain-of-thought (8-shot, strict / flexible extract) next to fp16; task scores are merged over all
rows of one quantization (a coded point and its packed twin, or a llama.cpp point and its Huffman-coded copy, share
the codes and so the scores).

  .venv/bin/python scripts/kq_tables.py   -> results/kq_tables.md
"""
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))

FP16 = {"phi3": 9.19627529670432, "mistral": 8.024064754867037, "smollm": 17.97652507382426}  # fp16 ppl_full (a1_frontier fp16 rows), unrounded
NAME = {"phi3": "Phi-3-mini", "mistral": "Mistral-7B", "smollm": "SmolLM-1.7B"}


def label(r):
    if r["method"] == "kq":
        return "%s GPTQ γ%g%s λ%g" % (r["qtype"], r["gamma"], " ×%g" % r["coarse"] if r.get("coarse", 1) != 1 else "",
                                      r["lam"]) + (" (RTN)" if r.get("rtn") else "")
    if r["method"].startswith("llama.cpp"):
        return r["qtype"] + (" +imatrix" if "imatrix" in r["method"] else "") + (" +Huffman" if "huf" in r["method"] else "")
    return "%s %db %s λ%g%s" % (r["method"].upper(), r["bits"], r["gran"], r["lam"], " +ao" if r.get("act_order") else "")


BUDGETS = (1.0, 2.0, 3.5)
TASKS = ("hellaswag", "arc_easy", "piqa", "gsm8k")


def key(r):   # identifies a quantization (coded and packed rows of one point share it)
    return tuple(r.get(k) for k in ("model", "method", "qtype", "bits", "gran", "lam", "gamma", "coarse", "act_order",
                                    "exp", "rtn"))


HEAD = [  # (model, budget, ours, llama.cpp comparator); a point = row fields that select it
    ("phi3", 1.0, dict(method="kq", qtype="IQ4_XS", gamma=0.0, coarse=1.25, lam=0.0, family="coded"),
     dict(method="llama.cpp+imatrix", qtype="IQ4_XS", family="packed")),
    ("mistral", 1.0, dict(method="kq", qtype="IQ4_XS", gamma=1.0, coarse=1.0, lam=0.0, family="coded"),
     dict(method="llama.cpp", qtype="Q5_K_M", family="packed")),
    ("smollm", 1.0, dict(method="kq", qtype="Q5_K_M", gamma=1.0, coarse=1.0, lam=0.0, family="coded"),
     dict(method="llama.cpp", qtype="Q5_K_M", family="packed")),
    ("phi3", 3.5, dict(method="kq", qtype="IQ4_XS", gamma=1.0, coarse=1.25, lam=0.0, family="coded"),
     dict(method="llama.cpp+imatrix", qtype="Q3_K_M", family="packed")),
    ("mistral", 3.5, dict(method="kq", qtype="IQ4_XS", gamma=1.0, coarse=1.5, lam=0.0, family="coded"),
     dict(method="llama.cpp+imatrix", qtype="IQ3_M", family="packed")),
    ("smollm", 3.5, dict(method="kq", qtype="IQ4_XS", gamma=1.0, coarse=1.0, lam=0.0, family="coded"),
     dict(method="llama.cpp+imatrix", qtype="IQ4_XS", family="packed")),
]
HTASKS = ("hellaswag", "gsm8k_cot", "gsm8k_cot_flex")


def tkey(r):   # task scores depend on the codes only: "+huf" (coded copy of a llama.cpp point) shares them
    k = list(key(r))
    k[1] = k[1].replace("+huf", "")
    return tuple(k)


def headline(rows):
    sc = {}
    for r in rows:
        d = sc.setdefault(tkey(r), {})
        for t in HTASKS + ("gsm8k_cot_stderr",):
            if r.get(t) is not None:
                d[t] = r[t]
    pick = lambda m, f: [r for r in rows if r["model"] == m and not r.get("rtn") and all(r.get(k) == v for k, v in f.items())]   # --rtn ablation rows share the point fields
    cell = lambda r, t: "%.3f" % sc[tkey(r)][t] if t in sc[tkey(r)] else "pending"
    out = ["", "## Headline points: task accuracy", "",
           "HellaSwag: lm-eval acc_norm (0-shot), all 10042. GSM8K-CoT: 8-shot chain of thought, all 1319, exact match "
           "(strict = 'The answer is N', flexible = last number); +- = stderr of the strict score.", "",
           "| model | budget | point | b/w | dPPL | HellaSwag | GSM8K-CoT strict | GSM8K-CoT flexible |", "|" + "---|" * 8]
    for m in ("phi3", "mistral", "smollm"):
        f = [r for r in rows if r["model"] == m and r["method"] == "fp16"]
        out.append("| %s | - | fp16 | 16 | 0 | %s | %s | %s |" % (
            NAME[m], cell(f[0], "hellaswag"), cell(f[0], "gsm8k_cot") + (
                " +- %.3f" % sc[tkey(f[0])]["gsm8k_cot_stderr"] if "gsm8k_cot_stderr" in sc[tkey(f[0])] else ""),
            cell(f[0], "gsm8k_cot_flex")))
        for mm, d, ours, cmp_ in HEAD:
            if mm != m:
                continue
            for who, f in (("ours", ours), ("llama.cpp", cmp_)):
                c = pick(m, f)
                if len({tkey(r) for r in c}) != 1:   # reruns with other tasks repeat a row; one point only
                    raise SystemExit("headline point %s %s: %d points" % (m, f, len({tkey(r) for r in c})))
                c = c[0]
                se = sc[tkey(c)].get("gsm8k_cot_stderr")
                out.append("| %s | <=%g%% | %s: %s | %.3f | %+.2f%% | %s | %s | %s |" % (
                    NAME[m], d, who, label(c), c["bpw"], c["dppl"], cell(c, "hellaswag"),
                    cell(c, "gsm8k_cot") + (" +- %.3f" % se if se else ""), cell(c, "gsm8k_cot_flex")))
    return out


def main():
    rows = [json.loads(l) for l in open(os.path.join(REPO, "results", "a1_frontier.jsonl"))]
    extra = {}   # task scores of every row (incl. task-only re-scoring rows) go to every row of the same quantization
    for r in rows:
        if r.get("method") == "kq" and r.get("coarse") is None:
            r["coarse"] = 1.0   # rows from before the coarse knob: the default grid
        d = extra.setdefault(tkey(r), {})
        d.update({t: r[t] for t in r if t.startswith(TASKS + HTASKS)})
    rows = [r for r in rows if "ppl_full" in r]
    for r in rows:
        for t, v in extra.get(tkey(r), {}).items():
            r.setdefault(t, v)
    for r in rows:
        r["dppl"] = 100 * (r["ppl_full"] / FP16[r["model"]] - 1)
    packed = {key(r): r for r in rows if r["family"] == "packed"}
    best = lambda c, d: min([r for r in c if r["dppl"] <= d], key=lambda r: r["bpw"], default=None)
    out = ["# Fixed quality budgets", "",
           "dPPL vs fp16 (WikiText-2 whole test set, 512-token chunks). b/w include scales, Huffman tables, lane offsets "
           "and the embedding. llama.cpp = smallest llama.cpp point (with or without imatrix) within the budget; "
           "fixed-width = our smallest packed point; coded = our smallest Huffman-coded point; twin = the same codes "
           "as the coded point, packed (no entropy coding). Saving vs llama.cpp = 1 - coded / llama.cpp; quantizer "
           "part = 1 - twin / llama.cpp; Huffman part = 1 - coded / twin.", "",
           "| model | budget | llama.cpp | fixed-width (ours) | coded (ours) | twin b/w | saving vs llama.cpp | "
           "quantizer part | Huffman part | saving vs best fixed-width |", "|" + "---|" * 10]
    for m in ("phi3", "mistral", "smollm"):
        mr = [r for r in rows if r["model"] == m]
        L = [r for r in mr if r["method"].startswith("llama.cpp") and r["family"] == "packed"]
        P = [r for r in mr if r["family"] == "packed"]
        Fw = [r for r in P if not r["method"].startswith("llama.cpp")]
        C = [r for r in mr if r["family"] == "coded"]
        for d in BUDGETS:
            l, f, c, p = best(L, d), best(Fw, d), best(C, d), best(P, d)
            tw = packed.get(key(c)) if c else None
            twb = tw["bpw"] if tw else (c.get("packed_bpw") if c else None)
            cell = lambda r: "%s, %.3f (%+.2f%%)" % (label(r), r["bpw"], r["dppl"]) if r else "-"
            out.append("| %s | <=%g%% | %s | %s | %s | %s | %.1f%% | %s | %s | %.1f%% |" % (
                NAME[m], d, cell(l), cell(f), cell(c), "%.3f" % twb if twb else "-",
                100 * (1 - c["bpw"] / l["bpw"]),
                "%.1f%%" % (100 * (1 - twb / l["bpw"])) if twb else "-",
                "%.1f%%" % (100 * (1 - c["bpw"] / twb)) if twb else "-",
                100 * (1 - c["bpw"] / p["bpw"])))
    out += ["", "## Task accuracy of the table points", "",
            "| model | point | family | b/w | dPPL | " + " | ".join(TASKS) + " |", "|" + "---|" * (5 + len(TASKS))]
    seen = set()
    for m in ("phi3", "mistral", "smollm"):
        mr = [r for r in rows if r["model"] == m]
        for d in BUDGETS:
            for c in (best([r for r in mr if r["method"].startswith("llama.cpp") and r["family"] == "packed"], d),
                      best([r for r in mr if r["family"] == "coded"], d)):
                if c is None or id(c) in seen:
                    continue
                seen.add(id(c))
                acc = [("%.3f" % c[t]) if t in c else "pending" for t in TASKS]
                out.append("| %s | %s | %s | %.3f | %+.2f%% | %s |" % (NAME[m], label(c), c["family"], c["bpw"],
                                                                       c["dppl"], " | ".join(acc)))
    out += headline(rows)
    open(os.path.join(REPO, "results", "kq_tables.md"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
