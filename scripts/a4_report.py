"""A4 report: C simulation (results/a4_csim.jsonl), co-simulation (results/a4_cosim.jsonl), ASAP7 post-synthesis
area / timing / power of the lane and engine (build/syn/<name>/ from hw/huff/syn/synth.sh and power.sh, logged here
to results/a4_asic.jsonl), and the hardware needed per headline point (engines reaching theta*, results/a5_system.jsonl).

  .venv/bin/python scripts/a4_report.py [--log-asic name ...]   -> results/a4_report.md
"""
import argparse
import glob
import json
import os
import re
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from lazyweights.log import JsonlLogger  # noqa: E402

SYN = os.path.join(REPO, "build", "syn")
POW = os.path.join(REPO, "build", "power")


def asic_row(name):
    """Post-synthesis numbers of build/syn/<name>; after OpenROAD repair (REPAIR=1) when it ran."""
    d = os.path.join(SYN, name)
    rd = lambda f: open(os.path.join(d, f)).read()
    rep = os.path.exists(os.path.join(d, "repair_wns.txt"))
    pre = "repair_" if rep else ""
    if rep:
        area = float(re.search(r"design area ([\d.]+)", rd("repair_area.txt")).group(1))
    else:
        area = float(re.search(r"Chip area for module '\\\S+': ([\d.]+)", rd("area.txt")).group(1))
    plat, clk = "asap7", 1000   # platform.txt (synth.sh): platform and clock (ps); Nangate45 times are in ns
    if os.path.exists(os.path.join(d, "platform.txt")):
        plat, clk = rd("platform.txt").split()
        clk = int(clk)
    tu = 1000.0 if plat == "nangate45" else 1.0
    wns = tu * float(re.search(r"wns\s+(-?[\d.]+)", rd(pre + "wns.txt")).group(1))
    r2r = tu * float(re.search(r"(-?[\d.]+)\s+slack", rd(pre + "timing_reg2reg.txt")).group(1))
    ys = rd("synth.ys")
    top = re.search(r"hierarchy -check -top (\S+)", ys).group(1)
    par = dict(re.findall(r"-set (\S+) (\S+)", ys))
    cells = int(re.search(r"Number of cells:\s+(\d+)", rd("area.txt")).group(1))
    flops = sum(int(x) for x in re.findall(r"DFF\S+\s+(\d+)", rd("area.txt")))
    abc = re.search(r"map -D (\d+)", rd("abc.script")).group(1)
    row = dict(name=name, top=top, params=par, area_um2=area, cells=cells, flops=flops, wns_ps=wns,
               reg2reg_slack_ps=r2r, clk_ps=clk,
               corner="Nangate45 typical" if plat == "nangate45" else "ASAP7 RVT TT",
               flow="yosys 0.38 + abc (map -D %s)%s, OpenSTA 2.5" % (abc, ", OpenROAD repair_design / repair_timing "
                                                                    "(no placement)" if rep else ""),
               default_activity_power_w=None, vcd_power_w=None)
    m = re.search(r"^Total\s+\S+\s+\S+\s+\S+\s+(\S+)", rd(pre + "power.txt"), re.M)
    if m:
        row["default_activity_power_w"] = float(m.group(1))
    for p in glob.glob(os.path.join(POW, name + "_*.power.txt")):
        m = re.search(r"^Total\s+\S+\s+\S+\s+\S+\s+(\S+)", open(p).read(), re.M)
        if m:
            row.setdefault("vcd_power", {})[os.path.basename(p)[len(name) + 1:-10]] = float(m.group(1))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-asic", nargs="*", default=[])
    a = ap.parse_args()
    for n in a.log_asic:
        JsonlLogger("a4_asic", cfg=dict(name=n)).log(asic_row(n))
    rd = lambda f: [json.loads(l) for l in open(os.path.join(REPO, "results", f))] if os.path.exists(
        os.path.join(REPO, "results", f)) else []
    cs, co, asic, a5 = rd("a4_csim.jsonl"), rd("a4_cosim.jsonl"), rd("a4_asic.jsonl"), rd("a5_system.jsonl")
    out = ["# A4: hardware Huffman decoder", "",
           "## C simulation (hw/huff/huff_engine.h) on every Huffman-coded matrix", "",
           "Engine = 32 lanes, 1 symbol per lane per cycle target (huf4p: 2 weights per symbol). Reference config: "
           "64-bit buffer, 4-word FIFO, 8-word (256-bit) read port, 8-cycle memory latency, stream-end prefetch.", "",
           "| bundle | matrices | mismatches (table, comparator) | symbols / cycle | weights / cycle | stall | "
           "words / cycle | no prefetch | port 6 | port 7 | latency 32 (FIFO 8) |", "|" + "---|" * 11]
    by = {}
    for r in cs:
        by.setdefault(r["bundle"], {})[r["config"]] = r
    for b, c in by.items():
        t = c["table"]
        out.append("| %s | %d | %d, %d | %.2f | %.1f | %.2f%% | %.2f | %.2f | %.2f | %.2f | %.2f |" % (
            b, t["matrices"], t["mismatch"], c["comparator"]["mismatch"], t["sym_per_cycle"], t["weights_per_cycle"],
            100 * t["stall_frac"], t["words_per_cycle"], c["no_prefetch"]["sym_per_cycle"],
            c["port6"]["sym_per_cycle"], c["port7"]["sym_per_cycle"], c["fifo8_lat32"]["sym_per_cycle"]))
    out += ["", "## Co-simulation (Verilator, hw/huff/rtl) against the C model", ""]
    cfgs, last = {}, {}   # the jsonl accumulates runs: per configuration, count only its latest run (git commit)
    for r in co:
        k = (r["lookup"], r["buf"], r["fifo"], r["port"], r["lat"])
        last[k] = max(last.get(k, ("", "")), (r["timestamp"], r["git_commit"]))
    for r in co:
        k = (r["lookup"], r["buf"], r["fifo"], r["port"], r["lat"])
        if r["git_commit"] != last[k][1]:
            continue
        c = cfgs.setdefault(k, [0, 0, set()])
        c[0] += 1
        c[1] += r["ok"]
        c[2].add(r["enc"])
    for k, (n, ok, encs) in sorted(cfgs.items()):
        out.append("- lookup %s, buffer %d, FIFO %d, port %d, latency %d: %d / %d matrices bit-exact and cycle-exact "
                   "(%s)" % ("comparator" if k[0] else "table", k[1], k[2], k[3], k[4], ok, n, ", ".join(sorted(encs))))
    out += ["", "## Post-synthesis (ASAP7 RVT TT at 1 GHz; Nangate45 typical as a sanity check; no place and route)", "",
            "| block | params | corner, clock | flow | area (um^2) | flops | reg-to-reg slack (ps) | WNS incl. I/O (ps) | "
            "power, default activity (mW) | power, co-sim activity (mW) |", "|" + "---|" * 10]
    last = {}
    for r in asic:
        last[r["name"]] = r
    for r in last.values():
        vp = r.get("vcd_power") or {}
        out.append("| %s | %s | %s, %.0f MHz | %s | %.0f | %d | %.0f | %.0f | %.2f | %s |" % (
            r["top"], " ".join("%s=%s" % kv for kv in r["params"].items()), r.get("corner", "ASAP7 RVT TT"),
            1e6 / r.get("clk_ps", 1000), r.get("flow", ""), r["area_um2"], r["flops"],
            r["reg2reg_slack_ps"], r["wns_ps"], 1e3 * (r["default_activity_power_w"] or 0),
            ", ".join("%.2f (%s)" % (1e3 * v, k) for k, v in vp.items()) or "-"))
    eng = next((r for r in last.values() if r["top"] == "huff_engine"), None)
    lane = next((r for r in reversed(list(last.values())) if r["top"] == "huff_lane" and r["params"].get("LOOKUP") == "1"
                 and r.get("corner", "ASAP7 RVT TT").startswith("ASAP7")
                 and r["wns_ps"] >= 0), None)   # a comparator lane that meets 1 GHz incl. I/O
    arb = next((r for r in reversed(list(last.values())) if r["top"] == "huff_arb" and r["wns_ps"] >= 0
                and r.get("corner", "ASAP7 RVT TT").startswith("ASAP7")), None)
    if eng is None and lane and arb:   # engine = 32 lanes + the arbiter (synthesized as blocks)
        eng = dict(area_um2=32 * lane["area_um2"] + arb["area_um2"])
        out += ["", "Engine (comparator lookup, PORT 8) = 32 x %s + %s: %.0f um^2 = %.4f mm^2 (blocks synthesized "
                "separately; both meet 1 GHz: lane WNS %.0f ps, arbiter WNS %.0f ps incl. 50 ps I/O budgets)." % (
                    lane["name"], arb["name"], eng["area_um2"], eng["area_um2"] / 1e6, lane["wns_ps"], arb["wns_ps"])]
    pts = [r for r in a5 if r.get("kind") == "point"]
    if pts:
        out += ["", "## Hardware per headline point (Orin Nano Super, beta = 102 GB/s)", "",
                "| model | budget | b_c (b/w) | theta* (G weights/s) | engines | area (mm^2) |", "|" + "---|" * 6]
        seen = set()
        for r in pts:
            if (r["model"], r["budget"]) in seen:
                continue
            seen.add((r["model"], r["budget"]))
            out.append("| %s | <=%s | %.2f | %.0f | %d | %s |" % (
                r["model"], r["budget"], r["b_c"], r["theta_star"] / 1e9, r["engines"],
                "%.3f" % (r["engines"] * eng["area_um2"] / 1e6) if eng else "-"))
    open(os.path.join(REPO, "results", "a4_report.md"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
