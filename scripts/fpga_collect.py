"""FPGA bring-up results -> results/a4_fpga.jsonl (one row per job, with the build's utilization / timing) and
results/a4_fpga.md. Input: the host's JSON lines (scripts/fpga_host.py, run on the board host), the job index
(scripts/fpga_jobs.py) and the Vivado reports of hw/huff/fpga/build.tcl (build/fpga/nexys/).

  .venv/bin/python scripts/fpga_collect.py results_hw.jsonl
"""
import json
import os
import re
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from lazyweights.log import JsonlLogger  # noqa: E402

B = os.path.join(REPO, "build", "fpga", "nexys")


def util():
    t = open(os.path.join(B, "util.rpt")).read()
    g = lambda k: int(re.search(r"^\| %s\s*\|\s*([\d.]+)" % re.escape(k), t, re.M).group(1).split(".")[0])
    u = dict(luts=g("Slice LUTs"), ffs=g("Slice Registers"), bram36=float(re.search(
        r"^\| Block RAM Tile\s*\|\s*([\d.]+)", t, re.M).group(1)), dsp=g("DSPs"))
    tr = open(os.path.join(B, "timing.rpt")).read()
    v = re.search(r"WNS\(ns\).*\n.*\n\s*(-?[\d.]+)\s+(-?[\d.]+)\s+\d+\s+\d+\s+(-?[\d.]+)", tr)
    u.update(wns_ns=float(v.group(1)), whs_ns=float(v.group(3)))
    return u


def main():
    rows = [json.loads(l) for l in open(sys.argv[1])]
    idx = {r["job"]: r for r in map(json.loads, open(os.path.join(REPO, "build", "fpga", "jobs", "index.jsonl")))}
    u = util()
    cfg = dict(board="Digilent Nexys 4 DDR", part="xc7a100tcsg324-1", clk_mhz=25, uart_baud=1000000, lanes=32,
               lookup=1, buf=64, fifo=4, port=8, lat=8, payload_words=16384, tool="Vivado 2025.2", **u)
    lg = JsonlLogger("a4_fpga", cfg=cfg)
    for r in rows:
        i = idx.get(r["job"], {})
        r.update(bundle=i.get("bundle"), matrix=i.get("matrix"), enc=i.get("enc"))
        lg.log(r)
    ok = sum(r["ok"] for r in rows)
    sym = sum(r.get("symbols", 0) for r in rows)
    mis = sum(r.get("mismatch", 0) for r in rows)
    cyc, ccyc = sum(r.get("cycles", 0) for r in rows), sum(r.get("c_cycles", 0) for r in rows)
    mats = sorted({(r["bundle"], r["matrix"]) for r in rows})
    md = ["# FPGA bring-up of the Huffman tile engine (Nexys 4 DDR, XC7A100T)", "",
          "32-lane engine (hw/huff/rtl, comparator lookup, buffer 64, FIFO 4, port 8) at %d MHz; payload in block RAM "
          "(4 copies = 8 read ports), every word returned %d cycles after its grant; driven over the USB-UART from the "
          "ZCU102 (scripts/fpga_host.py). Jobs: tile slices (first / middle / last) of the co-simulation's matrices "
          "(scripts/fpga_jobs.py); reference: the C model on the same slice." % (cfg["clk_mhz"], cfg["lat"]), "",
          "- jobs: %d, ok: %d; matrices: %d (%s)" % (len(rows), ok, len(mats), ", ".join(sorted({r["enc"] for r in rows}))),
          "- symbols: %d, mismatches: %d" % (sym, mis),
          "- cycles: %d on the board, %d in the C model; stalls and words equal on every job: %s" % (
              cyc, ccyc, all(r.get("stalls") == r.get("c_stalls") and r.get("words") == r.get("c_words") for r in rows)),
          "- utilization (whole design): %d LUTs (%.0f%%), %d FFs, %.0f BRAM36, %d DSP; timing at 25 MHz: WNS %+.3f ns, "
          "WHS %+.3f ns" % (u["luts"], 100 * u["luts"] / 63400, u["ffs"], u["bram36"], u["dsp"], u["wns_ns"], u["whs_ns"]),
          "- engine alone (out of context, hw/huff/fpga ooc): 36,145 LUTs, 5,211 FFs, 0 BRAM; critical path ~19.6 ns "
          "(~50 MHz)"]
    open(os.path.join(REPO, "results", "a4_fpga.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
