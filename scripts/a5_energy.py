"""A5 energy model: energy per token on the Orin Nano Super with the hardware Huffman decode unit.

  E_hw = E_twin - e_dram * (S_fixed - S_coded) + E_dec

- E_twin: measured J per token of our packed twin (module input rail, idle included; results/a2_spec_energy.jsonl,
  the kq11g run (kq7 + CUDA graphs): int8 activations, multi-query attention, Q5_K / Q6_K twins at the GGUF width, llama.cpp's prompt-lookup
  drafter). With the unit the GPU runs the twin
  kernel unchanged (t_hw = t_twin, as in a5_system.py), so the twin's energy stands except for the DRAM bytes that are
  no longer read.
- e_dram: energy per byte read from DRAM at the module input, measured on the device (kernels/membw_energy.cu,
  results/a5_energy/orin): the slope of VDD_IN power vs read bandwidth of a streaming read kernel on a 1 GiB buffer
  minus the same slope on a 1 MiB (L2-resident) buffer, each throttled to 5 bandwidths (same instructions per byte).
  This is the DRAM + memory controller + PHY + regulator energy the decode unit saves for every byte it does not read.
- S_fixed - S_coded: weight bytes per token saved by entropy coding (results/a5_system.jsonl).
- E_dec: the decode unit, from ASAP7 post-synthesis power (results/a4_asic.jsonl): lane at co-simulation activity,
  arbiter at default activity. Two bounds: clock-gated (engines powered only while decoding: n weights / engine rate)
  and always-on (all engines for the whole token time). Pre-layout: no clock tree or wires (factor DEC_MARGIN shown).

  .venv/bin/python scripts/a5_energy.py   -> results/a5_energy.jsonl, results/a5_energy.md
  A5_DRAFTER=lookup .venv/bin/python scripts/a5_energy.py   -> results/a5_energy_lk.md (our drafter, Orin)
"""
import json
import os
import re
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from lazyweights.log import JsonlLogger, config_hash  # noqa: E402

F_CLK = 1e9
DEC_MARGIN = 2.0   # sensitivity: decoder power x2 for clock tree, wires, post-layout effects
# per device: speed/energy run (results/a2_spec_energy.jsonl config), module input rail, title. Orin: bandwidth-bound
# (time with the unit in [t_twin x S_coded / S_fixed, t_twin]); Nano: compute-bound (t_hw = t_twin), llama.cpp on the CPU
# A5_DRAFTER=lookup: our runs with our own prompt-lookup drafter (Table IV's speculation columns; Orin only)
# -> results/a5_energy_lk.md
LK = os.environ.get("A5_DRAFTER") == "lookup"
DEVICES = {
    "orin": dict(run=config_hash(dict(device="orin", root="results/a2_orin/energy_kq",
                                      ours="results/a2_orin/energy_kq11g" + ("lk" if LK else ""),
                                      tag="kq11g" + ("lk" if LK else ""))), rail="VDD_IN", title="Orin Nano Super", bw_bound=True),
    "nano": dict(run=config_hash(dict(device="nano", root="results/a2_nano/energy_kq", ours="results/a2_nano/energy_kq",
                                      tag="")), rail="POM_5V_IN", title="Jetson Nano", bw_bound=False),
}
CMP = {("phi3", "1%"): "cmp_IQ4_XS_im", ("phi3", "3.5%"): "cmp_Q3_K_M_im", ("mistral", "1%"): "cmp_Q5_K_M_no",
       ("mistral", "3.5%"): "cmp_IQ3_M_im", ("smollm", "1%"): "cmp_Q5_K_M_no", ("smollm", "3.5%"): "cmp_IQ4_XS_im"}


def dram_energy(d, rail):
    T, P = [], []
    for line in open(os.path.join(d, "tegrastats.log")):
        m = re.search(rail + r" (\d+)(?:mW|/)", line)   # Orin "VDD_IN 5000mW/...", Nano "POM_5V_IN 999/999"
        if m:
            T.append(float(line.split()[0]))
            P.append(int(m.group(1)) / 1e3)
    T, P = np.array(T), np.array(P)
    runs = [json.loads(l) for l in open(os.path.join(d, "runs.jsonl"))]
    idle = float(np.median(P[T < min(r["t_start"] for r in runs) - 0.5]))
    fit = {}
    for big in (True, False):
        rs = [r for r in runs if (r["buffer_mib"] >= 256) == big]
        bw = np.array([r["gbps"] * 1e9 for r in rs])
        dp = np.array([P[(T > r["t_start"] + 0.5) & (T < r["t_end"] - 0.2)].mean() - idle for r in rs])
        (a, b), *_ = np.linalg.lstsq(np.stack([np.ones_like(bw), bw], 1), dp, rcond=None)
        fit["dram" if big else "l2"] = dict(slope_j_per_byte=b, intercept_w=a, bw=bw.tolist(), dp_w=dp.tolist(),
                                            resid_max_w=float(np.abs(a + b * bw - dp).max()))
    return fit["dram"]["slope_j_per_byte"] - fit["l2"]["slope_j_per_byte"], fit, idle


def energies(run):
    rows = [json.loads(l) for l in open(os.path.join(REPO, "results", "a2_spec_energy.jsonl"))]
    rows = {json.dumps(r, sort_keys=True): r for r in rows if r["config_hash"] == run}.values()   # drop re-logged rows

    def agg(m, rt, b, ph):   # -> J per token, s per token, weight passes per token (1 for plain greedy)
        rs = [r for r in rows if r["model"] == m and r["runtime"] == rt and r["bundle"] == b and r["phase"] == ph]
        n = sum(r["tokens"] for r in rs)
        if not n:
            return None
        passes = sum(r["tokens"] / (r.get("tokens_per_pass") or 1) for r in rs) / n
        return sum(r["power_w"] * r["seconds"] for r in rs) / n, sum(r["seconds"] for r in rs) / n, passes
    return agg


def main():
    asic = {}
    for l in open(os.path.join(REPO, "results", "a4_asic.jsonl")):
        r = json.loads(l)
        asic[r["name"]] = r

    def power(r):   # co-simulation activity (mean over traces) when available, else default activity
        v = r.get("vcd_power_w") or (np.mean(list(r["vcd_power"].values())) if r.get("vcd_power") else None)
        return v or r["default_activity_power_w"]
    p_lane, p_arb = power(asic["lane_cmp"]), power(asic["arb8_d350"])
    p_eng = 32 * p_lane + p_arb
    pts, nano_eng = {}, {}
    for l in open(os.path.join(REPO, "results", "a5_system.jsonl")):
        r = json.loads(l)
        if r.get("kind") == "point" and r.get("device", "orin") == "orin":
            pts[(r["model"], r["budget"])] = r   # bytes and engine rate per bundle (device independent)
        elif r.get("kind") == "nano_point":
            nano_eng[(r["model"], r["budget"])] = r["engines"]
    out = ["# A5 energy model: the hardware decode unit on the Jetsons", "",
           "Decode engine power (ASAP7, 1 GHz, pre-layout): 32 lanes x %.2f mW (co-simulation activity) + arbiter "
           "%.2f mW (default activity) = %.1f mW per engine." % (1e3 * p_lane, 1e3 * p_arb, 1e3 * p_eng), "",
           "E_hw = E_twin (measured) - e_dram x bytes saved + E_dec, per weight pass x passes per token (1 for plain "
           "greedy; prompt-lookup speculation, draft 7, no drafter model: measured passes of our runtime). t_hw = "
           "t_twin (conservative); on the Orin also the bandwidth-bound end t_twin x S_coded / S_fixed (the GPU reads "
           "the coded bytes from DRAM). E_dec: clock-gated (engines on while decoding) / always-on (all engines for the "
           "whole token); last column: always-on with decoder power x%.0f. Baseline = llama.cpp comparator in the same "
           "mode (CUDA on the Orin, CPU on the Nano)." % DEC_MARGIN]
    for dev, cfg in DEVICES.items():
        d = os.path.join(REPO, "results", "a5_energy", dev)
        if not os.path.exists(os.path.join(d, "runs.jsonl")):
            out += ["", "## %s: DRAM energy not measured yet (%s)" % (cfg["title"], os.path.relpath(d, REPO))]
            continue
        e_dram, fit, idle = dram_energy(d, cfg["rail"])
        agg = energies(cfg["run"])
        lg = JsonlLogger("a5_energy", cfg=dict(device=dev, f_clk=F_CLK, run=cfg["run"], dec_margin=DEC_MARGIN))
        lg.log(dict(kind="dram_fit", e_dram_j_per_byte=e_dram, idle_w=idle,
                    **{k + "_" + kk: vv for k, v in fit.items() for kk, vv in v.items()}))
        out += ["", "## %s" % cfg["title"], "",
                "DRAM read energy at the module input (%s, kernels/membw_energy.cu): slope %.1f pJ/B (1 GiB buffer) - "
                "%.1f pJ/B (L2-resident) = **%.1f pJ/byte (%.2f pJ/bit)**; idle %.2f W; fit residual max %.2f / %.2f W."
                % (cfg["rail"], 1e12 * fit["dram"]["slope_j_per_byte"], 1e12 * fit["l2"]["slope_j_per_byte"],
                   1e12 * e_dram, 1e12 * e_dram / 8, idle, fit["dram"]["resid_max_w"], fit["l2"]["resid_max_w"])]
        for ph, title in (("plain", "plain greedy"), ("spec", "prompt-lookup speculation")):
            out += ["", "### %s, %s" % (cfg["title"], title), "",
                    "| model | budget | passes / token | llama.cpp baseline ms / J | ours software ms / J | twin = ours + "
                    "hardware ms | bytes saved / token | DRAM energy saved | E_dec gated / always-on | ours + hardware J "
                    "| time vs baseline [bandwidth-bound, conservative] | energy vs baseline | energy (always-on x%.0f) |"
                    % DEC_MARGIN, "|" + "---|" * 13]
            for (m, bud), r in pts.items():
                b = r["bundle"]
                base = agg(m, "llama.cpp", CMP[(m, bud)], ph)
                ours_sw, twin = agg(m, "llmrt", b, ph), agg(m, "llmrt", b + "_pk", ph)
                if not (base and twin):
                    continue
                engines = r["engines"] if dev == "orin" else nano_eng.get((m, bud), 1)
                k = twin[2]
                saved_b = k * (r["S_fixed"] - r["S_coded"])
                e_save = e_dram * saved_b
                rate = r["engine_weights_per_cycle"] * F_CLK
                e_gated = k * r["n_matmul"] / rate * p_eng
                e_on = engines * p_eng * twin[1]
                e_hw = twin[0] - e_save + e_gated
                e_hw_worst = twin[0] - e_save + DEC_MARGIN * e_on
                t_lo = twin[1] * r["S_coded"] / r["S_fixed"] if cfg["bw_bound"] else twin[1]
                row = dict(kind="point", device=dev, phase=ph, model=m, budget=bud, bundle=b, comparator=CMP[(m, bud)],
                           passes_per_token=k, e_base_j=base[0], t_base_s=base[1],
                           e_sw_j=ours_sw[0] if ours_sw else None, t_sw_s=ours_sw[1] if ours_sw else None,
                           e_twin_j=twin[0], t_twin_s=twin[1], bytes_saved=saved_b, e_dram_saved_j=e_save,
                           e_dec_gated_j=e_gated, e_dec_on_j=e_on, engines=engines, e_hw_j=e_hw,
                           e_hw_worst_j=e_hw_worst, vs_base=e_hw / base[0] - 1, vs_base_worst=e_hw_worst / base[0] - 1,
                           vs_twin=e_hw / twin[0] - 1, t_vs_base=twin[1] / base[1] - 1, t_hw_lo_s=t_lo,
                           t_lo_vs_base=t_lo / base[1] - 1)
                lg.log(row)
                out.append("| %s | <=%s | %.2f | %.1f / %.3f | %s | %.1f | %.0f MB | %.1f mJ | %.1f / %.1f mJ | %.3f | "
                           "%+.0f%% .. %+.0f%% | %+.0f%% | %+.0f%% |" % (
                    m, bud, k, 1e3 * base[1], base[0],
                    "%.1f / %.3f" % (1e3 * ours_sw[1], ours_sw[0]) if ours_sw else "-", 1e3 * twin[1],
                    saved_b / 1e6, 1e3 * e_save, 1e3 * e_gated, 1e3 * e_on, e_hw, 100 * row["t_lo_vs_base"],
                    100 * row["t_vs_base"], 100 * row["vs_base"], 100 * row["vs_base_worst"]))
    open(os.path.join(REPO, "results", "a5_energy_lk.md" if LK else "a5_energy.md"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
