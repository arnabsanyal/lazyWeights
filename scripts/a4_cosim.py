"""A4 step 2: Verilator co-simulation of the SystemVerilog engine (hw/huff/rtl/) against the C model and the
reference decoder: decoded symbols bit-exact and cycles / stalls / fetched words identical to the C model, on a set
of matrices covering both symbol kinds (huf8, huf4p pair symbols), the longest codes (11 bits), per-tensor, per-row,
g128 and K-quant bundles, for both lookups and memory configurations with and without refill stalls.

  .venv/bin/python scripts/a4_cosim.py
Rows -> results/a4_cosim.jsonl (one per matrix and configuration)
"""
import json
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
from a4_csim import BUNDLES, NAS  # noqa: E402
from lazyweights.log import JsonlLogger  # noqa: E402

PICK = {   # bundle -> matrices (first / middle / last layer, lm_head for two)
    "smollm_kq_IQ4_XS_g1_lam0": ["l0.qkv", "l0.o", "l12.down", "l23.up", "lm_head"],
    "smollm_kq_Q5_K_M_g1_lam0": ["l0.q", "l11.gate", "l23.down"],
    "phi3_kq_IQ4_XS_g1_lam0_c1.25": ["l0.qkv", "l16.o", "l31.down"],
    "phi3_kq_IQ4_XS_g0_lam0_c1.25": ["l0.qkv", "l16.o", "l31.down"],
    "mistral_kq_IQ4_XS_g1_lam0_c1.5": ["l0.q", "l0.k", "l0.v", "l15.gate", "l31.down"],
    "phi3_q4_g128_lam0.1": ["l0.qkv", "l31.o"],
    "q8_gptq_huf8": ["l0.qkv", "l16.up", "l31.o"],
    "mistral_q8_tensor_lam0.3": ["l0.qkv", "l31.o"],
    "smollm_q4_channel_lam0.1": ["l0.o", "l23.down", "lm_head"],
}
CONFIGS = [  # (lookup, buf, fifo, port, lat)
    (1, 64, 4, 8, 8), (0, 64, 4, 8, 8), (1, 64, 4, 6, 8), (1, 48, 8, 8, 32), (1, 42, 2, 8, 1)]


def main():
    jobs = ["%s/%s/rt_bundles/%s:%s" % (NAS, BUNDLES[b], b, n) for b, ns in PICK.items() for n in ns]
    for c in CONFIGS:
        L, B, F, P, LAT = c
        exe = subprocess.check_output([os.path.join(REPO, "hw", "huff", "build_cosim.sh"), str(L), str(B), str(F),
                                       str(P)], text=True).strip().splitlines()[-1]
        p = subprocess.run([exe, str(LAT)] + jobs, capture_output=True, text=True)
        lg = JsonlLogger("a4_cosim", cfg=dict(lookup=L, buf=B, fifo=F, port=P, lat=LAT))
        rows = [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")]
        for r in rows:
            r["bundle"] = os.path.basename(r["bundle"])
            lg.log(r)
        print("lookup %d buf %d fifo %d port %d lat %d: %d matrices, %d ok, rc %d %s" % (
            L, B, F, P, LAT, len(rows), sum(r["ok"] for r in rows), p.returncode, p.stderr.strip()[-300:]), flush=True)
        if p.returncode or len(rows) != len(jobs):
            raise SystemExit("co-simulation stopped early: %d of %d matrices" % (len(rows), len(jobs)))


if __name__ == "__main__":
    main()
