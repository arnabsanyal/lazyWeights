"""Entropy-aware GPTQ that writes llama.cpp's own block formats (Q4_K, Q5_K, Q6_K, IQ4_XS), so the
result is a valid GGUF (llama.cpp runs it packed; our coder codes the same codes: scripts/kq_code.py).

Per matrix (GGUF row = output row, blocks along the input dim), GPTQ in 256-column super-blocks; the grid of each
super-block is fitted at its start on the error-updated weights (lazy groups, no act-order) and the codes are chosen by
argmin_c (w - w_c)^2 / [H^-1]_jj^2 / norm + lam * L(c) (L: ideal lengths of the raw codes over the matrix, from the
previous pass; pass 0 = lam 0; as scripts/q4_gptq.py). The grid is where the format is geared to the coder:
  gamma  sub-block range = max(own range, gamma * largest sub-block range of the super-block), widened about its
         centre. gamma = 0: minmax per sub-block (codes spread over the whole alphabet, as llama.cpp's grids);
         gamma = 1: one step for the super-block (codes concentrate where the weights do: low entropy).
Formats (value of code c in sub-block j): Q4_K / Q5_K  d*sc_j*c - dmin*m_j  (32-wide, 6-bit sc / m, fp16 d / dmin);
Q6_K  d*sc_j*(c - 32)  (16-wide, int8 sc); IQ4_XS  d*ls_j*kv[c]  (32-wide, 6-bit ls, kv = IQ4_NL table).
Tensor types follow the llama.cpp recipe (--recipe, quantized with the importance matrix); tensors of other types and an
untied token_embd are copied from that file. The written GGUF is then scored exactly as scripts/a1_gguf.py does (read
back, dequantized by gguf-py, WikiText-2 PPL, llama-perplexity, lm-eval tasks) and sized packed and coded.

  .venv/bin/python scripts/kq_quant.py --model phi3 --recipe Q4_K_M --gamma 1 --lam 0.1
-> results/a1_frontier.jsonl (method kq, family packed and coded), GGUF kept in <model dir>/gguf/kq/
"""
import argparse
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
import a1_gguf  # noqa: E402
import kq_code  # noqa: E402
from a1_gguf import ARCH, LLAMA, PERMUTED, gguf  # noqa: E402
from lazyweights.quant import hessians  # noqa: E402
from lazyweights.evaltasks import tasks  # noqa: E402
from lazyweights.log import JsonlLogger  # noqa: E402
from lazyweights.models import LIN, LIN_FUSED, SPECS, build  # noqa: E402
from lazyweights.quant import calib, ppl  # noqa: E402
from lazyweights.quant import lengths  # noqa: E402

T = gguf.GGMLQuantizationType
KV = torch.tensor([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=torch.float32)
# format: (sub-block width, number of codes)
FMT = {T.Q4_K: (32, 16), T.Q5_K: (32, 32), T.Q6_K: (16, 64), T.IQ4_XS: (32, 16)}


def f16(x):
    return x.half().float()


def nearest(x, a, b, lv):
    """Index of the nearest level a * lv + b: x [N, nb, sw], a, b [N, nb]."""
    if bool((lv.diff() == 1).all()):
        return torch.clamp(torch.round((x - b[..., None]) / a[..., None] - lv[0]), 0, len(lv) - 1).long()
    return (x[..., None] - (a[..., None, None] * lv + b[..., None, None])).abs().argmin(-1)


def lsq(x, l, h, asym):
    """Importance-weighted least-squares grid for fixed levels l: min sum h (a l + b - x)^2 (b <= 0; b = 0 if not
    asym). x, l, h [N, nb, sw] -> a, b, err [N, nb]."""
    sw_, sl, sx = h.sum(-1), (h * l).sum(-1), (h * x).sum(-1)
    sll, slx = (h * l * l).sum(-1), (h * l * x).sum(-1)
    a0 = slx / sll.clamp(min=1e-30)
    if asym:
        D = sw_ * sll - sl * sl
        a = (sw_ * slx - sl * sx) / D.clamp(min=1e-30)
        b = (sll * sx - sl * slx) / D.clamp(min=1e-30)
        ok = (D > 0) & (b <= 0)
        a, b = torch.where(ok, a, a0), torch.where(ok, b, torch.zeros_like(b))
    else:
        a, b = a0, torch.zeros_like(a0)
    e = a[..., None] * l + b[..., None] - x
    return a, b, (h * e * e).sum(-1)


def fit(W, ty, gamma, hd, search=True, coarse=1.0):
    """W [N, 256], hd [256] importance (diag H) -> grid of one super-block: dict(a [N, nb], b [N, nb], side fields).
    search: as llama.cpp's make_qkx3_quants / make_qx_quants / IQ4 scale search, with the same importance weights:
    several starting scales, levels by rounding, importance-weighted least-squares refit, the least error kept.
    Else minmax (asym) / absmax (sym). Then gamma: step of each sub-block >= gamma * the super-block's largest step
    (asym: widened about the sub-block's centre)."""
    N = W.shape[0]
    sw, A = FMT[ty]
    x = W.reshape(N, -1, sw)
    h = hd.reshape(1, -1, sw).expand_as(x)
    lv = levels(ty)
    asym = ty in (T.Q4_K, T.Q5_K)
    if asym:
        mn, mx = torch.clamp(x.min(2).values, max=0), x.max(2).values
        rng = (mx - mn).clamp(min=1e-30)
        a, b = rng / (A - 1), mn
        if search:
            _, _, best = lsq(x, nearest(x, a, b, lv).float(), h, True)
            for k in range(37):          # make_qkx3_quants(.., rmin -0.9, rdelta 0.05, nstep 36)
                isc = (-0.9 + 0.05 * k + (A - 1)) / rng
                l = torch.clamp(torch.round(isc[..., None] * (x - mn[..., None])), 0, A - 1)
                ak, bk, e = lsq(x, l, h, True)
                ok = (e < best) & (ak > 0)
                a, b, best = torch.where(ok, ak, a), torch.where(ok, bk, b), torch.where(ok, e, best)
    else:
        i = x.abs().argmax(2, keepdim=True)
        smax = x.gather(2, i)[..., 0]                                  # signed value of largest magnitude
        smax = torch.where(smax.abs() > 0, smax, torch.full_like(smax, 1e-30))
        a, b = smax / lv[0], torch.zeros_like(smax)                    # largest weight -> the lowest level
        if search:
            _, _, best = lsq(x, lv[nearest(x, a, b, lv)], h, False)
            for k in range(-9, 10):      # make_qx_quants (Q6_K) / IQ4_XS ntry 7 style
                ak0 = smax / (lv[0] + (0.1 if ty == T.Q6_K else 1.0) * k)
                ak, _, e = lsq(x, lv[nearest(x, ak0, b, lv)], h, False)
                ok = e < best
                a, best = torch.where(ok, ak, a), torch.where(ok, e, best)
    big = gamma * a.abs().max(1, keepdim=True).values
    if asym:
        c = b + a * (A - 1) / 2
        a = torch.maximum(a, big) * coarse
        b = torch.clamp(c - a * (A - 1) / 2, max=0)
        d = f16(a.max(1).values / 63)
        d = torch.where(d > 0, d, torch.ones_like(d))
        sc = torch.clamp(torch.round(a / d[:, None]), 1, 63)
        dmin = f16((-b).max(1).values / 63)
        m = torch.where(dmin[:, None] > 0, torch.clamp(torch.round(-b / torch.where(dmin > 0, dmin, 1)[:, None]),
                                                      0, 63), torch.zeros_like(b))
        return dict(a=d[:, None] * sc, b=-(dmin[:, None] * m), d=d, dmin=dmin, sc=sc, m=m)
    a = torch.where(a.abs() < big, torch.sign(a) * big, a) * coarse
    a = torch.where(a == 0, torch.ones_like(a) * 1e-8, a)
    if ty == T.Q6_K:
        d = f16(a.abs().max(1).values / 127)
        d = torch.where(d > 0, d, torch.ones_like(d))
        sc = torch.clamp(torch.round(a / d[:, None]), -128, 127)
        sc = torch.where(sc == 0, torch.ones_like(sc), sc)             # all-zero sub-blocks (padded vocab rows)
        return dict(a=d[:, None] * sc, b=torch.zeros_like(sc), d=d, sc=sc)
    d = f16(a.abs().max(1).values / 31)                                # IQ4_XS: ls in [-32, 31]
    d = torch.where(d > 0, d, torch.full_like(d, 2.0 ** -24))         # no zero level: smallest fp16 for zero blocks
    ls = torch.clamp(torch.round(a / d[:, None]), -32, 31)
    ls = torch.where(ls == 0, torch.ones_like(ls), ls)
    return dict(a=d[:, None] * ls, b=torch.zeros_like(ls), d=d, ls=ls)


def levels(ty):
    A = FMT[ty][1]
    if ty == T.IQ4_XS:
        return KV.cuda()
    lv = torch.arange(A, dtype=torch.float32, device="cuda")
    return lv - 32 if ty == T.Q6_K else lv


@torch.no_grad()
def gptq_kq(W, H, ty, gamma, lam, L, damp=0.01, search=True, rtn=False, coarse=1.0):
    """W [N, K] fp32 (GGUF row order), H [K, K]. Returns (W_hat, codes [N, K] uint8, super-block grids, hist)."""
    sw, A = FMT[ty]
    lv = levels(ty)
    W = W.clone()
    N, K = W.shape
    H = torch.diag(torch.diag(H)) if rtn else H.clone()
    hd = torch.diag(H).clone()
    dead = hd == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(K, device=W.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    d2 = torch.diag(Hinv) ** 2
    step = float(lv.diff().mean())
    Q = torch.zeros_like(W)
    qa = torch.zeros(N, K, dtype=torch.uint8, device=W.device)
    hist = torch.zeros(A, dtype=torch.int64, device=W.device)
    grids, norm = [], None
    win = A if A <= 32 else 8
    for i1 in range(0, K, 256):
        W1 = W[:, i1:i1 + 256].clone()
        E1 = torch.zeros_like(W1)
        Hi = Hinv[i1:i1 + 256, i1:i1 + 256]
        g = fit(W1, ty, gamma, hd[i1:i1 + 256], search, coarse)
        grids.append(g)
        if norm is None:   # cost of one bit in squared grid steps at average Hessian weight (first super-block)
            norm = float((1 / d2).mean()) * float(((g["a"] * step) ** 2).mean())
        for i in range(256):
            a, b = g["a"][:, i // sw], g["b"][:, i // sw]
            w = W1[:, i]
            d = Hi[i, i]
            if ty == T.IQ4_XS:
                c = torch.arange(A, device=W.device).expand(N, A)
            else:
                q0 = torch.clamp(torch.round((w - b) / a + (32 if ty == T.Q6_K else 0)), 0, A - 1)
                c = torch.clamp(q0[:, None] + torch.arange(-win, win + 1, device=W.device), 0, A - 1).long()
            wc = a[:, None] * lv[c] + b[:, None]
            cost = (w[:, None] - wc) ** 2
            if lam > 0:
                cost = cost / (d * d) / norm + lam * L[c]
            k = cost.argmin(1, keepdim=True)
            q = c.gather(1, k)[:, 0]
            wq = wc.gather(1, k)[:, 0]
            hist += torch.bincount(q, minlength=A)
            Q[:, i1 + i] = wq
            qa[:, i1 + i] = q.to(torch.uint8)
            err = (w - wq) / d
            W1[:, i:] -= err[:, None] * Hi[i, i:][None]
            E1[:, i] = err
        W[:, i1 + 256:] -= E1 @ Hinv[i1:i1 + 256, i1 + 256:]
    return Q, qa, grids, hist


def side_bytes(ty, grids):
    """Super-block grids (list over K / 256 of [N, ...]) -> side bytes [N * K / 256, s] in block order."""
    cat = lambda k: torch.stack([g[k] for g in grids], 1).cpu()  # noqa: E731  [N, nsb, ...]
    h = lambda x: x.half().numpy().view(np.uint8).reshape(-1, 2)  # noqa: E731
    d = h(cat("d").reshape(-1))
    if ty in (T.Q4_K, T.Q5_K):
        sc = cat("sc").reshape(-1, 8).numpy().astype(np.uint8)
        m = cat("m").reshape(-1, 8).numpy().astype(np.uint8)
        dmin = h(cat("dmin").reshape(-1))
        s = np.concatenate([sc[:, :4] | ((sc[:, 4:] >> 4) << 6), m[:, :4] | ((m[:, 4:] >> 4) << 6),
                            (sc[:, 4:] & 15) | ((m[:, 4:] & 15) << 4)], 1)
        return np.concatenate([d, dmin, s], 1)
    if ty == T.Q6_K:
        return np.concatenate([cat("sc").reshape(-1, 16).numpy().astype(np.int8).view(np.uint8), d], 1)
    ls = (cat("ls").reshape(-1, 8).numpy() + 32).astype(np.uint16)
    hi = np.bitwise_or.reduce((ls >> 4) << (2 * np.arange(8, dtype=np.uint16)), axis=1).astype(np.uint16)
    lo = (ls & 15).astype(np.uint8)
    return np.concatenate([d, hi.view(np.uint8).reshape(-1, 2), lo[:, 0::2] | (lo[:, 1::2] << 4)], 1)


def blocks(ty, qa, grids):
    """Codes [N, K] + grids -> GGUF tensor bytes [N, K / 256 * block bytes]."""
    N, K = qa.shape
    q = qa.cpu().numpy().reshape(-1, 256)
    b = kq_code.join(ty, q, side_bytes(ty, grids))
    return np.ascontiguousarray(b.reshape(N, -1))


def permute(w, n_head):
    """conversion/llama.py LlamaModel.permute (HF q / k rows -> GGUF rows)."""
    return w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def main():
    from datasets import load_dataset
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(SPECS), required=True)
    ap.add_argument("--recipe", default="Q4_K_M", help="llama.cpp type whose per-tensor types are used")
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--no-search", action="store_true", help="minmax / absmax grids (no weighted scale search)")
    ap.add_argument("--coarse", type=float, default=1.0, help="step multiplier after the fit (> 1: coarser)")
    ap.add_argument("--score-only", action="store_true", help="score the existing GGUF of this config")
    ap.add_argument("--rtn", action="store_true", help="H -> diag(H): importance-weighted rounding, no GPTQ feedback")
    ap.add_argument("--seed", type=int, default=0,
                    help="calibration sample seed (A3 stability check; != 0: rows -> results/a3_kq_seeds.jsonl)")
    ap.add_argument("--n-cal", type=int, default=128)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--tasks", default="ppl,hellaswag,gsm8k,arc_easy,piqa")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    spec = SPECS[a.model]
    fused = spec.get("fused", False)
    torch.manual_seed(0)
    gdir = os.path.join(spec["dir"], "gguf")
    os.makedirs(os.path.join(gdir, "kq"), exist_ok=True)
    name = "%s_g%g_lam%g%s%s%s" % (a.recipe, a.gamma, a.lam, "_c%g" % a.coarse if a.coarse != 1 else "",
                                 "_minmax" if a.no_search else "", "_rtn" if a.rtn else "")
    if a.seed:
        name += "_s%d" % a.seed
    out = os.path.join(gdir, "kq", name + ".gguf")
    cfgd = dict(model=a.model, method="kq", qtype=a.recipe, gamma=a.gamma, lam=a.lam, passes=a.passes, search=not a.no_search, rtn=a.rtn, coarse=a.coarse,
                n_cal=a.n_cal, seq=a.seq, tasks=a.tasks)
    if a.seed:   # seed rows stay out of the A1 frontier (no picking the luckiest calibration draw)
        cfgd["seed"] = a.seed
    lg = JsonlLogger("a1_frontier" if not a.seed else "a3_kq_seeds", cfg=cfgd)
    t0 = time.time()
    log = os.path.join(REPO, "results", "logs", "kq", "quant_%s.log" % a.model)
    ref_f = os.path.join(gdir, "kq", "%s_ref.gguf" % a.recipe)
    if not os.path.exists(ref_f):
        with open(log, "a") as f:
            subprocess.run([os.path.join(LLAMA, "build", "bin", "llama-quantize"), "--imatrix",
                            os.path.join(gdir, "imatrix.dat"), os.path.join(gdir, "f16.gguf"), ref_f, a.recipe],
                           check=True, stdout=f, stderr=subprocess.STDOUT)
    ref = gguf.GGUFReader(ref_f)
    rt = {t.name: t for t in ref.tensors}
    model, tok, _ = build(spec)
    cfg = model.config
    tied = cfg.tie_word_embeddings
    tmap = gguf.get_tensor_name_map(ARCH[a.model], cfg.num_hidden_layers)
    def gname(hn):
        g = tmap.get_name(hn, try_suffixes=(".weight", ".bias"))
        assert g is not None, (hn, cfg.num_hidden_layers)
        return g
    cal, src = calib(tok, a.n_cal, a.seq, a.seed)
    new, types = {}, {}

    def one(hn, lin, H):
        gn = "token_embd.weight" if (hn == "lm_head.weight" and tied) else gname(hn)
        ty = rt[gn].tensor_type
        types[ty.name] = types.get(ty.name, 0) + 1
        if ty not in FMT:
            return
        W = lin.weight.float()
        perm = None
        if a.model in PERMUTED and hn.endswith(("q_proj.weight", "k_proj.weight")):
            nh = cfg.num_attention_heads if "q_proj" in hn else cfg.num_key_value_heads
            perm = permute(torch.arange(W.shape[0], device=W.device), nh)
            W = W[perm]
        L = torch.zeros(FMT[ty][1], device="cuda")
        for p in range(a.passes + 1 if a.lam > 0 else 1):
            Q, qa, grids, h = gptq_kq(W, H, ty, a.gamma, a.lam if p > 0 else 0, L, search=not a.no_search, rtn=a.rtn,
                                      coarse=a.coarse)
            L = lengths(h)
        new[gn] = blocks(ty, qa, grids)
        assert new[gn].shape == tuple(rt[gn].data.shape), (gn, new[gn].shape, rt[gn].data.shape)
        dq = gguf.quants.dequantize(new[gn], ty).reshape(Q.shape)          # the file decodes to what GPTQ used
        assert np.allclose(dq, Q.cpu().numpy(), rtol=0, atol=1e-6 * float(Q.abs().max())), gn
        if perm is not None:
            Q = Q[torch.argsort(perm)]
        lin.weight.copy_(Q.half())

    quant_s = None
    if not (a.score_only and os.path.exists(out)):   # score-only: an existing file (e.g. scoring interrupted)
        with torch.no_grad():
            for li, layer in enumerate(model.model.layers):
                t1 = time.time()
                names = LIN_FUSED if fused else LIN
                g = [(n, getattr(layer.self_attn if n in LIN[:4] + LIN_FUSED[:2] else layer.mlp, n)) for n in names]
                Hs = hessians(model, cal, g, "down_proj")
                for n, lin in g:
                    sub = "self_attn" if n in LIN[:4] + LIN_FUSED[:2] else "mlp"
                    one("model.layers.%d.%s.%s.weight" % (li, sub, n), lin, Hs.pop(n))
                print("layer %2d %.0fs" % (li, time.time() - t1), flush=True)
            lm = model.lm_head
            one("lm_head.weight", lm, hessians(model, cal, [("lm_head", lm)], None)["lm_head"])
        quant_s = time.time() - t0
        del model
        torch.cuda.empty_cache()
        # write the GGUF: metadata and the other tensors from the recipe file, our tensors in place
        w = gguf.GGUFWriter(out, arch=gguf.MODEL_ARCH_NAMES[ARCH[a.model]])
        for field in ref.fields.values():
            if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
                continue
            vt = field.types[0]
            w.add_key_value(field.name, field.contents(), vt, sub_type=field.types[-1] if vt == gguf.GGUFValueType.ARRAY
                            else None)
        data = {t.name: new.get(t.name, t.data) for t in ref.tensors}
        for t in ref.tensors:
            w.add_tensor_info(t.name, data[t.name].shape, data[t.name].dtype, data[t.name].nbytes, t.tensor_type)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_ti_data_to_file()
        for t in ref.tensors:
            w.write_tensor_data(data[t.name], tensor_endianess=ref.endianess)
        w.close()
        del data, new
        print("wrote %s (%.1f MB), %.0fs" % (out, os.path.getsize(out) / 1e6, quant_s), flush=True)
    # score the file as a1_gguf does
    from transformers import AutoConfig, AutoModelForCausalLM
    hf_dir = os.path.join(spec["dir"], "hf_fp16")
    hcfg = AutoConfig.from_pretrained(hf_dir)
    with torch.device("cuda"):
        model = AutoModelForCausalLM.from_config(hcfg, dtype=torch.float16).eval()
    W, bits, gtypes = a1_gguf.read_gguf(out, a.model, hcfg, [n for n, _ in model.named_parameters()])
    a1_gguf.fill(model, W)
    del W
    nw = a1_gguf.n_weights(model)
    rec = dict(cfgd, calib=src, n_weights=nw, gguf_types=gtypes, file=out, hf=spec["hf"], quant_s=quant_s,
               ppl_protocol="wikitext2 test, all tokens, 512-token chunks")
    names = [t for t in a.tasks.split(",") if t]
    if "ppl" in names:
        text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
        rec["ppl_full"] = ppl(model, torch.tensor(tok(text)["input_ids"]))
        print("ppl_full %.4f" % rec["ppl_full"], flush=True)
        r = subprocess.run([os.path.join(LLAMA, "build", "bin", "llama-perplexity"), "-m", out, "-f",
                            os.path.join(gdir, "wiki.test.raw"), "-c", "512", "-ngl", "99", "-b", "512"],
                           capture_output=True, text=True)
        import re
        m = re.findall(r"Final estimate: PPL = ([0-9.]+)", r.stdout + r.stderr)
        rec["llamacpp_ppl"] = float(m[-1]) if m else None
        print("llama-perplexity %s" % rec["llamacpp_ppl"], flush=True)
    rec.update(tasks(model, tok, [t for t in names if t != "ppl"], a.batch))
    del model
    kq_code._R.update(reader=gguf.GGUFReader(out), tied=tied)
    with Pool(24) as pool:
        ts = list(pool.imap(kq_code.job, range(len(kq_code._R["reader"].tensors))))
    kq_code._R.clear()
    coded = sum(r["coded_bits"] for r in ts)
    assert sum(r["packed_bits"] for r in ts) == bits
    H = {}
    for r in ts:
        if r["coded"]:
            H.setdefault(r["type"], []).append((r["H"], r["N"] * r["K"]))
    rec["entropy_by_type"] = {k: sum(h * n for h, n in v) / sum(n for _, n in v) for k, v in H.items()}
    rec["seconds"] = time.time() - t0
    lg.log(dict(rec, family="packed", bpw=bits / nw, tensor_bits=bits))
    lg.log(dict(rec, family="coded", bpw=coded / nw, tensor_bits=coded, packed_tensor_bits=bits))
    print("%s %s: packed %.4f, coded %.4f b/w, H %s" % (a.model, name, bits / nw, coded / nw,
                                                     {k: round(v, 3) for k, v in rec["entropy_by_type"].items()}),
          flush=True)


if __name__ == "__main__":
    main()
