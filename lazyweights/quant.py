"""Quantizer core shared by the scripts: the calibration set, WikiText-2 perplexity, the minmax grid, and GPTQ with
entropy-aware code selection.

GPTQ (Hessian-compensated, column-by-column rounding) picks each weight's code by
    argmin_c  (w - w_c)^2 / d_j^2 / norm  +  lam * L(c)
instead of nearest rounding; L(c) is the ideal code length -log2 p(c) of the re-centered code in this tensor (p from
the previous pass; pass 0 = nearest rounding). norm makes lam dimensionless per tensor: mean(1/d^2) * mean(scale^2).
lam = 0 is plain GPTQ. The llama.cpp-format quantizer (scripts/kq_quant.py) uses its own grid fit on top of this.
"""
import numpy as np
import torch

CFG = dict(bits=4)                                  # code width of qdq / grid / gptq
GROUP = {"tensor": None, "channel": 0, "g128": 128}


class Stop(Exception):
    pass


def qdq(w, gran, parts=1, rnd=torch.round):
    """w [N, K] fp32 -> (dequantized fp32 [N, K], re-centered code histogram [2^bits]).
    'tensor': one grid per ONNX tensor (parts = row blocks); rnd = torch.trunc truncates instead of rounding."""
    B = CFG["bits"]
    LV, A = 2 ** B - 1, 2 ** B
    N, K = w.shape
    if gran == "tensor":
        x = w.reshape(parts, -1)
    else:
        x = w.reshape(-1, K if gran == "channel" else GROUP[gran])
    lo = torch.clamp(x.min(1).values, max=0)
    hi = torch.clamp(x.max(1).values, min=0)
    s = ((hi - lo) / LV).half().float()
    s = torch.where(s > 0, s, torch.ones_like(s))
    z = torch.clamp(torch.round(-lo / s), 0, LV)
    q = torch.clamp(rnd(x / s[:, None]) + z[:, None], 0, LV)
    hist = torch.bincount(((q - z[:, None] + A // 2) % A).to(torch.int64).ravel(), minlength=A)
    return ((q - z[:, None]) * s[:, None]).reshape(N, K), hist


def calib(tok, n, seq, seed=0):
    from datasets import load_dataset
    try:
        ds = load_dataset("allenai/c4", data_files="en/c4-validation.00000-of-00008.json.gz", split="train")
        src = "c4 validation shard 0"
    except Exception as e:                      # noqa: BLE001
        print("c4 unavailable (%s), using wikitext-2 train" % e, flush=True)
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        src = "wikitext-2 train"
    rng = np.random.default_rng(seed)   # seed 0: the calibration set of every A1 row
    out = []
    for i in rng.permutation(len(ds)):
        ids = tok(ds[int(i)]["text"])["input_ids"]
        if len(ids) > seq:
            j = int(rng.integers(0, len(ids) - seq))
            out.append(ids[j:j + seq])
        if len(out) == n:
            break
    return torch.tensor(out), src


@torch.no_grad()
def ppl(model, ids, bs=16):
    nlls = []
    chunks = [ids[i:i + 512] for i in range(0, len(ids), 512)]
    for b in range(0, len(chunks), bs):
        part = chunks[b:b + bs]
        L = max(len(c) for c in part)
        if any(len(c) != L for c in part):                    # last partial chunk alone
            for c in part:
                nlls += _nll(model, c[None])
            continue
        nlls += _nll(model, torch.stack(part))
    return float(np.exp(np.mean(nlls)))


def _nll(model, x):
    x = x.cuda()
    lg = model(x).logits[:, :-1].float()
    lp = torch.log_softmax(lg, -1).gather(-1, x[:, 1:, None])[..., 0]
    return (-lp.mean(1)).tolist()


def grid(w):
    """w [N, g] -> per-row scale, zp (minmax)."""
    lo = torch.clamp(w.min(1).values, max=0)
    hi = torch.clamp(w.max(1).values, min=0)
    LV = 2 ** CFG["bits"] - 1
    s = ((hi - lo) / LV).half().float()
    s = torch.where(s > 0, s, torch.ones_like(s))
    return s, torch.clamp(torch.round(-lo / s), 0, LV)


@torch.no_grad()
def gptq(W, H, gran, lam, L, parts=1, blk=128, damp=0.01, act_order=False, exp=1, out=None):
    """W [N, K] fp32, H [K, K]; L [2^bits] code lengths of re-centered codes. Returns (W_hat, re-centered hist).
    Candidates: every code for <= 4 bits, nearest +- 8 above.
    act_order: columns in decreasing diag(H) order; g128 then uses static groups (grids of the original weights per
    128 consecutive input columns, as AutoGPTQ static_groups). exp: distortion weight 1 / [H^-1]_jj^exp in the
    entropy-aware selection (1: the GPTQ / OBS error). out (dict): filled with the integer codes q [N, K] uint8 and the
    grid s, z [N, groups] (CPU tensors, original column order)."""
    B = CFG["bits"]
    LV, A = 2 ** B - 1, 2 ** B
    W = W.clone()
    N, K = W.shape
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    perm = None
    if act_order:
        perm = torch.argsort(torch.diag(H), descending=True)
        if gran == "g128":
            sg, zg = grid(W.reshape(-1, 128))
            sg, zg = sg.reshape(N, K // 128), zg.reshape(N, K // 128)
        W, H = W[:, perm], H[perm][:, perm]
    H += damp * torch.mean(torch.diag(H)) * torch.eye(K, device=W.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    d2 = torch.diag(Hinv) ** 2
    if gran == "tensor":
        s, z = grid(W.reshape(parts, -1))                  # one grid per ONNX tensor
        s, z = s.repeat_interleave(N // parts), z.repeat_interleave(N // parts)
    elif gran == "channel":
        s, z = grid(W)
    norm = float((1 / d2 ** exp).mean()) * float((s ** 2).mean()) if gran != "g128" else None
    if gran == "g128":                                     # mean scale^2 over the RTN grid of the original weights
        s0, _ = grid(W.reshape(-1, 128))
        norm = float((1 / d2 ** exp).mean()) * float((s0 ** 2).mean())
    win = LV if B <= 4 else 8
    offs = torch.arange(-win, win + 1, device=W.device, dtype=torch.float32)
    Q = torch.zeros_like(W)
    hist = torch.zeros(A, dtype=torch.int64, device=W.device)
    if out is not None:
        qa = torch.zeros(N, K, dtype=torch.uint8, device=W.device)
        ng = K // 128 if gran == "g128" else 1
        sa = torch.zeros(N, ng, device=W.device)
        za = torch.zeros(N, ng, device=W.device)
    for i1 in range(0, K, blk):
        i2 = min(i1 + blk, K)
        W1 = W[:, i1:i2].clone()
        E1 = torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            j = i1 + i
            if gran == "g128" and perm is not None:
                gi = int(perm[j]) // 128
                s, z = sg[:, gi], zg[:, gi]
            elif gran == "g128" and j % 128 == 0:
                s, z = grid(W1[:, i:i + 128] if i + 128 <= W1.shape[1] else W[:, j:j + 128])
            w = W1[:, i]
            d = Hi[i, i]
            q = torch.clamp(torch.round(w / s) + z, 0, LV)
            if lam > 0:
                c = torch.clamp(q[:, None] + offs[None], 0, LV)                # [N, candidates]
                wc = (c - z[:, None]) * s[:, None]
                r = ((c - z[:, None] + A // 2) % A).long()
                cost = (w[:, None] - wc) ** 2 / (d * d) ** exp / norm + lam * L[r]
                q = c.gather(1, cost.argmin(1, keepdim=True))[:, 0]
            hist += torch.bincount(((q - z + A // 2) % A).long(), minlength=A)
            wq = (q - z) * s
            Q[:, j] = wq
            if out is not None:
                jo = int(perm[j]) if perm is not None else j
                qa[:, jo] = q.to(torch.uint8)
                g = jo // 128 if gran == "g128" else 0
                sa[:, g], za[:, g] = s, z
            err = (w - wq) / d
            W1[:, i:] -= err[:, None] * Hi[i, i:][None]
            E1[:, i] = err
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    if perm is not None:
        Q = Q[:, torch.argsort(perm)]
    if out is not None:
        out.update(q=qa.cpu(), s=sa.cpu(), z=za.cpu())
    return Q, hist


def lengths(hist):
    p = hist.double() / hist.sum()
    return (-torch.log2(p.clamp(min=1e-6))).float()


def ent(h):
    p = h.double() / h.sum()
    p = p[p > 0]
    return float(-(p * torch.log2(p)).sum())


@torch.no_grad()
def hessians(model, cal, targets, stop_at, bs=8):
    Hs = {n: torch.zeros(m.in_features, m.in_features, device="cuda") for n, m in targets}
    cnt = {n: 0 for n, _ in targets}
    hooks = []
    for n, m in targets:
        def pre(mod, inp, n=n):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            Hs[n] += x.T @ x
            cnt[n] += len(x)
            if n == stop_at:
                raise Stop
        hooks.append(m.register_forward_pre_hook(pre))
    for b in range(0, len(cal), bs):
        try:
            model(cal[b:b + bs].cuda())
        except Stop:
            pass
    for h in hooks:
        h.remove()
    return {n: 2 * Hs[n] / cnt[n] for n in Hs}
