"""D2 V4/V5 golden codecs: tANS (FSE) and length-limited canonical Huffman, ECW1 tiling, per-lane bitstreams.

Container 'ECWL' (little-endian), shared by both:
  uint32 magic 0x4C574345 ("ECWL"), N, K, TN, TK, A, kind (1 = Huffman, 2 = tANS), table_log, n_tiles
  table: Huffman -> uint8 code_length[A]; tANS -> uint16 norm_count[A] (sums to 2^table_log)     (padded to 4)
  meta[n_tiles]: uint32 lane_start_bit[32], uint32 lane_end_bit[32], uint32 tans_init_state[32] (0 for Huffman)
  uint32 payload_words; uint32 payload[]   (bit b = (payload[b >> 5] >> (b & 31)) & 1, LSB-first)
Symbols are assigned to lanes exactly as in ECW1 (tile row-major, symbol i -> lane i % 32, step i / 32).
Huffman: lane reads forward from lane_start_bit; codes are canonical, stored bit-reversed (deflate style), max
length 11 bits, so an 11-bit LSB-first peek indexes a 2^11-entry LUT. tANS: FSE tables (zstd spread), the lane
reads backward from lane_end_bit; init state in meta (0..2^table_log - 1).
"""
import struct

import numpy as np

from lazyweights.huffman import code_lengths
from lazyweights.rans import tile_grid, _shape_classes

MAGIC = 0x4C574345
LANES = 32
HUFF_LMAX = 11
TANS_LOG = 11


# ── tables ───────────────────────────────────────────────────────────────────

def limited_lengths(counts, lmax=HUFF_LMAX):
    c = np.asarray(counts, np.int64)
    k = 0
    while True:
        cc = np.where(c > 0, np.maximum(1, c >> k), 0)
        L = code_lengths(cc)
        if L.max() <= lmax:
            if (c > 0).sum() == 1:
                L = np.where(c > 0, 1, 0)
            return L.astype(np.uint8)
        k += 1


def canonical_codes(lengths):
    """Canonical codes (MSB-first values) for the given lengths; returns codes array."""
    L = np.asarray(lengths, np.int64)
    codes = np.zeros(len(L), np.int64)
    code = 0
    for l in range(1, int(L.max()) + 1):
        for s in np.nonzero(L == l)[0]:
            codes[s] = code
            code += 1
        code <<= 1
    return codes


def bitrev(v, n):
    """Reverse the low n bits of each v (elementwise)."""
    v = np.asarray(v, np.int64)
    n = np.asarray(n, np.int64)
    r = np.zeros_like(v)
    for i in range(int(n.max()) if n.size else 0):
        take = i < n
        r |= np.where(take, ((v >> i) & 1) << np.maximum(n - 1 - i, 0), 0)
    return r


def huff_lut(lengths, lmax=HUFF_LMAX):
    """2^lmax LUT indexed by the next lmax bits LSB-first -> (symbol, length)."""
    L = np.asarray(lengths, np.int64)
    codes = canonical_codes(L)
    lut_sym = np.zeros(1 << lmax, np.int64)
    lut_len = np.zeros(1 << lmax, np.int64)
    for s in np.nonzero(L)[0]:
        rc = int(bitrev(np.array([codes[s]]), np.array([L[s]]))[0])
        for hi in range(1 << (lmax - L[s])):
            idx = rc | (hi << L[s])
            lut_sym[idx], lut_len[idx] = s, L[s]
    return lut_sym, lut_len


def tans_norm(counts, log=TANS_LOG):
    c = np.asarray(counts, np.int64)
    M = 1 << log
    f = np.where(c > 0, np.maximum(1, (c * M) // c.sum()), 0)
    while f.sum() > M:
        f[np.argmax(np.where(f > 1, f, -1))] -= 1
    while f.sum() < M:
        f[np.argmax(c)] += 1
    return f


def tans_tables(norm, log=TANS_LOG):
    M = 1 << log
    A = len(norm)
    spread = np.zeros(M, np.int64)
    pos, step = 0, (M >> 1) + (M >> 3) + 3
    for s in range(A):
        for _ in range(int(norm[s])):
            spread[pos] = s
            pos = (pos + step) & (M - 1)
    nxt = np.array(norm, np.int64).copy()
    d_sym = np.zeros(M, np.int64)
    d_nb = np.zeros(M, np.int64)
    d_new = np.zeros(M, np.int64)
    for u in range(M):
        s = spread[u]
        ns = nxt[s]
        nxt[s] += 1
        nb = log - int(np.floor(np.log2(ns)))
        d_sym[u], d_nb[u], d_new[u] = s, nb, (ns << nb) - M
    # encode table: for symbol s, its states (in increasing u) map from x in [n_s, 2 n_s)
    cum = np.concatenate([[0], np.cumsum(norm)])
    enc = np.zeros(M, np.int64)
    fill = cum[:-1].copy()
    for u in range(M):
        s = spread[u]
        enc[fill[s]] = M + u
        fill[s] += 1
    return dict(sym=d_sym, nb=d_nb, new=d_new, enc=enc, cum=cum, norm=np.asarray(norm, np.int64), M=M, log=log)


# ── lane layout ──────────────────────────────────────────────────────────────

def _lane_matrix(codes, tiles, idxs, tn, tk):
    S = np.stack([codes[tiles[t][0]:tiles[t][1], tiles[t][2]:tiles[t][3]].reshape(-1) for t in idxs])
    n = tn * tk
    steps = (n + LANES - 1) // LANES
    pad = steps * LANES - n
    act = np.ones((len(idxs), steps * LANES), bool)
    if pad:
        S = np.pad(S, ((0, 0), (0, pad)))
        act[:, n:] = False
    return S.reshape(len(idxs), steps, LANES), act.reshape(len(idxs), steps, LANES)


def _pack_bits(vals, nbits):
    """vals/nbits: flat arrays in emission order. Returns (uint32 payload, start bit of each element)."""
    nb = nbits.astype(np.int64)
    start = np.concatenate([[0], np.cumsum(nb)[:-1]])
    total = int(nb.sum())
    bits = np.zeros(total + 64, np.uint8)
    maxb = int(nb.max()) if len(nb) else 0
    for i in range(maxb):
        sel = nb > i
        bits[start[sel] + i] = (vals[sel] >> i) & 1
    # LSB-first packing into uint32
    nw = (total + 31) // 32
    b = bits[:nw * 32].reshape(nw, 32).astype(np.uint64)
    words = (b << np.arange(32, dtype=np.uint64)).sum(1).astype(np.uint32)
    return words, start, total


def encode_layer(codes, A, kind, TN=64, TK=128):
    codes = np.asarray(codes, np.int64)
    N, K = codes.shape
    cnt = np.bincount(codes.ravel(), minlength=A)
    tiles = tile_grid(N, K, TN, TK)
    if kind == 1:
        L = limited_lengths(cnt)
        rc = bitrev(canonical_codes(L), L.astype(np.int64))
        table = L
        log = HUFF_LMAX
    else:
        norm = tans_norm(cnt)
        T = tans_tables(norm)
        table = norm.astype(np.uint16)
        log = TANS_LOG
    # build every lane's emission list (value, nbits) in decoder read order
    lane_vals, lane_nb, init = {}, {}, {}
    for (tn, tk), idxs in sorted(_shape_classes(tiles).items()):
        S, act = _lane_matrix(codes, tiles, idxs, tn, tk)
        B, steps, _ = S.shape
        if kind == 1:
            V = np.where(act, rc[S], 0)
            NB = np.where(act, L[S], 0)
            for j, t in enumerate(idxs):
                lane_vals[t], lane_nb[t] = V[j].T, NB[j].T          # [32, steps] forward order
                init[t] = np.zeros(LANES, np.int64)
        else:
            M = T["M"]
            x = np.full((B, LANES), M, np.int64)
            V = np.zeros((B, steps, LANES), np.int64)
            NB = np.zeros((B, steps, LANES), np.int64)
            nrm = T["norm"]
            maxb = T["log"] - np.floor(np.log2(np.maximum(nrm, 1))).astype(np.int64)
            minplus = nrm << maxb
            for t in range(steps - 1, -1, -1):
                a = act[:, t]
                s = S[:, t]
                nb = maxb[s] - (x < minplus[s])
                nb = np.where(a, nb, 0)
                V[:, t] = x & ((1 << nb) - 1)
                NB[:, t] = nb
                xs = x >> nb
                newx = T["enc"][np.where(a, T["cum"][s] + xs - nrm[s], 0)]   # inactive lanes: index unused
                x = np.where(a, newx, x)
            for j, t in enumerate(idxs):
                # decoder reads backward from the lane end: emission order = reverse step order
                lane_vals[t], lane_nb[t] = V[j].T[:, ::-1], NB[j].T[:, ::-1]
                init[t] = x[j] - M
    # lay out lanes contiguously: tile-major, lane-major
    order_t = range(len(tiles))
    vals = np.concatenate([lane_vals[t].ravel() for t in order_t]) if tiles else np.zeros(0, np.int64)
    nbs = np.concatenate([lane_nb[t].ravel() for t in order_t]) if tiles else np.zeros(0, np.int64)
    words, start, total = _pack_bits(vals, nbs)
    lane_bits = np.stack([lane_nb[t].sum(1) for t in order_t])          # [tiles, 32]
    starts = np.concatenate([[0], np.cumsum(lane_bits.ravel())[:-1]]).reshape(lane_bits.shape)
    ends = starts + lane_bits
    meta = np.concatenate([starts, ends, np.stack([init[t] for t in order_t])], 1).astype("<u4")
    hdr = struct.pack("<9I", MAGIC, N, K, TN, TK, A, kind, log, len(tiles))
    tb = table.astype("<u2" if kind == 2 else np.uint8).tobytes()
    tb += b"\0" * ((-len(tb)) % 4)
    return hdr + tb + meta.tobytes() + struct.pack("<I", len(words)) + words.astype("<u4").tobytes()


def decode_layer(buf):
    magic, N, K, TN, TK, A, kind, log, ntiles = struct.unpack_from("<9I", buf, 0)
    assert magic == MAGIC
    o = 36
    if kind == 1:
        L = np.frombuffer(buf, np.uint8, A, o).astype(np.int64)
        o += A + ((-A) % 4)
        lut_sym, lut_len = huff_lut(L)
    else:
        norm = np.frombuffer(buf, "<u2", A, o).astype(np.int64)
        o += 2 * A + ((-(2 * A)) % 4)
        T = tans_tables(norm, log)
    meta = np.frombuffer(buf, "<u4", ntiles * 96, o).reshape(ntiles, 3, 32).astype(np.int64)
    o += 4 * 96 * ntiles
    (nw,) = struct.unpack_from("<I", buf, o)
    o += 4
    W = np.concatenate([np.frombuffer(buf, "<u4", nw, o).astype(np.uint64), np.zeros(2, np.uint64)])
    tiles = tile_grid(N, K, TN, TK)
    out = np.full((N, K), -1, np.int64)

    def peek(pos, n):
        w = pos >> 5
        v = (W[w] | (W[w + 1] << np.uint64(32))) >> (pos & 31).astype(np.uint64)
        return (v & ((np.uint64(1) << n.astype(np.uint64)) - np.uint64(1))).astype(np.int64)

    for (tn, tk), idxs in sorted(_shape_classes(tiles).items()):
        n = tn * tk
        steps = (n + LANES - 1) // LANES
        idxs = np.asarray(idxs)
        res = np.zeros((len(idxs), steps, LANES), np.int64)
        lane_active = (np.arange(steps)[:, None] * LANES + np.arange(LANES)[None]) < n    # [steps, 32]
        if kind == 1:
            pos = meta[idxs, 0].copy()
            for t in range(steps):
                a = lane_active[t][None, :].repeat(len(idxs), 0)
                code = peek(pos, np.full(pos.shape, HUFF_LMAX))
                s, ln = lut_sym[code], lut_len[code]
                res[:, t] = s
                pos = np.where(a, pos + ln, pos)
            ok = np.array_equal(pos, meta[idxs, 1])
        else:
            pos = meta[idxs, 1].copy()
            st = meta[idxs, 2].copy()
            for t in range(steps):
                a = lane_active[t][None, :].repeat(len(idxs), 0)
                s, nb, new = T["sym"][st], T["nb"][st], T["new"][st]
                res[:, t] = s
                v = peek(np.maximum(pos - nb, 0), nb)
                st = np.where(a, new + v, st)
                pos = np.where(a, pos - nb, pos)
            ok = np.array_equal(pos, meta[idxs, 0])
        if not ok:
            raise ValueError("lane bit pointers did not land on the lane boundaries")
        for j, t in enumerate(idxs):
            r0, r1, c0, c1 = tiles[t]
            out[r0:r1, c0:c1] = res[j].reshape(-1)[:n].reshape(tn, tk)
    return out
