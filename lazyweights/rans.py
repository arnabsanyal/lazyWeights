"""Golden reference rANS codec for ecw (docs/bitstream.md). NumPy only, vectorized over tiles.

Decoding is bit-exact by construction; every public decode path is checked with np.array_equal in tests.
"""
import struct

import numpy as np

PROB_BITS = 12
M = 1 << PROB_BITS
L = 1 << 16
LANES = 32
MAGIC = 0x31574345
META_BITS_PER_TILE = 32 * 32 + 32


# ── tables ───────────────────────────────────────────────────────────────────

def normalize_freqs(counts):
    """Deterministic normalization to sum M (docs/bitstream.md Section 2). Returns uint16 array."""
    c = np.asarray(counts, dtype=np.int64)
    C = int(c.sum())
    if C == 0:
        raise ValueError("empty histogram")
    nz = c > 0
    if int(nz.sum()) > M:
        raise ValueError("more distinct symbols than M")
    f = np.zeros_like(c)
    f[nz] = np.maximum(1, (c[nz] * M) // C)
    s = int(f.sum())
    while s > M:
        cand = np.where(f > 1, f, -1)
        i = int(np.argmax(cand))  # argmax returns lowest index on ties
        f[i] -= 1
        s -= 1
    while s < M:
        i = int(np.argmax(c))
        f[i] += 1
        s += 1
    return f.astype(np.uint16)


class Tables(object):
    """Stack of frequency tables [T, A] with cum and slot->symbol lookup."""

    def __init__(self, freqs):
        self.freq = np.atleast_2d(np.asarray(freqs, dtype=np.uint16))
        assert np.all(self.freq.astype(np.int64).sum(1) == M)
        self.n, self.A = self.freq.shape
        f64 = self.freq.astype(np.uint64)
        self.f = f64
        self.cum = np.concatenate([np.zeros((self.n, 1), np.uint64), np.cumsum(f64, 1)[:, :-1]], 1)
        self.lut = np.zeros((self.n, M), dtype=np.int32)
        for t in range(self.n):
            self.lut[t] = np.repeat(np.arange(self.A, dtype=np.int32), self.freq[t].astype(np.int64))

    @classmethod
    def from_symbols(cls, symbols, row_table, A, n_tables):
        """One table per table index from the symbols of rows mapped to it."""
        sym = np.asarray(symbols)
        rt = np.asarray(row_table)
        freqs = []
        for t in range(n_tables):
            cnt = np.bincount(sym[rt == t].ravel(), minlength=A)
            if cnt.sum() == 0:      # table referenced by no row: placeholder, never decoded
                cnt[0] = 1
            freqs.append(normalize_freqs(cnt))
        return cls(np.stack(freqs))


# ── tiling helpers ───────────────────────────────────────────────────────────

def tile_grid(N, K, TN, TK):
    """List of (r0, r1, c0, c1) in row-major tile order."""
    out = []
    for r0 in range(0, N, TN):
        for c0 in range(0, K, TK):
            out.append((r0, min(N, r0 + TN), c0, min(K, c0 + TK)))
    return out


def _shape_classes(tiles):
    cls = {}
    for idx, (r0, r1, c0, c1) in enumerate(tiles):
        cls.setdefault((r1 - r0, c1 - c0), []).append(idx)
    return cls


# ── batched tile encode / decode (B tiles of identical shape) ────────────────

def _encode_batch(sym, tab, tables):
    """sym, tab: int arrays [B, n] (row-major tile symbols and their table index).
    Returns (states uint32 [B, 32], list of uint16 word arrays)."""
    B, n = sym.shape
    steps = (n + LANES - 1) // LANES
    pad = steps * LANES - n
    active = np.ones((B, steps * LANES), bool)
    if pad:
        sym = np.pad(sym, ((0, 0), (0, pad)))
        tab = np.pad(tab, ((0, 0), (0, pad)))
        active[:, n:] = False
    sym = sym.reshape(B, steps, LANES)
    tab = tab.reshape(B, steps, LANES)
    active = active.reshape(B, steps, LANES)
    f_all = tables.f[tab, sym]           # [B, steps, 32] uint64
    c_all = tables.cum[tab, sym]
    x = np.full((B, LANES), L, dtype=np.uint64)
    buf = np.zeros((B, n + 1), dtype=np.uint16)
    cnt = np.zeros(B, dtype=np.int64)
    rows = np.arange(B)[:, None]
    for t in range(steps - 1, -1, -1):
        a = active[:, t]
        f = f_all[:, t]
        c = c_all[:, t]
        emit = a & (x >= (f << np.uint64(20)))
        # emission order within a step: decreasing lane index
        rev = emit[:, ::-1]
        pos_rev = np.cumsum(rev, 1) - rev
        pos = pos_rev[:, ::-1] + cnt[:, None]
        bi, li = np.nonzero(emit)
        buf[bi, pos[bi, li]] = (x[bi, li] & np.uint64(0xFFFF)).astype(np.uint16)
        cnt += emit.sum(1)
        x = np.where(emit, x >> np.uint64(16), x)
        fs = np.where(a, f, np.uint64(1))
        xe = ((x // fs) << np.uint64(PROB_BITS)) + (x % fs) + c
        x = np.where(a, xe, x)
    del rows
    assert np.all(x < (1 << 32))
    words = [buf[b, :cnt[b]][::-1].copy() for b in range(B)]
    return x.astype(np.uint32), words


def _decode_batch(states, words, tab, n, tables):
    """states uint32 [B, 32]; words: list of uint16 arrays; tab int [B, n]. Returns symbols [B, n]."""
    B = states.shape[0]
    steps = (n + LANES - 1) // LANES
    pad = steps * LANES - n
    active = np.ones((B, steps * LANES), bool)
    if pad:
        tab = np.pad(tab, ((0, 0), (0, pad)))
        active[:, n:] = False
    tab = tab.reshape(B, steps, LANES)
    active = active.reshape(B, steps, LANES)
    maxw = max([len(w) for w in words] + [1])
    W = np.zeros((B, maxw + 1), dtype=np.uint64)
    for b, w in enumerate(words):
        W[b, :len(w)] = w
    ptr = np.zeros(B, dtype=np.int64)
    x = states.astype(np.uint64)
    out = np.zeros((B, steps, LANES), dtype=np.int32)
    mask = np.uint64(M - 1)
    for t in range(steps):
        a = active[:, t]
        tb = tab[:, t]
        slot = x & mask
        s = tables.lut[tb, slot.astype(np.int64)]
        f = tables.f[tb, s]
        c = tables.cum[tb, s]
        xd = f * (x >> np.uint64(PROB_BITS)) + slot - c
        x = np.where(a, xd, x)
        out[:, t] = s
        need = a & (x < np.uint64(L))
        pos = np.cumsum(need, 1) - need + ptr[:, None]   # increasing lane order
        bi, li = np.nonzero(need)
        x[bi, li] = (x[bi, li] << np.uint64(16)) | W[bi, pos[bi, li]]
        ptr += need.sum(1)
    lens = np.array([len(w) for w in words])
    if not (np.all(ptr == lens) and np.all(x == L)):
        raise ValueError("rANS integrity check failed")
    return out.reshape(B, steps * LANES)[:, :n]


# ── layer-level API ──────────────────────────────────────────────────────────

class LayerStream(object):
    def __init__(self, N, K, TN, TK, tables, table_fmt_id, row_table, states, tile_words):
        self.N, self.K, self.TN, self.TK = N, K, TN, TK
        self.tables = tables
        self.table_fmt_id = np.asarray(table_fmt_id, dtype=np.uint8)
        self.row_table = np.asarray(row_table, dtype=np.uint8)
        self.states = states            # uint32 [n_tiles, 32]
        self.tile_words = tile_words    # list of uint16 arrays
        self.n_tiles = len(tile_words)

    # sizes in bits
    @property
    def payload_bits(self):
        return 16 * sum(len(w) for w in self.tile_words)

    @property
    def meta_bits(self):
        return META_BITS_PER_TILE * self.n_tiles

    @property
    def header_bits(self):
        return 8 * len(self.header_bytes())

    def header_bytes(self):
        h = struct.pack("<8I", MAGIC, self.N, self.K, self.TN, self.TK, self.tables.A, self.tables.n,
                        self.n_tiles)
        h += _pad4(self.table_fmt_id.tobytes())
        h += self.tables.freq.astype("<u2").tobytes()
        h += _pad4(self.row_table.tobytes())
        return h

    def to_bytes(self):
        payload = b"".join(w.astype("<u2").tobytes() for w in self.tile_words)
        offs = np.cumsum([0] + [2 * len(w) for w in self.tile_words[:-1]]).astype("<u4")
        meta = np.concatenate([offs[:, None], self.states.astype("<u4")], 1).astype("<u4").tobytes()
        return self.header_bytes() + meta + struct.pack("<I", len(payload)) + payload

    @classmethod
    def from_bytes(cls, buf):
        magic, N, K, TN, TK, A, nt, ntiles = struct.unpack_from("<8I", buf, 0)
        assert magic == MAGIC
        o = 32
        fid = np.frombuffer(buf, np.uint8, nt, o); o += _r4(nt)
        freq = np.frombuffer(buf, "<u2", nt * A, o).reshape(nt, A); o += 2 * nt * A
        rt = np.frombuffer(buf, np.uint8, N, o); o += _r4(N)
        meta = np.frombuffer(buf, "<u4", ntiles * 33, o).reshape(ntiles, 33); o += 4 * 33 * ntiles
        (plen,) = struct.unpack_from("<I", buf, o); o += 4
        payload = np.frombuffer(buf, "<u2", plen // 2, o)
        offs = list(meta[:, 0] // 2) + [plen // 2]
        words = [payload[offs[i]:offs[i + 1]].copy() for i in range(ntiles)]
        return cls(N, K, TN, TK, Tables(freq), fid, rt, meta[:, 1:].copy(), words)

    def decode(self, tile_order=None):
        """Decode to symbols [N, K]. tile_order permutes the order tiles are decoded in (independence test)."""
        tiles = tile_grid(self.N, self.K, self.TN, self.TK)
        out = np.full((self.N, self.K), -1, dtype=np.int32)
        order = list(range(len(tiles))) if tile_order is None else list(tile_order)
        pos_in_order = {t: i for i, t in enumerate(order)}
        for (tn, tk), idxs in sorted(_shape_classes(tiles).items()):
            idxs = sorted(idxs, key=lambda t: pos_in_order[t])
            tab = np.stack([np.repeat(self.row_table[tiles[t][0]:tiles[t][1]].astype(np.int64), tk)
                            for t in idxs])
            sym = _decode_batch(self.states[idxs], [self.tile_words[t] for t in idxs], tab, tn * tk, self.tables)
            for j, t in enumerate(idxs):
                r0, r1, c0, c1 = tiles[t]
                out[r0:r1, c0:c1] = sym[j].reshape(tn, tk)
        return out

    def decode_tile(self, t):
        tiles = tile_grid(self.N, self.K, self.TN, self.TK)
        r0, r1, c0, c1 = tiles[t]
        tab = np.repeat(self.row_table[r0:r1].astype(np.int64), c1 - c0)[None]
        return _decode_batch(self.states[[t]], [self.tile_words[t]], tab, tab.shape[1], self.tables)[0] \
            .reshape(r1 - r0, c1 - c0)


def encode_layer(sym, row_table=None, A=16, TN=64, TK=128, tables=None, table_fmt_id=None, chunk=4096):
    """sym: int [N, K] in [0, A). row_table: table index per row (default all 0)."""
    sym = np.asarray(sym)
    N, K = sym.shape
    if row_table is None:
        row_table = np.zeros(N, dtype=np.int64)
    row_table = np.asarray(row_table, dtype=np.int64)
    n_tab = int(row_table.max()) + 1
    if tables is None:
        tables = Tables.from_symbols(sym, row_table, A, n_tab)
    if table_fmt_id is None:
        table_fmt_id = np.arange(tables.n)
    tiles = tile_grid(N, K, TN, TK)
    states = np.zeros((len(tiles), LANES), np.uint32)
    words = [None] * len(tiles)
    for (tn, tk), idxs in sorted(_shape_classes(tiles).items()):
        for s0 in range(0, len(idxs), chunk):
            part = idxs[s0:s0 + chunk]
            S = np.stack([sym[tiles[t][0]:tiles[t][1], tiles[t][2]:tiles[t][3]].reshape(-1) for t in part])
            T = np.stack([np.repeat(row_table[tiles[t][0]:tiles[t][1]], tk) for t in part])
            st, ws = _encode_batch(S.astype(np.int64), T, tables)
            states[part] = st
            for j, t in enumerate(part):
                words[t] = ws[j]
    return LayerStream(N, K, TN, TK, tables, table_fmt_id, row_table, states, words)


# ── scalar reference (tests only; mirrors the bitstream text literally) ──────

def decode_tile_scalar(states, words, freq, n):
    """Single-table, single-tile literal decoder (for cross-checking the vectorized path)."""
    freq = [int(v) for v in freq]
    cum = [0]
    for v in freq[:-1]:
        cum.append(cum[-1] + v)
    x = [int(s) for s in states]
    ptr = 0
    out = []
    for t in range((n + LANES - 1) // LANES):
        act = [l for l in range(LANES) if t * LANES + l < n]
        for l in act:
            slot = x[l] & (M - 1)
            s = max(i for i in range(len(freq)) if cum[i] <= slot and freq[i] > 0)
            x[l] = freq[s] * (x[l] >> PROB_BITS) + slot - cum[s]
            out.append(s)
        for l in act:
            if x[l] < L:
                x[l] = (x[l] << 16) | int(words[ptr])
                ptr += 1
    assert ptr == len(words) and all(v == L for v in x)
    return out


def entropy_bits(counts):
    c = np.asarray(counts, dtype=np.float64)
    c = c[c > 0]
    p = c / c.sum()
    return float(-(c * np.log2(p)).sum())


def ideal_bits(sym, freq):
    """Cross-entropy of symbols under the quantized table (what rANS should approach)."""
    f = np.asarray(freq, dtype=np.float64)
    cnt = np.bincount(np.asarray(sym).ravel(), minlength=len(f))
    m = cnt > 0
    return float(-(cnt[m] * np.log2(f[m] / M)).sum())


def _pad4(b):
    return b + b"\0" * ((-len(b)) % 4)


def _r4(n):
    return n + ((-n) % 4)
