"""Runtime bundle for the uint8 phi3 ONNX model (weight-only, per-tensor uint8 + fp16 scale + uint8 zero point).

The ONNX graph has DequantizeLinear (DQ) only on initializers and no QuantizeLinear, so the DQs are folded:
  - DQ -> MatMul (161): weights stay uint8 codes q; the runtime computes y = s * sum_k x_k (q_kn - z)
    (exact algebra; the weight is converted once, in a register). Stored transposed, [N, K] row-major.
  - DQ -> Gather (embedding): codes kept, a row is dequantized when gathered.
  - DQ -> norm weights / rotary cos-sin caches: constant-folded at conversion, exactly as ORT does
    (fp32 (q - z) * s rounded to fp16), stored as fp32.

Bundle directory: manifest.txt + weights.bin (every blob 256-byte aligned).
manifest.txt lines:
  cfg <key> <value>
  t <name> <enc> <N> <K> <scale> <zp> <offset> <nbytes>
enc: f32 (N*K float32), raw8 (N*K uint8, row-major [N, K]), huf8 (see below).

huf8 blob (length-limited canonical Huffman, max code length 11, one table per tensor):
  uint16 lut[2048]            index = next 11 stream bits LSB-first -> sym | len << 8
  uint32 lane_start[T][32]    absolute bit offset of each lane in payload, T = (N/64) * (K/256)
  uint32 payload[W + 2]       bit b = (payload[b >> 5] >> (b & 31)) & 1; 2 zero words of tail padding
Tiles are 64 rows x 256 columns, tile t = (n / 64) * (K / 256) + k / 256. Lane l of a tile codes row l, then
row l + 32 of the tile, 256 symbols each, in column order. Codes are canonical, stored bit-reversed (LSB-first).
"""
import os

import numpy as np

from lazyweights.lanecodecs import bitrev, canonical_codes, limited_lengths

TN, TK, LMAX, LANES = 64, 256, 11, 32
ALIGN = 256


def lane_order(q):
    """[N, K] codes -> 1-D stream order (tile, lane, half, col)."""
    N, K = q.shape
    t = q.reshape(N // TN, TN, K // TK, TK).transpose(0, 2, 1, 3)          # [NT, KT, 64, 256]
    t = t.reshape(N // TN, K // TK, 2, LANES, TK).transpose(0, 1, 3, 2, 4)  # [NT, KT, 32, 2, 256]
    return np.ascontiguousarray(t).reshape(-1)


def lane_order_inv(s, N, K):
    t = s.reshape(N // TN, K // TK, LANES, 2, TK).transpose(0, 1, 3, 2, 4).reshape(N // TN, K // TK, TN, TK)
    return np.ascontiguousarray(t.transpose(0, 2, 1, 3)).reshape(N, K)


def build_lut(lengths):
    L = np.asarray(lengths, np.int64)
    codes = canonical_codes(L)
    lut = np.zeros(1 << LMAX, np.uint16)
    for s in np.nonzero(L)[0]:
        rc = int(bitrev(np.array([codes[s]]), np.array([L[s]]))[0])
        idx = rc + (np.arange(1 << (LMAX - L[s])) << L[s])
        lut[idx] = s | (L[s] << 8)
    return lut


def encode_huf8(q):
    """uint8 [N, K] -> huf8 blob (bytes) and coded payload bits."""
    N, K = q.shape
    assert N % TN == 0 and K % TK == 0, (N, K)
    hist = np.bincount(q.ravel(), minlength=256)
    L = limited_lengths(hist, LMAX).astype(np.int64)
    if (L > 0).sum() == 1:                       # degenerate single-symbol tensor: length-1 code
        L = np.where(hist > 0, 1, 0)
    codes = canonical_codes(L)
    rcode = bitrev(codes, np.maximum(L, 1)).astype(np.int64)
    s = lane_order(q)
    ln = L[s]
    pos = np.cumsum(ln) - ln                     # int64 bit position of each symbol
    total = int(pos[-1] + ln[-1])
    T = (N // TN) * (K // TK)
    starts = pos[::TK * 2].reshape(T, LANES).astype(np.uint32)
    assert total < 2 ** 32
    val = rcode[s]
    w = pos >> 5
    sh = pos & 31
    lo = (val << sh) & 0xFFFFFFFF                # codes are <= 11 bits: at most two words
    hi = np.where(sh > 0, val >> np.where(sh > 0, 32 - sh, 1), 0)
    nw = (total + 31) // 32 + 2
    # bits of different codes are disjoint, so a (float64-exact) sum is an OR
    words = np.bincount(w, weights=lo.astype(np.float64), minlength=nw)
    words += np.bincount(w + 1, weights=hi.astype(np.float64), minlength=nw + 1)[:nw]
    words = words.astype(np.uint64).astype(np.uint32)
    blob = build_lut(L).tobytes() + starts.tobytes() + words.tobytes()
    return blob, total, hist


def decode_huf8(blob, N, K):
    """NumPy reference decoder (slow, per lane; for tests)."""
    T = (N // TN) * (K // TK)
    lut = np.frombuffer(blob, np.uint16, 1 << LMAX)
    starts = np.frombuffer(blob, np.uint32, T * LANES, 2 << LMAX).reshape(T, LANES)
    words = np.frombuffer(blob, np.uint32, offset=(2 << LMAX) + T * LANES * 4).astype(np.uint64)
    stream = np.zeros(T * LANES * 2 * TK, np.uint8)
    n = 2 * TK
    bp = starts.reshape(-1).astype(np.int64)     # all lanes in lock step
    out = np.zeros((T * LANES, n), np.uint8)
    for i in range(n):
        wi = bp >> 5
        sh = (bp & 31).astype(np.uint64)
        v = ((words[wi] | (words[wi + 1] << np.uint64(32))) >> sh) & np.uint64((1 << LMAX) - 1)
        e = lut[v.astype(np.int64)]
        out[:, i] = e & 255
        bp += (e >> 8).astype(np.int64)
    stream[:] = out.reshape(-1)
    return lane_order_inv(stream, N, K)


def ort_dequant(q, s, z):
    """ORT DequantizeLinear with an fp16 scale: fp32 (q - z) * s, rounded to fp16."""
    return ((q.astype(np.float32) - np.float32(z)) * np.float32(s)).astype(np.float16).astype(np.float32)


class Writer:
    def __init__(self, path):
        os.makedirs(path, exist_ok=True)
        self.path = path
        self.f = open(os.path.join(path, "weights.bin"), "wb")
        self.lines = []
        self.off = 0

    def cfg(self, k, v):
        self.lines.append("cfg %s %s" % (k, v))

    def add(self, name, enc, N, K, scale, zp, blob):
        pad = (-self.off) % ALIGN
        self.f.write(b"\0" * pad)
        self.off += pad
        self.f.write(blob)
        self.lines.append("t %s %s %d %d %.9g %d %d %d" % (name, enc, N, K, scale, zp, self.off, len(blob)))
        self.off += len(blob)

    def close(self):
        self.f.close()
        with open(os.path.join(self.path, "manifest.txt"), "w") as f:
            f.write("\n".join(self.lines) + "\n")


def read_manifest(path):
    cfg, ts = {}, []
    for line in open(os.path.join(path, "manifest.txt")):
        p = line.split()
        if p and p[0] == "cfg":
            cfg[p[1]] = p[2]
        elif p and p[0] == "t":
            ts.append(dict(name=p[1], enc=p[2], N=int(p[3]), K=int(p[4]), scale=float(p[5]), zp=int(p[6]),
                           offset=int(p[7]), nbytes=int(p[8])))
    return cfg, ts


# rotary cos / sin rows stored in a bundle
ROPE_ROWS = 4096


def eos_ids(hf, c):
    """End-of-turn ids: config / generation config eos_token_id and the tokenizer's eos token, comma-separated."""
    from transformers import AutoTokenizer, GenerationConfig
    ids = set()
    for src in (c, GenerationConfig.from_pretrained(hf)):
        e = getattr(src, "eos_token_id", None)
        ids |= set(e if isinstance(e, (list, tuple)) else [e] if e is not None else [])
    t = AutoTokenizer.from_pretrained(hf)
    if t.eos_token_id is not None:
        ids.add(t.eos_token_id)
    return ",".join(str(i) for i in sorted(ids))
