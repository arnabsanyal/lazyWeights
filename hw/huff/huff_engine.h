// A4: Huffman tile engine for the llmrt bundle bitstreams huf8 / huf4p (ecw/rt/bundle.py, scripts/llm_bundle.py),
// written for Vitis HLS and verified by g++ C simulation (hw/huff/tb/csim_main.cpp) against the reference decoder.
// One engine = 32 lane datapaths; lane l of a 64 x 256-symbol tile decodes tile rows l and l + 32 (512 symbols) from
// its own bitstream (absolute start bit lane_start[t][l]). Target: one symbol per lane per cycle, i.e. 32 weights per
// cycle for huf8 and 64 for huf4p (a symbol = two 4-bit codes).
//
// Lane datapath, per cycle:
//   1. lookup on the low 11 bits of the bit buffer (codes stored LSB-first, max length 11) -> symbol, length;
//      the symbol is accepted when length <= valid bits (a short code needs no full 11-bit window)
//   2. shifter: buffer >>= length
//   3. refill: one 32-bit word from the lane's prefetch FIFO when the buffer has room (valid <= BUF - 32)
//   4. re-centering: q = (r + add) mod 2^b per weight (add per row / group; 0 for the K-quant bundles)
// LOOKUP = 0: the bundle's 2048-entry table (sym | len << 8), replicated per lane (RAM)
// LOOKUP = 1: canonical first-code comparators: limit[l] = (first[l] + count[l]) << (11 - l) against the bit-reversed
//             window, length = 1 + #{l < 11 : window >= limit[l]}, symbol = sorted[base[l] + (window >> (11 - l)) -
//             first[l]] (10 comparators + a 256-entry symbol RAM per lane)
// Memory model: the engine's read port delivers PORT words per cycle, round-robin over the lanes whose FIFO (FIFO
// words, in-flight included) has room, each arriving LAT cycles after the grant. A lane that finishes its 512
// symbols moves to the next tile at once (its FIFO is flushed: the next stream starts elsewhere); the engine holds
// a matrix's tiles in order and is done when every lane is. Stalls = lane-cycles without a symbol.
#pragma once
#include <cstdint>
#include <cstring>

namespace ecwhw {

static const int H_LANES = 32, H_TN = 64, H_TK = 256, H_LMAX = 11, H_SYMS = 2 * H_TK;
static const int H_MAX_FIFO = 16, H_MAX_LAT = 64;

struct HufTable {
  uint16_t lut[1 << H_LMAX];        // LOOKUP = 0
  uint16_t limit[H_LMAX + 1];       // LOOKUP = 1: left-justified (first + count) << (11 - l), l = 1..11
  uint16_t first[H_LMAX + 1], base[H_LMAX + 1];
  uint8_t sorted[256];
};

// Canonical comparator tables from the bundle LUT (lengths per symbol, then codes as ecw/codec/lanecodecs.py
// canonical_codes: increasing code within a length, symbols ascending).
static inline void build_canon(HufTable& T) {
  int len[256] = {0};
  for (int i = 0; i < (1 << H_LMAX); ++i) len[T.lut[i] & 255] = T.lut[i] >> 8;
  int cnt[H_LMAX + 2] = {0};
  for (int s = 0; s < 256; ++s) if (len[s]) ++cnt[len[s]];
  int code = 0, b = 0;
  for (int l = 1; l <= H_LMAX; ++l) {
    T.first[l] = code; T.base[l] = b;
    for (int s = 0; s < 256; ++s) if (len[s] == l) T.sorted[b++] = s;
    code += cnt[l];
    T.limit[l] = code << (H_LMAX - l);
    code <<= 1;
  }
}

static inline uint32_t rev11(uint32_t v) {
  uint32_t r = 0;
  for (int i = 0; i < H_LMAX; ++i) {
#pragma HLS UNROLL
    r |= ((v >> i) & 1u) << (H_LMAX - 1 - i);
  }
  return r;
}

template <int LOOKUP>
static inline void lookup(const HufTable& T, uint32_t win, uint32_t& sym, uint32_t& len) {
  if (LOOKUP == 0) {
    uint16_t e = T.lut[win];
    sym = e & 255; len = e >> 8;
    return;
  }
  uint32_t v = rev11(win), l = 1;
  for (int k = 1; k < H_LMAX; ++k) {
#pragma HLS UNROLL
    l += (v >= T.limit[k]) ? 1u : 0u;
  }
  sym = T.sorted[T.base[l] + (v >> (H_LMAX - l)) - T.first[l]];
  len = l;
}

struct MatIn {
  int N, KS;                        // rows, symbols per row (huf4p: K / 2)
  int bits;                         // 8 (huf8) or 4 (huf4p)
  const uint32_t* lane_start;       // [T][32]
  const uint32_t* words;            // payload (AXI read port in hardware)
  size_t n_words;
  const int* add;                   // [N][G] re-centering add (nullptr: 0)
  int G;
};

struct HStats {
  long long cycles = 0, symbols = 0, stalls = 0, words = 0, bits = 0, tiles = 0, port_busy = 0;
};

struct Lane {
  uint64_t bb; int nb;              // bit buffer, valid bits
  int done, tile;                   // symbols of the current tile; current tile (index into the engine's tiles)
  int rtile; uint32_t wa, wend;     // request side: tile, next word, last word of that tile's stream
  uint32_t fifo[H_MAX_FIFO]; long long ready[H_MAX_FIFO]; int8_t skip[H_MAX_FIFO]; int16_t ftile[H_MAX_FIFO];
  int fh, fn;                       // FIFO incl. in-flight; skip >= 0: first word of a stream (bits to drop)
};

// Decodes the tiles t0, t0 + step, ... of one matrix into out[N][KS] (re-centered symbols, as the reference decoder
// + add); cycles etc. added to st.
// PRE = 0: a lane requests words past its stream end until it finishes the tile, then flushes its FIFO and restarts
//          at the next tile's stream (a refill bubble of >= LAT cycles per tile, wasted fetches)
// PRE = 1: a lane's stream ends where the next lane's begins (lane 31: lane 0 of the next tile; last tile: the
//          payload end), so after its last word it goes on requesting the next tile's stream (FIFO words are
//          tagged with their tile): no bubble, no wasted fetch
template <int LOOKUP, int BUF, int FIFO, int PORT, int LAT, int PRE>
void decode_mat(const HufTable& T, const MatIn& in, uint8_t* out, HStats& st, int t0 = 0, int step = 1) {
  static_assert(BUF >= 32 + H_LMAX - 1 && BUF <= 64, "bit buffer 42..64 bits (a refill must fit before a lane is stuck)");
  static_assert(FIFO >= 1 && FIFO <= H_MAX_FIFO && LAT >= 1 && LAT <= H_MAX_LAT, "fifo / latency");
  const int KT = in.KS / H_TK, NT_ = in.N / H_TN, NTT = NT_ * KT, nt = (NTT - t0 + step - 1) / step;
  Lane L[H_LANES];
#pragma HLS ARRAY_PARTITION variable = L complete
  auto sbit = [&](int ti, int l) -> uint64_t {   // start bit of lane l of engine tile ti (lane 32 = next tile's 0)
    int t = t0 + ti * step;
    if (l == H_LANES) { if (t + 1 >= NTT) return (uint64_t)in.n_words * 32; t += 1; l = 0; }
    return in.lane_start[(size_t)t * H_LANES + l];
  };
  auto req = [&](Lane& a, int l, int ti) {
    a.rtile = ti;
    if (ti >= nt) return;
    uint64_t b = sbit(ti, l), e = PRE ? sbit(ti, l + 1) : (uint64_t)in.n_words * 32;
    a.wa = (uint32_t)(b >> 5); a.wend = (uint32_t)(e > b ? (e - 1) >> 5 : b >> 5);
  };
  for (int l = 0; l < H_LANES; ++l) {
    Lane& a = L[l];
    a.bb = 0; a.nb = 0; a.done = 0; a.tile = 0; a.fh = 0; a.fn = 0;
    req(a, l, 0);
  }
  int rr = 0, active = nt > 0 ? H_LANES : 0;
  long long cyc = 0;
  while (active) {
    // 1-4: lane datapaths (decode on the state at the start of the cycle)
    for (int l = 0; l < H_LANES; ++l) {
#pragma HLS UNROLL
      Lane& a = L[l];
      if (a.tile >= nt) continue;
      bool got = false;
      if (a.nb > 0) {
        uint32_t sym, len;
        lookup<LOOKUP>(T, (uint32_t)(a.bb & ((1u << H_LMAX) - 1)), sym, len);
        if ((int)len <= a.nb) {
          int ti = t0 + a.tile * step, tr = ti / KT, tc = ti % KT;
          int row = tr * H_TN + l + (a.done >= H_TK ? 32 : 0), col = tc * H_TK + (a.done % H_TK);
          int g = in.add ? (int)((long long)col * in.G / in.KS) : 0;
          uint32_t ad = in.add ? (uint32_t)in.add[(size_t)row * in.G + g] : 0u, q;
          if (in.bits == 8) q = (sym + ad) & 255;
          else q = ((sym + ad) & 15) | ((((sym >> 4) + ad) & 15) << 4);
          out[(size_t)row * in.KS + col] = (uint8_t)q;
          a.bb >>= len; a.nb -= len; st.bits += len; ++st.symbols; got = true;
          if (++a.done == H_SYMS) {   // next tile: drop the rest of this stream
            ++st.tiles; ++a.tile; a.done = 0; a.bb = 0; a.nb = 0;
            if (!PRE) { a.fh = 0; a.fn = 0; req(a, l, a.tile); }
            else while (a.fn > 0 && a.ftile[a.fh] < a.tile) { a.fh = (a.fh + 1) % FIFO; --a.fn; }
            if (a.tile >= nt) --active;
            continue;
          }
        }
      }
      if (!got) ++st.stalls;
      if (a.nb <= BUF - 32 && a.fn > 0 && a.ready[a.fh] <= cyc && a.ftile[a.fh] == a.tile) {   // refill
        uint64_t w = a.fifo[a.fh];
        int drop = a.skip[a.fh] > 0 ? a.skip[a.fh] : 0;
        a.bb |= (w >> drop) << a.nb; a.nb += 32 - drop;
        a.fh = (a.fh + 1) % FIFO; --a.fn;
      }
    }
    // memory port: PORT word grants, round-robin over lanes with FIFO room
    int g = 0;
    for (int k = 0; k < H_LANES && g < PORT; ++k) {
      int l = (rr + k) % H_LANES;
      Lane& a = L[l];
      if (a.rtile >= nt || a.fn >= FIFO) continue;
      int slot = (a.fh + a.fn) % FIFO;
      uint64_t b = sbit(a.rtile, l);
      a.fifo[slot] = a.wa < in.n_words ? in.words[a.wa] : 0u;
      a.skip[slot] = a.wa == (uint32_t)(b >> 5) ? (int8_t)(b & 31) : (int8_t)-1;
      a.ftile[slot] = (int16_t)a.rtile;
      a.ready[slot] = cyc + LAT;
      ++a.fn; ++g; ++st.words;
      if (PRE && a.wa == a.wend) req(a, l, a.rtile + 1);
      else ++a.wa;
    }
    if (g) rr = (rr + 1) % H_LANES;
    st.port_busy += g;
    ++cyc;
  }
  st.cycles += cyc;
}

}  // namespace ecwhw
