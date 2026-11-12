// D3: rANS tile engine for the ECW1 bitstream (docs/bitstream.md), written for Vitis HLS and verified by g++
// C simulation (hw/rans/tb/csim_main.cpp). One engine = 32 lane datapaths, one tile step per cycle (II = 1).
//
// Per step (one cycle at II = 1):
//   1. table select: the step's 32 symbols lie in at most two consecutive rows (two only for ragged tiles whose
//      width is not a multiple of 32); the engine broadcasts the row's frequency table (and the next row's).
//   2. 32 lanes: slot = x & 0xFFF; symbol via LOOKUP; x = f_s * (x >> 12) + slot - cum_s
//   3. renormalization network: need[l] = (x_l < 2^16); 32-wide prefix sum -> word offset; x_l = x_l << 16 | word
//   4. symbols written to the packed output (4- or 8-bit, packed_w4 / packed_w8 layout)
// LOOKUP = 0: 4096-entry slot->symbol table replicated per lane (BRAM/LUTRAM)
// LOOKUP = 1: comparator tree against the broadcast cum values (A-1 comparators, no RAM; A = 16 intended)
#pragma once
#include <cstdint>

namespace ecwhw {

static const int LANES = 32;
static const int PROB_BITS = 12;
static const int M = 1 << PROB_BITS;
static const int MAX_TABLES = 4;
static const int MAX_A = 256;
static const int MAX_TN = 128, MAX_TK = 128;

struct Tables {
  int A, n;
  uint16_t freq[MAX_TABLES][MAX_A];
  uint16_t cum[MAX_TABLES][MAX_A];
  uint8_t lut[MAX_TABLES][M];       // used by LOOKUP = 0 (replicated per lane in hardware)
};

struct TileIn {
  int tn, tk;                        // tile shape (ragged edges allowed)
  uint32_t state[LANES];             // initial decoder states
  const uint16_t* words;             // this tile's word stream (AXI read port in hardware)
  int n_words;
  const uint8_t* row_tab;            // table index per tile row
};

struct Stats {
  long long steps = 0, words = 0, two_row_steps = 0, two_table_steps = 0;
};

template <int A, int LOOKUP>
static inline uint32_t lookup(const Tables& T, int tb, uint32_t slot) {
  if (LOOKUP == 0) return T.lut[tb][slot];
  uint32_t s = 0;
  for (int k = 1; k < A; ++k) {
#pragma HLS UNROLL
    s += (slot >= T.cum[tb][k]) ? 1u : 0u;
  }
  return s;
}

// Decodes one tile into sym[tn][tk] (row-major). Returns false on an integrity error (final state != 2^16,
// or the word stream not fully consumed).
template <int A, int LOOKUP>
bool decode_tile(const Tables& T, const TileIn& in, uint8_t sym[MAX_TN * MAX_TK], Stats& st) {
  uint32_t x[LANES];
#pragma HLS ARRAY_PARTITION variable = x complete
  for (int l = 0; l < LANES; ++l) {
#pragma HLS UNROLL
    x[l] = in.state[l];
  }
  const int n = in.tn * in.tk;
  const int steps = (n + LANES - 1) / LANES;
  int ptr = 0;
step_loop:
  for (int t = 0; t < steps; ++t) {
#pragma HLS PIPELINE II = 1
    // 1. table broadcast: row of lane 0 and the following row
    const int i0 = t * LANES;
    const int row0 = i0 / in.tk;
    const int last = (i0 + LANES - 1 < n ? i0 + LANES - 1 : n - 1);
    const int row1 = last / in.tk;
    const int tb0 = in.row_tab[row0], tb1 = in.row_tab[row1];
    st.two_row_steps += (row1 != row0);
    st.two_table_steps += (tb1 != tb0);
    // 2. lane datapaths
    bool need[LANES];
#pragma HLS ARRAY_PARTITION variable = need complete
    for (int l = 0; l < LANES; ++l) {
#pragma HLS UNROLL
      const int i = i0 + l;
      need[l] = false;
      if (i < n) {
        const int tb = (i / in.tk == row0) ? tb0 : tb1;
        const uint32_t slot = x[l] & (M - 1);
        const uint32_t s = lookup<A, LOOKUP>(T, tb, slot);
        x[l] = (uint32_t)T.freq[tb][s] * (x[l] >> PROB_BITS) + slot - T.cum[tb][s];
        sym[i] = (uint8_t)s;
        need[l] = x[l] < (1u << 16);
      }
    }
    // 3. renormalization network: exclusive prefix sum of need flags -> word offsets (increasing lane order)
    int off[LANES];
#pragma HLS ARRAY_PARTITION variable = off complete
    int acc = 0;
    for (int l = 0; l < LANES; ++l) {
#pragma HLS UNROLL
      off[l] = acc;
      acc += need[l] ? 1 : 0;
    }
    for (int l = 0; l < LANES; ++l) {
#pragma HLS UNROLL
      if (need[l]) x[l] = (x[l] << 16) | in.words[ptr + off[l]];
    }
    ptr += acc;
    st.words += acc;
  }
  st.steps += steps;
  // integrity: every lane back at L = 2^16 and the stream fully consumed
  bool ok = (ptr == in.n_words);
  for (int l = 0; l < LANES; ++l) {
#pragma HLS UNROLL
    ok = ok && (x[l] == (1u << 16));
  }
  return ok;
}

}  // namespace ecwhw
