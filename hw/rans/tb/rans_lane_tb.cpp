// Verilator testbench of hw/rans/rtl/rans_lane.sv: a C++ rANS encoder (12-bit probabilities, 32-bit state, 16-bit
// renormalization, decode order = the lane's bit stream) writes T tile streams of NSYM symbols back to back (each
// stream: initial state, then the 16-bit words in decode order, LSB-first); the lane decodes them through its request
// side against a memory that answers every grant LAT cycles later. Every decoded symbol is checked against the encoded
// one; cycles and symbols are reported.
//   rans_lane_tb <tiles> <lat> <seed> [entropy-shape s]   -> one JSON line on stdout, exit 1 on any mismatch
#include "Vrans_lane.h"
#include "verilated.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <deque>
#include <random>
#include <vector>

#ifndef CFG_A
#define CFG_A 256
#endif
static const int A = CFG_A, NSYM = A == 256 ? 512 : 1024, PB = 12, M = 1 << PB;
static const uint64_t L = 1u << 16;

int main(int argc, char** argv) {
  const int T = argc > 1 ? atoi(argv[1]) : 8, LAT = argc > 2 ? atoi(argv[2]) : 8, seed = argc > 3 ? atoi(argv[3]) : 1;
  const double shape = argc > 4 ? atof(argv[4]) : 0.35;   // 4-bit code distribution ~ exp(-shape |c - 7.5|)
  std::mt19937 rng(seed);
  // distribution: 4-bit codes peaked in the middle (as IQ4_XS / K-quant codes); A = 256: pairs of independent codes
  std::vector<double> p4(16);
  double z = 0;
  for (int c = 0; c < 16; ++c) z += p4[c] = std::exp(-shape * std::fabs(c - 7.5));
  std::vector<double> p(A);
  for (int s = 0; s < A; ++s) p[s] = A == 16 ? p4[s] / z : (p4[s & 15] / z) * (p4[s >> 4] / z);
  std::vector<uint32_t> freq(A), cum(A);
  int tot = 0;
  for (int s = 0; s < A; ++s) tot += freq[s] = std::max(1, (int)std::lround(p[s] * M));
  while (tot != M) {   // fix the sum on the most probable symbol
    int k = (int)(std::max_element(freq.begin(), freq.end()) - freq.begin());
    int d = tot > M ? -1 : 1;
    freq[k] += d; tot += d;
  }
  for (int s = 1; s < A; ++s) cum[s] = cum[s - 1] + freq[s - 1];
  std::discrete_distribution<int> dist(p.begin(), p.end());
  double H = 0;
  for (int s = 0; s < A; ++s) H -= p[s] * std::log2(p[s]);
  // encode T streams
  std::vector<std::vector<int>> sym(T, std::vector<int>(NSYM));
  std::vector<uint64_t> bits;   // bit positions: stream t starts at start[t]
  std::vector<uint32_t> words;
  std::vector<uint64_t> start(T + 1);
  uint64_t pos = 0;
  auto put = [&](uint32_t v, int n) {
    for (int i = 0; i < n; ++i, ++pos) {
      if ((pos >> 5) >= words.size()) words.push_back(0);
      words[pos >> 5] |= ((v >> i) & 1u) << (pos & 31);
    }
  };
  for (int t = 0; t < T; ++t) {
    for (auto& s : sym[t]) s = dist(rng);
    uint64_t x = L;
    std::vector<uint16_t> out;
    for (int i = NSYM - 1; i >= 0; --i) {
      const uint32_t f = freq[sym[t][i]];
      const uint64_t xmax = ((L >> PB) << 16) * f;
      while (x >= xmax) { out.push_back((uint16_t)(x & 0xFFFF)); x >>= 16; }
      x = ((x / f) << PB) + (x % f) + cum[sym[t][i]];
    }
    start[t] = pos;
    put((uint32_t)x, 32);
    for (size_t i = out.size(); i-- > 0;) put(out[i], 16);
  }
  start[T] = pos;
  words.push_back(0); words.push_back(0);
  // simulate
  Vrans_lane* d = new Vrans_lane;
  {
    uint32_t* cw = (uint32_t*)&d->cum;
    uint32_t* fw = (uint32_t*)&d->freq;
    memset(cw, 0, sizeof d->cum); memset(fw, 0, sizeof d->freq);
    for (int s = 1; s < A; ++s)
      for (int b = 0; b < 12; ++b) if ((cum[s] >> b) & 1) { int bit = (s - 1) * 12 + b; cw[bit >> 5] |= 1u << (bit & 31); }
    for (int s = 0; s < A; ++s)
      for (int b = 0; b < 13; ++b) if ((freq[s] >> b) & 1) { int bit = s * 13 + b; fw[bit >> 5] |= 1u << (bit & 31); }
  }
  d->ntiles = T; d->rst = 1; d->start = 0; d->grant = 0; d->resp_valid = 0;
  auto tick = [&]() { d->clk = 0; d->eval(); d->clk = 1; d->eval(); };
  d->desc_start = 0; d->desc_end = 0;
  tick(); d->rst = 0; d->start = 1; tick(); d->start = 0;
  struct Rsp { long long at; uint32_t data; uint8_t meta; };
  std::deque<Rsp> q;
  long long cyc = 0, nsym = 0, bad = 0, words_req = 0;
  while (d->busy && cyc < 50000000) {
    const int rt = d->rtile;
    d->desc_start = (uint32_t)start[std::min(rt, T)];
    d->desc_end = (uint32_t)start[std::min(rt + 1, T)];
    d->eval();
    d->grant = d->want;
    d->resp_valid = 0;
    if (!q.empty() && q.front().at <= cyc) {
      d->resp_valid = 1; d->resp_data = q.front().data; d->resp_meta = q.front().meta; q.pop_front();
    }
    d->eval();
    if (d->out_valid) {
      ++nsym;
      if ((int)d->out_sym != sym[d->out_tile][d->out_idx]) {
        if (bad < 5) fprintf(stderr, "mismatch tile %d idx %d: %d vs %d\n", d->out_tile, d->out_idx, d->out_sym,
                             sym[d->out_tile][d->out_idx]);
        ++bad;
      }
    }
    if (d->grant) {
      q.push_back({cyc + LAT, d->req_addr < words.size() ? words[d->req_addr] : 0u, (uint8_t)d->req_meta});
      ++words_req;
    }
    tick();
    ++cyc;
  }
  printf("{\"A\": %d, \"tiles\": %d, \"lat\": %d, \"seed\": %d, \"shape\": %.3f, \"entropy_bits\": %.4f, "
         "\"coded_bits_per_symbol\": %.4f, \"symbols\": %lld, \"expected\": %lld, \"mismatch\": %lld, \"cycles\": %lld, "
         "\"sym_per_cycle\": %.4f, \"words\": %lld}\n",
         A, T, LAT, seed, shape, H, (double)start[T] / (T * (double)NSYM), nsym, (long long)T * NSYM, bad, cyc,
         (double)nsym / cyc, words_req);
  delete d;
  return bad || nsym != (long long)T * NSYM ? 1 : 0;
}
