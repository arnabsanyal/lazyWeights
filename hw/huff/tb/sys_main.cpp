// A4 / A5 system check: N Huffman tile engines (hw/huff/huff_engine.h, comparator or table lookup) sharing one DRAM
// with a bandwidth cap and a long read latency. Engine e decodes tiles e, e + N, ... of each matrix (decode_mat's
// t0 / step); the engines start every matrix together and the matrix ends when the last engine does (a barrier per
// matrix: conservative, a GEMV needs all of its weights). Memory: each cycle the DRAM grants at most `cap` words
// (fractional, carried; unused budget is not banked beyond one cycle), split round-robin over the engines, each
// engine at most PORT words over its own round-robin lane arbiter; a word arrives LAT cycles after its grant.
// The engine loop is a runtime-parameter copy of decode_mat (PRE = 1); --xcheck runs decode_mat<LK, 64, 4, 8, 8, 1>
// on the same matrix and requires equal cycles with N = 1, cap = 0 (no cap), lat 8, fifo 4, port 8, buf 64.
//   g++ -O2 -std=c++17 -pthread -I hw/huff hw/huff/tb/sys_main.cpp -o build/huff_sys
//   build/huff_sys <bundle> [--engines N] [--lat L] [--fifo F] [--port P] [--buf B] [--lookup 0|1] [--cap W]
//                  [--prefix l0.,l15.] [--threads T] [--xcheck]
// stdout: one JSON line per matrix and a summary line; every matrix is checked bit-exact vs the reference decoder.
#include "huff_engine.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace ecwhw;

struct P { int engines = 4, lat = 8, fifo = 4, port = 8, buf = 64, lookup = 1; double cap = 0; };

struct LaneD {
  uint64_t bb = 0; int nb = 0, done = 0, tile = 0, rtile = 0; uint32_t wa = 0, wend = 0;
  std::vector<uint32_t> fifo; std::vector<long long> ready; std::vector<int8_t> skip; std::vector<int> ftile;
  int fh = 0, fn = 0;
};

struct Eng {   // decode_mat<LOOKUP, BUF, FIFO, PORT, LAT, 1> as a steppable object
  const P* p; const HufTable* T; const MatIn* in; uint8_t* out;
  int t0, step, KT, NTT, nt, rr = 0, active;
  LaneD L[H_LANES];
  HStats st;
  long long done_cyc = 0;
  uint64_t sbit(int ti, int l) const {
    int t = t0 + ti * step;
    if (l == H_LANES) { if (t + 1 >= NTT) return (uint64_t)in->n_words * 32; t += 1; l = 0; }
    return in->lane_start[(size_t)t * H_LANES + l];
  }
  void req(LaneD& a, int l, int ti) {
    a.rtile = ti;
    if (ti >= nt) return;
    uint64_t b = sbit(ti, l), e = sbit(ti, l + 1);
    a.wa = (uint32_t)(b >> 5); a.wend = (uint32_t)(e > b ? (e - 1) >> 5 : b >> 5);
  }
  void init(const P& p_, const HufTable& T_, const MatIn& in_, uint8_t* out_, int t0_, int step_) {
    p = &p_; T = &T_; in = &in_; out = out_; t0 = t0_; step = step_;
    KT = in->KS / H_TK; NTT = (in->N / H_TN) * KT; nt = NTT > t0 ? (NTT - t0 + step - 1) / step : 0;
    for (int l = 0; l < H_LANES; ++l) {
      LaneD& a = L[l];
      a.fifo.assign(p->fifo, 0); a.ready.assign(p->fifo, 0); a.skip.assign(p->fifo, 0); a.ftile.assign(p->fifo, 0);
      req(a, l, 0);
    }
    active = nt > 0 ? H_LANES : 0;
  }
  void decode(long long cyc) {   // lane datapaths, as decode_mat steps 1-4
    for (int l = 0; l < H_LANES; ++l) {
      LaneD& a = L[l];
      if (a.tile >= nt) continue;
      bool got = false;
      if (a.nb > 0) {
        uint32_t sym, len;
        if (p->lookup) lookup<1>(*T, (uint32_t)(a.bb & ((1u << H_LMAX) - 1)), sym, len);
        else lookup<0>(*T, (uint32_t)(a.bb & ((1u << H_LMAX) - 1)), sym, len);
        if ((int)len <= a.nb) {
          int ti = t0 + a.tile * step, tr = ti / KT, tc = ti % KT;
          int row = tr * H_TN + l + (a.done >= H_TK ? 32 : 0), col = tc * H_TK + (a.done % H_TK);
          int g = in->add ? (int)((long long)col * in->G / in->KS) : 0;
          uint32_t ad = in->add ? (uint32_t)in->add[(size_t)row * in->G + g] : 0u, q;
          if (in->bits == 8) q = (sym + ad) & 255;
          else q = ((sym + ad) & 15) | ((((sym >> 4) + ad) & 15) << 4);
          out[(size_t)row * in->KS + col] = (uint8_t)q;
          a.bb >>= len; a.nb -= len; st.bits += len; ++st.symbols; got = true;
          if (++a.done == H_SYMS) {
            ++st.tiles; ++a.tile; a.done = 0; a.bb = 0; a.nb = 0;
            while (a.fn > 0 && a.ftile[a.fh] < a.tile) { a.fh = (a.fh + 1) % p->fifo; --a.fn; }
            if (a.tile >= nt && --active == 0) done_cyc = cyc + 1;
            continue;
          }
        }
      }
      if (!got) ++st.stalls;
      if (a.nb <= p->buf - 32 && a.fn > 0 && a.ready[a.fh] <= cyc && a.ftile[a.fh] == a.tile) {
        uint64_t w = a.fifo[a.fh];
        int drop = a.skip[a.fh] > 0 ? a.skip[a.fh] : 0;
        a.bb |= (w >> drop) << a.nb; a.nb += 32 - drop;
        a.fh = (a.fh + 1) % p->fifo; --a.fn;
      }
    }
  }
  int wants() const {   // lanes with FIFO room and a stream left (= words this port could take, up to PORT)
    int n = 0;
    for (int l = 0; l < H_LANES; ++l) n += L[l].rtile < nt && L[l].fn < p->fifo;
    return std::min(n, p->port);
  }
  int grant(long long cyc, int budget) {   // memory port: up to min(PORT, budget) grants, round-robin over lanes
    int g = 0, lim = std::min(p->port, budget);
    for (int k = 0; k < H_LANES && g < lim; ++k) {
      int l = (rr + k) % H_LANES;
      LaneD& a = L[l];
      if (a.rtile >= nt || a.fn >= p->fifo) continue;
      int slot = (a.fh + a.fn) % p->fifo;
      uint64_t b = sbit(a.rtile, l);
      a.fifo[slot] = a.wa < in->n_words ? in->words[a.wa] : 0u;
      a.skip[slot] = a.wa == (uint32_t)(b >> 5) ? (int8_t)(b & 31) : (int8_t)-1;
      a.ftile[slot] = a.rtile;
      a.ready[slot] = cyc + p->lat;
      ++a.fn; ++g; ++st.words;
      if (a.wa == a.wend) req(a, l, a.rtile + 1);
      else ++a.wa;
    }
    if (g) rr = (rr + 1) % H_LANES;
    st.port_busy += g;
    return g;
  }
};

// all engines on one matrix; returns cycles until the last engine is done
static long long run_mat(const P& p, const HufTable& T, const MatIn& m, uint8_t* out, HStats& tot, long long& words,
                         long long& capped) {
  std::vector<Eng> E(p.engines);
  for (int e = 0; e < p.engines; ++e) E[e].init(p, T, m, out, e, p.engines);
  double credit = 0;
  int erot = 0;
  long long cyc = 0;
  auto busy = [&]() { for (auto& e : E) if (e.active) return true; return false; };
  while (busy()) {
    for (auto& e : E) if (e.active) e.decode(cyc);
    int budget = 1 << 30;
    if (p.cap > 0) { credit = std::min(credit + p.cap, p.cap); budget = (int)std::floor(credit + 1e-9); }
    int want = 0;
    for (auto& e : E) if (e.active) want += e.wants();
    if (want > budget) ++capped;
    for (int k = 0; k < p.engines; ++k) {
      Eng& e = E[(erot + k) % p.engines];
      if (!e.active || budget <= 0) continue;
      int g = e.grant(cyc, budget);
      budget -= g; words += g;
      if (p.cap > 0) credit -= g;
    }
    erot = (erot + 1) % p.engines;
    ++cyc;
  }
  for (auto& e : E) {
    tot.symbols += e.st.symbols; tot.stalls += e.st.stalls; tot.bits += e.st.bits; tot.tiles += e.st.tiles;
    tot.words += e.st.words;
  }
  return cyc;
}

static void ref_decode(const HufTable& tb, const MatIn& m, uint8_t* out) {   // as csim_main.cpp
  int KT = m.KS / H_TK, NT_ = m.N / H_TN;
  for (int t = 0; t < NT_ * KT; ++t)
    for (int l = 0; l < H_LANES; ++l) {
      uint64_t bp = m.lane_start[(size_t)t * H_LANES + l];
      for (int i = 0; i < H_SYMS; ++i) {
        size_t wi = bp >> 5;
        uint64_t w = (uint64_t)m.words[wi] | ((uint64_t)m.words[wi + 1] << 32);
        uint16_t e = tb.lut[(w >> (bp & 31)) & ((1u << H_LMAX) - 1)];
        uint32_t r = e & 255;
        bp += e >> 8;
        int row = (t / KT) * H_TN + l + (i >= H_TK ? 32 : 0), col = (t % KT) * H_TK + i % H_TK;
        uint32_t ad = m.add ? (uint32_t)m.add[(size_t)row * m.G + (long long)col * m.G / m.KS] : 0u;
        out[(size_t)row * m.KS + col] = m.bits == 8 ? (uint8_t)((r + ad) & 255)
                                                    : (uint8_t)(((r + ad) & 15) | ((((r >> 4) + ad) & 15) << 4));
      }
    }
}

struct T { std::string name, enc; int N, K; size_t off, nb; };

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: huff_sys <bundle> [options]\n"); return 2; }
  std::string dir = argv[1];
  std::vector<std::string> prefixes;
  P p;
  int nth = 8;
  bool xcheck = false;
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--engines") p.engines = atoi(argv[++i]);
    else if (a == "--lat") p.lat = atoi(argv[++i]);
    else if (a == "--fifo") p.fifo = atoi(argv[++i]);
    else if (a == "--port") p.port = atoi(argv[++i]);
    else if (a == "--buf") p.buf = atoi(argv[++i]);
    else if (a == "--lookup") p.lookup = atoi(argv[++i]);
    else if (a == "--cap") p.cap = atof(argv[++i]);
    else if (a == "--threads") nth = atoi(argv[++i]);
    else if (a == "--xcheck") xcheck = true;
    else if (a == "--prefix") {
      std::stringstream ss(argv[++i]);
      for (std::string s; std::getline(ss, s, ',');) prefixes.push_back(s);
    } else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
  }
  if (p.buf < 32 + H_LMAX - 1 || p.buf > 64 || p.fifo < 1 || p.lat < 1) { fprintf(stderr, "bad params\n"); return 2; }
  std::ifstream mf(dir + "/manifest.txt");
  std::map<std::string, T> ts;
  std::vector<std::string> mats;
  for (std::string line; std::getline(mf, line);) {
    std::istringstream ss(line);
    std::string k; ss >> k;
    if (k != "t") continue;
    T t; double sc; long zp; ss >> t.name >> t.enc >> t.N >> t.K >> sc >> zp >> t.off >> t.nb;
    ts[t.name] = t;
    bool sel = prefixes.empty();
    for (auto& s : prefixes) sel |= t.name.compare(0, s.size(), s) == 0;
    if ((t.enc == "huf8" || t.enc == "huf4p") && sel) mats.push_back(t.name);
  }
  int fd = open((dir + "/weights.bin").c_str(), O_RDONLY);
  struct stat sb; fstat(fd, &sb);
  const uint8_t* W = (const uint8_t*)mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (W == MAP_FAILED) { perror("mmap"); return 1; }
  std::mutex mu;
  std::atomic<int> next(0), bad(0), xbad(0);
  HStats tot;
  long long tcyc = 0, twords = 0, tcapped = 0, tweights = 0;
  auto work = [&]() {
    for (int i; (i = next++) < (int)mats.size();) {
      const T& t = ts[mats[i]];
      MatIn m;
      m.N = t.N; m.bits = t.enc == "huf8" ? 8 : 4; m.KS = m.bits == 8 ? t.K : t.K / 2;
      int NTl = (m.N / H_TN) * (m.KS / H_TK);
      HufTable tb;
      memcpy(tb.lut, W + t.off, sizeof tb.lut);
      build_canon(tb);
      m.lane_start = (const uint32_t*)(W + t.off + sizeof tb.lut);
      m.words = m.lane_start + (size_t)NTl * H_LANES;
      m.n_words = (t.nb - sizeof tb.lut - (size_t)NTl * H_LANES * 4) / 4;
      std::vector<int> add;
      m.add = nullptr; m.G = 1;
      auto ra = ts.find(t.name + ".radd");
      if (ra != ts.end()) {
        const float* f = (const float*)(W + ra->second.off);
        size_t n = (size_t)ra->second.N * ra->second.K;
        add.resize(n);
        for (size_t j = 0; j < n; ++j) add[j] = (int)f[j];
        m.G = (int)(n / m.N); m.add = add.data();
      }
      std::vector<uint8_t> ref((size_t)m.N * m.KS), got(ref.size(), 0xAA);
      ref_decode(tb, m, ref.data());
      HStats st;
      long long words = 0, capped = 0;
      long long cyc = run_mat(p, tb, m, got.data(), st, words, capped);
      long long mm = 0;
      for (size_t j = 0; j < ref.size(); ++j) mm += ref[j] != got[j];
      long long xc = -1;
      if (xcheck) {
        HStats s2;
        std::vector<uint8_t> o2(ref.size());
        if (p.lookup) decode_mat<1, 64, 4, 8, 8, 1>(tb, m, o2.data(), s2);
        else decode_mat<0, 64, 4, 8, 8, 1>(tb, m, o2.data(), s2);
        xc = s2.cycles;
        if (xc != cyc) ++xbad;
      }
      long long wts = st.symbols * (m.bits == 4 ? 2 : 1);
      std::lock_guard<std::mutex> g(mu);
      if (mm) ++bad;
      tot.symbols += st.symbols; tot.stalls += st.stalls; tot.bits += st.bits; tot.words += st.words;
      tcyc += cyc; twords += words; tcapped += capped; tweights += wts;
      printf("{\"kind\": \"matrix\", \"name\": \"%s\", \"enc\": \"%s\", \"N\": %d, \"K\": %d, \"mismatch\": %lld, "
             "\"symbols\": %lld, \"weights\": %lld, \"cycles\": %lld, \"words\": %lld, \"capped_cycles\": %lld, "
             "\"bits\": %lld, \"weights_per_cycle\": %.4f, \"xcheck_cycles\": %lld}\n",
             t.name.c_str(), t.enc.c_str(), t.N, t.K, mm, st.symbols, wts, cyc, words, capped, st.bits,
             (double)wts / cyc, xc);
      fflush(stdout);
    }
  };
  std::vector<std::thread> th;
  for (int i = 0; i < nth; ++i) th.emplace_back(work);
  for (auto& x : th) x.join();
  printf("{\"kind\": \"summary\", \"engines\": %d, \"lat\": %d, \"fifo\": %d, \"port\": %d, \"buf\": %d, "
         "\"lookup\": %d, \"cap\": %.4f, \"matrices\": %zu, \"mismatch_matrices\": %d, \"xcheck\": %d, "
         "\"xcheck_bad\": %d, \"symbols\": %lld, \"weights\": %lld, \"cycles\": %lld, \"words\": %lld, "
         "\"capped_cycles\": %lld, \"bits\": %lld, \"stalls\": %lld, \"weights_per_cycle\": %.4f, "
         "\"words_per_cycle\": %.4f, \"bits_per_weight\": %.4f}\n",
         p.engines, p.lat, p.fifo, p.port, p.buf, p.lookup, p.cap, mats.size(), bad.load(), (int)xcheck, xbad.load(),
         tot.symbols, tweights, tcyc, twords, tcapped, tot.bits, tot.stalls, (double)tweights / tcyc,
         (double)twords / tcyc, (double)tot.bits / tweights);
  fprintf(stderr, "%zu matrices, %d mismatched, %d xcheck cycle differences\n", mats.size(), bad.load(), xbad.load());
  return bad || xbad ? 1 : 0;
}
