// C simulation of hw/huff/huff_engine.h on every Huffman-coded matrix (huf8 / huf4p) of an llmrt bundle:
// reference decoder (a port of ecw/rt/bundle.py decode_huf8, + the per-row / group re-centering add of
// scripts/llm_bundle.py) vs the engine, for both lookups and a set of memory configurations.
//   g++ -O2 -std=c++17 -pthread -I hw/huff hw/huff/tb/csim_main.cpp -o build/huff_csim
//   build/huff_csim <bundle dir> [--threads N] [--only name] [--dump-ref name out.bin]
// stdout: one JSON line per (matrix, config) and one summary line per config ("kind": "summary").
#include "huff_engine.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
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

struct T { std::string name, enc; int N, K; size_t off, nb; };

struct Cfg { const char* name; int lookup, buf, fifo, port, lat, pre;
             std::function<void(const HufTable&, const MatIn&, uint8_t*, HStats&)> run; };

#define CFG(NM, LK, B, F, P, LT, PR) Cfg{NM, LK, B, F, P, LT, PR, \
    [](const HufTable& t, const MatIn& m, uint8_t* o, HStats& s) { decode_mat<LK, B, F, P, LT, PR>(t, m, o, s); }}
static std::vector<Cfg> cfgs() {   // first = the reference configuration of the sweep
  return {CFG("table", 0, 64, 4, 8, 8, 1),          CFG("comparator", 1, 64, 4, 8, 8, 1),
          CFG("no_prefetch", 0, 64, 4, 8, 8, 0),    CFG("buf48", 0, 48, 4, 8, 8, 1),
          CFG("buf42", 0, 42, 4, 8, 8, 1),          CFG("port4", 0, 64, 4, 4, 8, 1),
          CFG("port6", 0, 64, 4, 6, 8, 1),          CFG("port7", 0, 64, 4, 7, 8, 1),
          CFG("port12", 0, 64, 4, 12, 8, 1),        CFG("port16", 0, 64, 4, 16, 8, 1),
          CFG("fifo2_lat1", 0, 64, 2, 8, 1, 1),     CFG("fifo4_lat32", 0, 64, 4, 8, 32, 1),
          CFG("fifo8_lat32", 0, 64, 8, 8, 32, 1),   CFG("fifo16_lat64", 0, 64, 16, 8, 64, 1)};
}

// reference: all lanes of a tile in lock step, a 64-bit window per symbol (decode_huf8)
static void ref_decode(const HufTable& tb, const MatIn& m, uint8_t* out) {
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

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: huff_csim <bundle> [--threads N] [--only name] [--dump-ref name f]\n"); return 2; }
  std::string dir = argv[1], only, dump_name, dump_file;
  int nth = 8;
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--threads") nth = atoi(argv[++i]);
    else if (a == "--only") only = argv[++i];
    else if (a == "--dump-ref") { dump_name = argv[++i]; dump_file = argv[++i]; }
  }
  std::ifstream mf(dir + "/manifest.txt");
  std::map<std::string, T> ts;
  std::vector<std::string> mats;
  for (std::string line; std::getline(mf, line);) {
    std::istringstream ss(line);
    std::string k; ss >> k;
    if (k != "t") continue;
    T t; double sc; long zp; ss >> t.name >> t.enc >> t.N >> t.K >> sc >> zp >> t.off >> t.nb;
    ts[t.name] = t;
    if ((t.enc == "huf8" || t.enc == "huf4p") && (only.empty() || t.name == only)) mats.push_back(t.name);
  }
  int fd = open((dir + "/weights.bin").c_str(), O_RDONLY);
  struct stat sb; fstat(fd, &sb);
  const uint8_t* W = (const uint8_t*)mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (W == MAP_FAILED) { perror("mmap"); return 1; }
  auto C = cfgs();
  std::vector<HStats> tot(C.size());
  std::vector<long long> mism(C.size(), 0);
  std::mutex mu;
  std::atomic<int> next(0), bad(0);
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
      if (ra != ts.end()) {   // [N, G] or [1, N] (per row)
        const float* f = (const float*)(W + ra->second.off);
        size_t n = (size_t)ra->second.N * ra->second.K;
        add.resize(n);
        for (size_t j = 0; j < n; ++j) add[j] = (int)f[j];
        m.G = (int)(n / m.N); m.add = add.data();
      }
      int maxlen = 0;
      for (int j = 0; j < (1 << H_LMAX); ++j) maxlen = std::max(maxlen, tb.lut[j] >> 8);
      std::vector<uint8_t> ref((size_t)m.N * m.KS), got(ref.size());
      ref_decode(tb, m, ref.data());
      if (!dump_name.empty() && t.name == dump_name) {
        FILE* f = fopen(dump_file.c_str(), "wb"); fwrite(ref.data(), 1, ref.size(), f); fclose(f);
      }
      for (size_t c = 0; c < C.size(); ++c) {
        HStats st;
        std::fill(got.begin(), got.end(), 0xAA);
        C[c].run(tb, m, got.data(), st);
        long long mm = 0;
        for (size_t j = 0; j < ref.size(); ++j) mm += ref[j] != got[j];
        std::lock_guard<std::mutex> g(mu);
        if (mm) ++bad;
        mism[c] += mm;
        HStats& a = tot[c];
        a.cycles += st.cycles; a.symbols += st.symbols; a.stalls += st.stalls; a.words += st.words;
        a.bits += st.bits; a.tiles += st.tiles; a.port_busy += st.port_busy;
        printf("{\"kind\": \"matrix\", \"name\": \"%s\", \"enc\": \"%s\", \"N\": %d, \"K\": %d, \"G\": %d, "
               "\"max_len\": %d, \"config\": \"%s\", \"lookup\": %d, \"buf\": %d, \"fifo\": %d, \"port\": %d, "
               "\"lat\": %d, \"pre\": %d, \"mismatch\": %lld, \"symbols\": %lld, \"cycles\": %lld, \"stalls\": %lld, "
               "\"words\": %lld, \"bits\": %lld, \"sym_per_cycle\": %.4f}\n",
               t.name.c_str(), t.enc.c_str(), t.N, t.K, m.G, maxlen, C[c].name, C[c].lookup, C[c].buf, C[c].fifo,
               C[c].port, C[c].lat, C[c].pre, mm, st.symbols, st.cycles, st.stalls, st.words, st.bits,
               (double)st.symbols / st.cycles);
        fflush(stdout);
      }
    }
  };
  std::vector<std::thread> th;
  for (int i = 0; i < nth; ++i) th.emplace_back(work);
  for (auto& x : th) x.join();
  for (size_t c = 0; c < C.size(); ++c) {
    const HStats& a = tot[c];
    printf("{\"kind\": \"summary\", \"config\": \"%s\", \"lookup\": %d, \"buf\": %d, \"fifo\": %d, \"port\": %d, "
           "\"lat\": %d, \"pre\": %d, \"matrices\": %zu, \"mismatch\": %lld, \"symbols\": %lld, \"cycles\": %lld, "
           "\"stalls\": %lld, \"words\": %lld, \"bits\": %lld, \"sym_per_cycle\": %.4f, \"bits_per_symbol\": %.4f, "
           "\"words_per_cycle\": %.4f}\n",
           C[c].name, C[c].lookup, C[c].buf, C[c].fifo, C[c].port, C[c].lat, C[c].pre, mats.size(), mism[c], a.symbols,
           a.cycles, a.stalls, a.words, a.bits, (double)a.symbols / a.cycles, (double)a.bits / a.symbols,
           (double)a.words / a.cycles);
  }
  fprintf(stderr, "%zu matrices, %d (matrix, config) mismatches\n", mats.size(), bad.load());
  return bad ? 1 : 0;
}
