// Verilator co-simulation of hw/huff/rtl/huff_engine.sv against the C model (hw/huff/huff_engine.h, PRE = 1) and
// the reference decoder, on chosen Huffman-coded matrices of llmrt bundles. Memory: every granted word comes back
// LAT cycles after its grant (as the C model's ready time). Checks: decoded symbols == reference, cycles and stall
// count == C model. --vcd F: waveform of the first matrix (switching activity for power).
// --stim F N: lane 0's inputs, one hex line per clock (table load, start, then N decode cycles) of the first matrix,
// and F.vh with its comparator registers, for the gate-level power simulation (hw/huff/syn/power.sh).
//   cosim <LAT> [--vcd F] [--stim F N] <bundle>:<matrix> ...
// stdout: one JSON line per matrix.
#include "Vhuff_engine.h"
#include "huff_engine.h"
#include "verilated.h"
#include "verilated_vcd_c.h"

#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#ifndef CFG_BUF
#define CFG_BUF 64
#endif
#ifndef CFG_FIFO
#define CFG_FIFO 4
#endif
#ifndef CFG_PORT
#define CFG_PORT 8
#endif
#ifndef CFG_LOOKUP
#define CFG_LOOKUP 1
#endif

using namespace ecwhw;

struct T { std::string enc; int N, K; size_t off, nb; };

struct Bundle {
  std::map<std::string, T> ts;
  const uint8_t* W = nullptr;
  explicit Bundle(const std::string& dir) {
    std::ifstream mf(dir + "/manifest.txt");
    for (std::string line; std::getline(mf, line);) {
      std::istringstream ss(line);
      std::string k, name; ss >> k;
      if (k != "t") continue;
      T t; double sc; long zp; ss >> name >> t.enc >> t.N >> t.K >> sc >> zp >> t.off >> t.nb;
      ts[name] = t;
    }
    int fd = open((dir + "/weights.bin").c_str(), O_RDONLY);
    struct stat sb; fstat(fd, &sb);
    W = (const uint8_t*)mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  }
};

struct Resp { long long due; int lane; uint32_t data; uint8_t meta; };

// packed ports: <= 64 bits are integers, wider ones VlWide (32-bit words); element i of width w at bit i * w
template <class V> static uint32_t getb(const V& v, int off, int w) {
  if constexpr (std::is_integral<V>::value) return (uint32_t)((uint64_t)v >> off) & (uint32_t)((1ull << w) - 1);
  else {
    uint64_t x = v[off >> 5];
    if ((off & 31) + w > 32) x |= (uint64_t)v[(off >> 5) + 1] << 32;
    return (uint32_t)(x >> (off & 31)) & (uint32_t)((1ull << w) - 1);
  }
}
template <class V> static void setb(V& v, int off, int w, uint32_t val) {
  uint64_t m = ((1ull << w) - 1);
  if constexpr (std::is_integral<V>::value) v = (V)(((uint64_t)v & ~(m << off)) | (((uint64_t)val & m) << off));
  else {
    for (int b = 0; b < w; ++b) {   // bit by bit (wide fields are 32-bit aligned in practice)
      int o = off + b;
      v[o >> 5] = (v[o >> 5] & ~(1u << (o & 31))) | (((val >> b) & 1u) << (o & 31));
    }
  }
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  int LAT = atoi(argv[1]);
  std::string vcd, stim;
  long stim_n = 0;
  std::vector<std::string> jobs;
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--vcd") vcd = argv[++i];
    else if (a == "--stim") { stim = argv[++i]; stim_n = atol(argv[++i]); }
    else if (a[0] != '+') jobs.push_back(a);
  }
  std::map<std::string, Bundle*> bundles;
  int bad = 0;
  for (size_t j = 0; j < jobs.size(); ++j) {
    std::string bdir = jobs[j].substr(0, jobs[j].rfind(':')), name = jobs[j].substr(jobs[j].rfind(':') + 1);
    if (!bundles.count(bdir)) bundles[bdir] = new Bundle(bdir);
    Bundle& B = *bundles[bdir];
    const T& t = B.ts.at(name);
    MatIn m;
    m.N = t.N; m.bits = t.enc == "huf8" ? 8 : 4; m.KS = m.bits == 8 ? t.K : t.K / 2;
    m.add = nullptr; m.G = 1;
    const int KT = m.KS / H_TK, NTT = (m.N / H_TN) * KT;
    HufTable tb;
    memcpy(tb.lut, B.W + t.off, sizeof tb.lut);
    build_canon(tb);
    m.lane_start = (const uint32_t*)(B.W + t.off + sizeof tb.lut);
    m.words = m.lane_start + (size_t)NTT * H_LANES;
    m.n_words = (t.nb - sizeof tb.lut - (size_t)NTT * H_LANES * 4) / 4;
    int maxlen = 0;
    for (int i = 0; i < (1 << H_LMAX); ++i) maxlen = std::max(maxlen, tb.lut[i] >> 8);
    // C model and reference (symbols without the re-centering add: the RTL lane outputs r)
    std::vector<uint8_t> ref((size_t)m.N * m.KS), got(ref.size(), 0xAA);
    HStats cs;
    switch (LAT) {
      case 1: decode_mat<CFG_LOOKUP, CFG_BUF, CFG_FIFO, CFG_PORT, 1, 1>(tb, m, ref.data(), cs); break;
      case 8: decode_mat<CFG_LOOKUP, CFG_BUF, CFG_FIFO, CFG_PORT, 8, 1>(tb, m, ref.data(), cs); break;
      case 32: decode_mat<CFG_LOOKUP, CFG_BUF, CFG_FIFO, CFG_PORT, 32, 1>(tb, m, ref.data(), cs); break;
      default: fprintf(stderr, "LAT must be 1, 8 or 32\n"); return 2;
    }
    // RTL
    Vhuff_engine* top = new Vhuff_engine;
    VerilatedVcdC* tr = nullptr;
    if (!vcd.empty() && j == 0) { Verilated::traceEverOn(true); tr = new VerilatedVcdC; top->trace(tr, 99); tr->open(vcd.c_str()); }
    uint64_t tick = 0;
    FILE* sf = (!stim.empty() && j == 0) ? fopen(stim.c_str(), "w") : nullptr;
    long sl = 0;
    // lane 0 inputs: start grant resp_valid resp_meta resp_data desc_start desc_end srt_we srt_addr srt_wdata lut_we
    // lut_addr lut_wdata (148 bits, MSB first)
    auto emit = [&]() {
      if (!sf || sl >= stim_n + 2100) return;
      int gr = 0;
      for (int p = 0; p < CFG_PORT; ++p) gr |= getb(top->req_valid, p, 1) && getb(top->req_lane, 5 * p, 5) == 0;
      fprintf(sf, "%x%x%x%02x%08x%08x%08x%x%02x%02x%x%03x%03x\n", (unsigned)top->start, gr,
              (unsigned)(top->resp_valid & 1u), getb(top->resp_meta, 0, 8), (unsigned)top->resp_data[0],
              (unsigned)top->desc_start[0], (unsigned)top->desc_end[0], (unsigned)top->srt_we, (unsigned)top->srt_addr,
              (unsigned)top->srt_wdata, (unsigned)top->lut_we, (unsigned)top->lut_addr, (unsigned)top->lut_wdata);
      ++sl;
    };
    auto clock = [&]() {
      top->clk = 1; top->eval(); if (tr) tr->dump(tick++);
      top->clk = 0; top->eval(); if (tr) tr->dump(tick++);
    };
    top->rst = 1; top->clk = 0; top->eval(); clock(); top->rst = 0;
    for (int i = 0; i < (1 << H_LMAX); ++i) {   // table load (not timed)
      top->lut_we = 1; top->lut_addr = i; top->lut_wdata = tb.lut[i] & 0xFFF;
      top->srt_we = i < 256; top->srt_addr = i & 255; top->srt_wdata = tb.sorted[i & 255];
      top->cmp_we = i >= 1 && i <= H_LMAX; top->cmp_l = i & 15;
      if (i >= 1 && i <= H_LMAX) { top->cmp_limit = tb.limit[i] & 0xFFF; top->cmp_first = tb.first[i]; top->cmp_base = tb.base[i]; }
      top->eval(); emit();
      clock();
    }
    top->lut_we = 0; top->srt_we = 0; top->cmp_we = 0;
    top->ntiles = NTT; top->start = 1; top->eval(); emit(); clock(); top->start = 0;
    if (sf) {
      FILE* hf = fopen((stim + ".vh").c_str(), "w");
      fprintf(hf, "localparam [15:0] NTILES = 16'd%d;\nlocalparam [%d:0] LIMIT = {", NTT, 12 * (H_LMAX - 1) - 1);
      for (int k = H_LMAX - 1; k >= 1; --k) fprintf(hf, "12'd%d%s", tb.limit[k] & 0xFFF, k > 1 ? ", " : "};\n");
      fprintf(hf, "localparam [%d:0] OFF = {", 8 * H_LMAX - 1);
      for (int k = H_LMAX; k >= 1; --k) fprintf(hf, "8'd%d%s", (tb.base[k] - tb.first[k]) & 255, k > 1 ? ", " : "};\n");
      fclose(hf);
    }
    auto sbit = [&](int ti, int l) -> uint32_t {
      if (l == H_LANES) { if (ti + 1 >= NTT) return (uint32_t)(m.n_words * 32); ti += 1; l = 0; }
      return m.lane_start[(size_t)ti * H_LANES + l];
    };
    std::deque<Resp> q;
    long long cyc = 0, stalls = 0, words = 0;
    while (true) {
      for (int l = 0; l < H_LANES; ++l) {
        int rt = getb(top->rtile, 16 * l, 16);
        top->desc_start[l] = rt < NTT ? sbit(rt, l) : 0;
        top->desc_end[l] = rt < NTT ? sbit(rt, l + 1) : 0;
      }
      top->resp_valid = 0;
      top->eval();
      if (!top->busy) break;
      for (int p = 0; p < CFG_PORT; ++p)
        if (getb(top->req_valid, p, 1)) {
          uint32_t a = getb(top->req_addr, 32 * p, 32);
          q.push_back({cyc + LAT - 1, (int)getb(top->req_lane, 5 * p, 5), a < m.n_words ? m.words[a] : 0u,
                       (uint8_t)getb(top->req_meta, 8 * p, 8)});
          ++words;
        }
      while (!q.empty() && q.front().due <= cyc) {   // in order: fixed latency
        Resp r = q.front(); q.pop_front();
        top->resp_valid |= 1u << r.lane; top->resp_data[r.lane] = r.data; setb(top->resp_meta, 8 * r.lane, 8, r.meta);
      }
      top->eval();
      stalls += top->n_stall;
      for (int l = 0; l < H_LANES; ++l)
        if ((top->out_valid >> l) & 1u) {
          int ti = getb(top->out_tile, 16 * l, 16), i = getb(top->out_idx, 9 * l, 9);
          int row = (ti / KT) * H_TN + l + (i >= H_TK ? 32 : 0), col = (ti % KT) * H_TK + i % H_TK;
          got[(size_t)row * m.KS + col] = (uint8_t)getb(top->out_sym, 8 * l, 8);
        }
      emit();
      clock();
      ++cyc;
      if (cyc > 4 * cs.cycles + 1000) { fprintf(stderr, "%s: timeout\n", name.c_str()); break; }
    }
    long long mm = 0;
    for (size_t i = 0; i < ref.size(); ++i) mm += ref[i] != got[i];
    bool ok = mm == 0 && cyc == cs.cycles && stalls == cs.stalls && words == cs.words;
    bad += !ok;
    printf("{\"bundle\": \"%s\", \"name\": \"%s\", \"enc\": \"%s\", \"N\": %d, \"K\": %d, \"max_len\": %d, "
           "\"lookup\": %d, \"buf\": %d, \"fifo\": %d, \"port\": %d, \"lat\": %d, \"mismatch\": %lld, "
           "\"rtl_cycles\": %lld, \"c_cycles\": %lld, \"rtl_stalls\": %lld, \"c_stalls\": %lld, \"rtl_words\": %lld, "
           "\"c_words\": %lld, \"ok\": %d}\n",
           bdir.c_str(), name.c_str(), t.enc.c_str(), t.N, t.K, maxlen, CFG_LOOKUP, CFG_BUF, CFG_FIFO, CFG_PORT, LAT,
           mm, cyc, cs.cycles, stalls, cs.stalls, words, cs.words, (int)ok);
    fflush(stdout);
    if (tr) { tr->close(); delete tr; }
    if (sf) fclose(sf);
    top->final();
    delete top;
  }
  return bad ? 1 : 0;
}
