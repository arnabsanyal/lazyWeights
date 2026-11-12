// ZCU102 jobs (hw/huff/fpga/zcu_core.sv, scripts/zcu_host.py): a WHOLE Huffman-coded matrix of an llmrt bundle cut
// into chunks of consecutive tiles that fit an engine's payload memory (max_words) and descriptor memory (max_tiles).
// Each chunk is a stand-alone slice as in slice_main.cpp (lane starts rebased to its first payload word, the words
// copied plus 2 zero words, the last lane's end = n_words x 32). The C model (comparator lookup, buffer 64, FIFO 4,
// port 8, latency 8, PRE 1) decodes every chunk: its cycles / stalls / words are what the board must reproduce, its
// symbols are checked here against the full-matrix decode, and the board checks its own symbols against per-lane
// CRC-32s (zlib) of the C model's, lane stream order (tile, symbol index).
//   chunks <bundle dir> <matrix> <out.hmat> [max_words=32768] [max_tiles=16]
// File (little endian): "HMAT", u32 version 1, n_chunks, ntt (matrix tiles), enc bits (8 / 4), N, K,
//   u16 lut[2048], u8 srt[256], 11 x (u16 limit, u16 first, u16 base),
//   per chunk: u32 t0, nt, n_words, c_cycles, c_stalls, c_words, nt x 32 x (u32 start, u32 end), u32 words[n_words],
//   u32 crc[32].
// stdout: one JSON line for the matrix.
#include "huff_engine.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

using namespace ecwhw;

static uint32_t crc_tab[256];
static void crc_init() {
  for (uint32_t i = 0; i < 256; ++i) {
    uint32_t c = i;
    for (int k = 0; k < 8; ++k) c = c & 1 ? (c >> 1) ^ 0xEDB88320u : c >> 1;
    crc_tab[i] = c;
  }
}

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: chunks <bundle> <matrix> <out.hmat> [max_words] [max_tiles]\n"); return 2; }
  crc_init();
  std::string dir = argv[1], name = argv[2];
  size_t max_words = argc > 4 ? strtoul(argv[4], nullptr, 10) : 32768;
  int max_tiles = argc > 5 ? atoi(argv[5]) : 16;
  std::ifstream mf(dir + "/manifest.txt");
  std::string enc; int N = 0, K = 0; size_t off = 0, nb = 0; bool found = false;
  for (std::string line; std::getline(mf, line);) {
    std::istringstream ss(line);
    std::string k, n; ss >> k;
    if (k != "t") continue;
    double sc; long zp; std::string e; int n_, k_; size_t o, b;
    ss >> n >> e >> n_ >> k_ >> sc >> zp >> o >> b;
    if (n == name) { enc = e; N = n_; K = k_; off = o; nb = b; found = true; }
  }
  if (!found || (enc != "huf8" && enc != "huf4p")) { fprintf(stderr, "%s: not a Huffman matrix\n", name.c_str()); return 2; }
  int fd = open((dir + "/weights.bin").c_str(), O_RDONLY);
  struct stat sb; fstat(fd, &sb);
  const uint8_t* W = (const uint8_t*)mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  HufTable tb;
  memcpy(tb.lut, W + off, sizeof tb.lut);
  build_canon(tb);
  MatIn m;
  m.N = N; m.bits = enc == "huf8" ? 8 : 4; m.KS = m.bits == 8 ? K : K / 2; m.add = nullptr; m.G = 1;
  const int KT = m.KS / H_TK, NTT = (m.N / H_TN) * KT;
  m.lane_start = (const uint32_t*)(W + off + sizeof tb.lut);
  m.words = m.lane_start + (size_t)NTT * H_LANES;
  m.n_words = (nb - sizeof tb.lut - (size_t)NTT * H_LANES * 4) / 4;
  auto sbit = [&](int t, int l) -> uint64_t {
    if (l == H_LANES) { if (t + 1 >= NTT) return (uint64_t)m.n_words * 32; t += 1; l = 0; }
    return m.lane_start[(size_t)t * H_LANES + l];
  };
  std::vector<uint8_t> full((size_t)m.N * m.KS);
  HStats fs;
  decode_mat<1, 64, 4, 8, 8, 1>(tb, m, full.data(), fs);

  FILE* f = fopen(argv[3], "wb");
  auto u32 = [&](uint32_t v) { fwrite(&v, 4, 1, f); };
  auto u16 = [&](uint16_t v) { fwrite(&v, 2, 1, f); };
  fwrite("HMAT", 1, 4, f);
  u32(1); u32(0); u32(NTT); u32(m.bits); u32(N); u32(K);
  for (int i = 0; i < (1 << H_LMAX); ++i) u16(tb.lut[i] & 0xFFF);
  fwrite(tb.sorted, 1, 256, f);
  for (int l = 1; l <= H_LMAX; ++l) { u16(tb.limit[l] & 0xFFF); u16(tb.first[l]); u16(tb.base[l]); }

  long long cyc = 0, stl = 0, wrd = 0, sym = 0, mm = 0, nwords = 0;
  int nch = 0, t0 = 0;
  while (t0 < NTT) {
    const uint32_t base = (uint32_t)(sbit(t0, 0) >> 5);
    auto words_for = [&](int n) { return (size_t)((sbit(t0 + n - 1, H_LANES) + 31) >> 5) - base + 2; };
    int nt = 0;
    while (t0 + nt < NTT && nt < max_tiles && words_for(nt + 1) <= max_words) ++nt;
    if (nt == 0) { fprintf(stderr, "%s: tile %d does not fit %zu words\n", name.c_str(), t0, max_words); return 3; }
    std::vector<uint32_t> ls((size_t)nt * H_LANES), wd(words_for(nt), 0u);
    for (int t = 0; t < nt; ++t)
      for (int l = 0; l < H_LANES; ++l) ls[(size_t)t * H_LANES + l] = (uint32_t)(sbit(t0 + t, l) - (uint64_t)base * 32);
    const size_t last = (size_t)((sbit(t0 + nt - 1, H_LANES) + 31) >> 5);
    for (size_t i = base; i < last && i < m.n_words; ++i) wd[i - base] = m.words[i];
    MatIn s = m;
    s.N = nt * H_TN; s.KS = H_TK; s.lane_start = ls.data(); s.words = wd.data(); s.n_words = wd.size();
    std::vector<uint8_t> ref((size_t)s.N * s.KS);
    HStats cs;
    decode_mat<1, 64, 4, 8, 8, 1>(tb, s, ref.data(), cs);
    for (int t = 0; t < nt; ++t) {
      int tr = (t0 + t) / KT, tc = (t0 + t) % KT;
      for (int r = 0; r < H_TN; ++r)
        for (int c = 0; c < H_TK; ++c)
          mm += ref[((size_t)t * H_TN + r) * H_TK + c] != full[((size_t)tr * H_TN + r) * m.KS + (size_t)tc * H_TK + c];
    }
    u32(t0); u32(nt); u32((uint32_t)wd.size()); u32((uint32_t)cs.cycles); u32((uint32_t)cs.stalls); u32((uint32_t)cs.words);
    for (int t = 0; t < nt; ++t)
      for (int l = 0; l < H_LANES; ++l) {
        u32(ls[(size_t)t * H_LANES + l]);
        u32(l + 1 < H_LANES ? ls[(size_t)t * H_LANES + l + 1] : (t + 1 < nt ? ls[(size_t)(t + 1) * H_LANES] : (uint32_t)(wd.size() * 32)));
      }
    fwrite(wd.data(), 4, wd.size(), f);
    for (int l = 0; l < H_LANES; ++l) {
      uint32_t c = 0xFFFFFFFFu;
      for (int t = 0; t < nt; ++t)
        for (int i = 0; i < H_SYMS; ++i) {
          int r = l + (i >= H_TK ? 32 : 0), cc = i % H_TK;
          c = crc_tab[(c ^ ref[((size_t)t * H_TN + r) * H_TK + cc]) & 0xFF] ^ (c >> 8);
        }
      u32(c ^ 0xFFFFFFFFu);
    }
    cyc += cs.cycles; stl += cs.stalls; wrd += cs.words; sym += cs.symbols; nwords += wd.size();
    ++nch; t0 += nt;
  }
  fseek(f, 8, SEEK_SET); u32(nch);
  fclose(f);
  if (mm) { fprintf(stderr, "%s: chunk decode != full-matrix decode: %lld\n", name.c_str(), mm); return 4; }
  printf("{\"matrix\": \"%s\", \"enc\": \"%s\", \"N\": %d, \"K\": %d, \"ntt\": %d, \"chunks\": %d, \"payload_words\": %lld, "
         "\"c_cycles\": %lld, \"c_stalls\": %lld, \"c_words\": %lld, \"symbols\": %lld, \"full_cycles\": %lld}\n",
         name.c_str(), enc.c_str(), N, K, NTT, nch, nwords, cyc, stl, wrd, sym, fs.cycles);
  return 0;
}
