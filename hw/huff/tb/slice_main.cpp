// FPGA bring-up jobs (hw/huff/fpga, scripts/fpga_host.py): a slice of consecutive tiles of a Huffman-coded matrix
// of an llmrt bundle, as a stand-alone stream the board decodes from block RAM. The slice is a matrix of its own
// (one tile per row block, KS = 256): lane starts rebased to the slice's first payload word, the words copied (plus
// 2 zero words, as the bundle format), the last lane's end = n_words x 32 (as the C model for a matrix's last tile).
// The C model (comparator lookup, buffer 64, FIFO 4, port 8, latency 8, PRE 1) decodes exactly that slice: its
// cycles / stalls / words are what the board must reproduce, and its symbols are checked here against the
// full-matrix decode (the slice cut changes nothing in the decoded values).
//   slice <bundle dir> <matrix> <t0> <nt> <out.job>     (nt = 0: as many tiles as fit max_words)
// Job file (little endian): "HJOB", u32 version 1, nt, n_words, c_cycles, c_stalls, c_words, t0, ntt (matrix tiles),
// u16 lut[2048], u8 srt[256], 11 x (u16 limit, u16 first, u16 base), nt x 32 x (u32 start, u32 end),
// u32 words[n_words], u8 expected[32][nt][512] (lane, tile, symbol index: row lane (+32 for index >= 256)).
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

int main(int argc, char** argv) {
  if (argc < 6) { fprintf(stderr, "usage: slice <bundle> <matrix> <t0> <nt> <out.job> [max_words]\n"); return 2; }
  std::string dir = argv[1], name = argv[2];
  int t0 = atoi(argv[3]), nt = atoi(argv[4]);
  size_t max_words = argc > 6 ? strtoul(argv[6], nullptr, 10) : 16384;
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
  if (t0 >= NTT) { fprintf(stderr, "t0 %d >= %d tiles\n", t0, NTT); return 2; }
  const uint32_t base = (uint32_t)(sbit(t0, 0) >> 5);
  auto words_for = [&](int n) { return (size_t)((sbit(t0 + n - 1, H_LANES) + 31) >> 5) - base + 2; };
  if (nt <= 0) { nt = 0; while (t0 + nt < NTT && nt < 8 && words_for(nt + 1) <= max_words) ++nt; }
  if (nt <= 0 || t0 + nt > NTT || words_for(nt) > max_words) { fprintf(stderr, "slice does not fit\n"); return 3; }
  // slice matrix
  std::vector<uint32_t> ls((size_t)nt * H_LANES), wd(words_for(nt), 0u);
  for (int t = 0; t < nt; ++t)
    for (int l = 0; l < H_LANES; ++l) ls[(size_t)t * H_LANES + l] = (uint32_t)(sbit(t0 + t, l) - (uint64_t)base * 32);
  const size_t last = (size_t)((sbit(t0 + nt - 1, H_LANES) + 31) >> 5);
  for (size_t i = base; i < last && i < m.n_words; ++i) wd[i - base] = m.words[i];
  MatIn s = m;
  s.N = nt * H_TN; s.KS = H_TK; s.lane_start = ls.data(); s.words = wd.data(); s.n_words = wd.size();
  std::vector<uint8_t> ref((size_t)s.N * s.KS), full((size_t)m.N * m.KS);
  HStats cs, fs;
  decode_mat<1, 64, 4, 8, 8, 1>(tb, s, ref.data(), cs);
  decode_mat<1, 64, 4, 8, 8, 1>(tb, m, full.data(), fs);
  long long mm = 0;
  for (int t = 0; t < nt; ++t) {
    int tr = (t0 + t) / KT, tc = (t0 + t) % KT;
    for (int r = 0; r < H_TN; ++r)
      for (int c = 0; c < H_TK; ++c)
        mm += ref[((size_t)t * H_TN + r) * H_TK + c] != full[((size_t)tr * H_TN + r) * m.KS + (size_t)tc * H_TK + c];
  }
  if (mm) { fprintf(stderr, "slice decode != full-matrix decode: %lld\n", mm); return 4; }
  // job file
  FILE* f = fopen(argv[5], "wb");
  auto u32 = [&](uint32_t v) { fwrite(&v, 4, 1, f); };
  auto u16 = [&](uint16_t v) { fwrite(&v, 2, 1, f); };
  fwrite("HJOB", 1, 4, f);
  u32(1); u32(nt); u32((uint32_t)wd.size()); u32((uint32_t)cs.cycles); u32((uint32_t)cs.stalls); u32((uint32_t)cs.words);
  u32(t0); u32(NTT);
  for (int i = 0; i < (1 << H_LMAX); ++i) u16(tb.lut[i] & 0xFFF);
  fwrite(tb.sorted, 1, 256, f);
  for (int l = 1; l <= H_LMAX; ++l) { u16(tb.limit[l] & 0xFFF); u16(tb.first[l]); u16(tb.base[l]); }
  for (int t = 0; t < nt; ++t)
    for (int l = 0; l < H_LANES; ++l) {
      u32(ls[(size_t)t * H_LANES + l]);
      u32(l + 1 < H_LANES ? ls[(size_t)t * H_LANES + l + 1] : (t + 1 < nt ? ls[(size_t)(t + 1) * H_LANES] : (uint32_t)(wd.size() * 32)));
    }
  fwrite(wd.data(), 4, wd.size(), f);
  for (int l = 0; l < H_LANES; ++l)
    for (int t = 0; t < nt; ++t)
      for (int i = 0; i < H_SYMS; ++i) {
        int r = l + (i >= H_TK ? 32 : 0), c = i % H_TK;
        fputc(ref[((size_t)t * H_TN + r) * H_TK + c], f);
      }
  fclose(f);
  printf("{\"matrix\": \"%s\", \"enc\": \"%s\", \"t0\": %d, \"nt\": %d, \"ntt\": %d, \"n_words\": %zu, \"c_cycles\": %lld, "
         "\"c_stalls\": %lld, \"c_words\": %lld, \"symbols\": %lld}\n", name.c_str(), enc.c_str(), t0, nt, NTT,
         wd.size(), cs.cycles, cs.stalls, cs.words, cs.symbols);
  return 0;
}
