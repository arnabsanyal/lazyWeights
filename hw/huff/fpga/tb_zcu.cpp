// Verilator test of huff_zcu_core (zcu_core.sv) on HMAT chunk files (hw/huff/tb/chunks_main.cpp): loads the tables
// and up to NE chunks at a time through the register bus, starts the engines together, and checks every chunk's
// per-lane CRC-32 / symbol counts and cycles / stalls / words against the C model's.
//   tb_zcu <file.hmat> [max_chunks]
#include "Vhuff_zcu_core.h"
#include "verilated.h"

#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

static Vhuff_zcu_core* t;
static uint64_t ncyc = 0;
static void tick() { t->clk = 0; t->eval(); t->clk = 1; t->eval(); ++ncyc; }
static void wr(uint32_t a, uint32_t d) { t->we = 1; t->wa = a; t->wd = d; tick(); t->we = 0; }
static uint32_t rdreg(uint32_t a) { t->ra = a; for (int i = 0; i < 3; ++i) tick(); t->eval(); return t->rd; }

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  t = new Vhuff_zcu_core;
  FILE* f = fopen(argv[1], "rb");
  std::vector<uint8_t> B;
  { fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET); B.resize(n); if (fread(B.data(), 1, n, f) != (size_t)n) return 2; }
  int maxc = argc > 2 ? atoi(argv[2]) : 1 << 30;
  auto U32 = [&](size_t o) { uint32_t v; memcpy(&v, &B[o], 4); return v; };
  auto U16 = [&](size_t o) { uint16_t v; memcpy(&v, &B[o], 2); return v; };
  int nch = U32(8);
  size_t o = 28;
  size_t lut = o; o += 4096; size_t srt = o; o += 256; size_t cmp = o; o += 66;
  const int NE = 4, NTM = 16;
  t->rst = 1; for (int i = 0; i < 8; ++i) tick(); t->rst = 0; tick();
  if (rdreg(0xE00000) != 0x32465548u) { printf("bad id\n"); return 1; }
  for (int e = 0; e < NE; ++e) {
    uint32_t b = e << 21;
    (void)lut;
    for (int i = 0; i < 256; ++i) wr(b + 0x0D0000 + 4 * i, B[srt + i]);
    for (int l = 1; l <= 11; ++l) {
      size_t c = cmp + 6 * (l - 1);
      wr(b + 0x0E0000 + 16 * l + 0, U16(c)); wr(b + 0x0E0000 + 16 * l + 4, U16(c + 2)); wr(b + 0x0E0000 + 16 * l + 8, U16(c + 4));
    }
  }
  struct Ch { size_t o; uint32_t t0, nt, nw, cc, cs, cw; size_t desc, words, crc; };
  std::vector<Ch> chs;
  for (int c = 0; c < nch; ++c) {
    Ch h; h.o = o; h.t0 = U32(o); h.nt = U32(o + 4); h.nw = U32(o + 8); h.cc = U32(o + 12); h.cs = U32(o + 16); h.cw = U32(o + 20);
    o += 24; h.desc = o; o += h.nt * 32 * 8; h.words = o; o += h.nw * 4; h.crc = o; o += 128;
    chs.push_back(h);
  }
  int bad = 0, done = 0;
  for (size_t c0 = 0; c0 < chs.size() && (int)c0 < maxc; c0 += NE) {
    int n = std::min<int>(NE, chs.size() - c0);
    for (int e = 0; e < n; ++e) {
      const Ch& h = chs[c0 + e];
      uint32_t b = e << 21;
      for (uint32_t i = 0; i < h.nw; ++i) wr(b + 4 * i, U32(h.words + 4 * i));
      for (uint32_t tt = 0; tt < h.nt; ++tt)
        for (int l = 0; l < 32; ++l) {
          wr(b + 0x080000 + 8 * (l * NTM + tt), U32(h.desc + 8 * (tt * 32 + l)));
          wr(b + 0x080000 + 8 * (l * NTM + tt) + 4, U32(h.desc + 8 * (tt * 32 + l) + 4));
        }
      wr(b + 0x100000, h.nt);
    }
    uint64_t c_start = ncyc;
    wr(0xE00008, (1u << n) - 1);
    for (int i = 0; i < 4; ++i) tick();
    while (rdreg(0xE00008) != 0) {}
    for (int e = 0; e < n; ++e) {
      const Ch& h = chs[c0 + e];
      uint32_t b = e << 21;
      uint32_t cy = rdreg(b + 0x100008), st = rdreg(b + 0x10000C), wo = rdreg(b + 0x100010), stat = rdreg(b + 0x100004);
      int crcbad = 0, symbad = 0;
      for (int l = 0; l < 32; ++l) {
        crcbad += (rdreg(b + 0x100000 + 4 * (64 + l)) ^ 0xFFFFFFFFu) != U32(h.crc + 4 * l);
        symbad += rdreg(b + 0x100000 + 4 * (128 + l)) != h.nt * 512;
      }
      bool ok = !crcbad && !symbad && cy == h.cc && st == h.cs && wo == h.cw && !(stat & 4);
      bad += !ok; ++done;
      if (!ok || c0 + e < 4)
        printf("chunk %zu t0 %u nt %u: cycles %u/%u stalls %u/%u words %u/%u crc_bad %d sym_bad %d %s\n", c0 + e, h.t0,
               h.nt, cy, h.cc, st, h.cs, wo, h.cw, crcbad, symbad, ok ? "OK" : "FAIL");
    }
    (void)c_start;
  }
  printf("%s: %d chunks, %d bad\n", argv[1], done, bad);
  delete t;
  return bad ? 1 : 0;
}
