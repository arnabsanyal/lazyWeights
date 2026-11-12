// D3 C-simulation testbench: decodes ECW1 layer streams with the HLS engine model and compares the packed output
// bit-exactly with the reference (packed_w4 / packed_w8 / raw u8), the way ecw/cuda/rans_decode.cu does.
// usage: csim <index.tsv> <lookup 0|1>     index lines: name stream_file ref_file ref_offset ref_nbytes ref_storage
// Output CSV: layer,A,lookup,N,K,TN,TK,tiles,steps,words,two_row_steps,two_table_steps,cycles_ii1,bit_exact,integrity
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "../rans_engine.h"

using namespace ecwhw;

static std::vector<uint8_t> rd(const std::string& p, long long off = 0, long long nb = -1) {
  FILE* f = fopen(p.c_str(), "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(2); }
  if (nb < 0) { fseek(f, 0, SEEK_END); nb = ftell(f) - off; }
  fseek(f, off, SEEK_SET);
  std::vector<uint8_t> b(nb);
  if (fread(b.data(), 1, nb, f) != (size_t)nb) exit(3);
  fclose(f);
  return b;
}
static size_t r4(size_t n) { return (n + 3) & ~(size_t)3; }

template <int A, int LK>
static bool run_layer(const std::vector<uint8_t>& b, std::vector<uint8_t>& out, uint32_t& N, uint32_t& K,
                      uint32_t& TN, uint32_t& TK, uint32_t& ntiles, Stats& st, long long& integrity_fail) {
  const uint32_t* u = (const uint32_t*)b.data();
  N = u[1]; K = u[2]; TN = u[3]; TK = u[4];
  const uint32_t nt = u[6];
  ntiles = u[7];
  static Tables T;
  T.A = A; T.n = nt;
  size_t o = 32 + r4(nt);
  for (uint32_t t = 0; t < nt; ++t)
    for (int s = 0; s < A; ++s) { T.freq[t][s] = b[o] | (b[o + 1] << 8); o += 2; }
  for (uint32_t t = 0; t < nt; ++t) {
    uint32_t acc = 0;
    for (int s = 0; s < A; ++s) {
      T.cum[t][s] = (uint16_t)acc;
      for (uint32_t j = 0; j < T.freq[t][s]; ++j) T.lut[t][acc + j] = (uint8_t)s;
      acc += T.freq[t][s];
    }
    for (int s = A; s < MAX_A; ++s) T.cum[t][s] = M;
  }
  const uint8_t* rt = b.data() + o;
  o += r4(N);
  const uint32_t* meta = (const uint32_t*)(b.data() + o);
  o += 4 * 33 * (size_t)ntiles;
  uint32_t plen; memcpy(&plen, b.data() + o, 4); o += 4;
  const uint16_t* payload = (const uint16_t*)(b.data() + o);
  out.assign((size_t)N * K, 0xff);
  static uint8_t sym[MAX_TN * MAX_TK];
  const uint32_t tpr = (K + TK - 1) / TK;
  bool all_ok = true;
  for (uint32_t t = 0; t < ntiles; ++t) {
    TileIn in;
    const uint32_t r0 = (t / tpr) * TN, c0 = (t % tpr) * TK;
    in.tn = std::min(TN, N - r0);
    in.tk = std::min(TK, K - c0);
    for (int l = 0; l < LANES; ++l) in.state[l] = meta[t * 33 + 1 + l];
    const uint32_t b0 = meta[t * 33], b1 = (t + 1 < ntiles) ? meta[(t + 1) * 33] : plen;
    in.words = payload + b0 / 2;
    in.n_words = (int)((b1 - b0) / 2);
    in.row_tab = rt + r0;
    if (!decode_tile<A, LK>(T, in, sym, st)) { ++integrity_fail; all_ok = false; }
    for (int r = 0; r < in.tn; ++r)
      memcpy(&out[(size_t)(r0 + r) * K + c0], &sym[r * in.tk], in.tk);
  }
  return all_ok;
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: csim index.tsv lookup\n"); return 2; }
  const int lk = atoi(argv[2]);
  FILE* idx = fopen(argv[1], "r");
  char name[512], sfile[1024], rfile[1024], rstor[32];
  long long roff, rnb, n_fail = 0;
  printf("layer,A,lookup,N,K,TN,TK,tiles,steps,words,two_row_steps,two_table_steps,cycles_ii1,bit_exact,integrity\n");
  while (fscanf(idx, "%511s %1023s %1023s %lld %lld %31s", name, sfile, rfile, &roff, &rnb, rstor) == 6) {
    std::vector<uint8_t> b = rd(sfile), out;
    const uint32_t A = ((const uint32_t*)b.data())[5];
    uint32_t N, K, TN, TK, ntiles;
    Stats st;
    long long integ = 0;
    if (A == 16) {
      if (lk) run_layer<16, 1>(b, out, N, K, TN, TK, ntiles, st, integ);
      else run_layer<16, 0>(b, out, N, K, TN, TK, ntiles, st, integ);
    } else {
      run_layer<256, 0>(b, out, N, K, TN, TK, ntiles, st, integ);   // A = 256: replicated table
    }
    std::vector<uint8_t> ref = rd(rfile, roff, rnb);
    bool exact = true;
    if (std::string(rstor) == "packed_w4") {
      const size_t kb = (K + 1) / 2;
      for (size_t n = 0; n < N && exact; ++n)
        for (size_t k = 0; k < K; ++k) {
          const uint8_t v = ref[n * kb + k / 2];
          if (out[n * K + k] != ((k & 1) ? (v >> 4) : (v & 15))) { exact = false; break; }
        }
    } else {
      exact = ref.size() == out.size() && memcmp(ref.data(), out.data(), out.size()) == 0;
    }
    n_fail += !exact || integ;
    printf("%s,%u,%d,%u,%u,%u,%u,%u,%lld,%lld,%lld,%lld,%lld,%d,%d\n", name, A, A == 16 ? lk : 0, N, K, TN, TK, ntiles,
           st.steps, st.words, st.two_row_steps, st.two_table_steps, st.steps, exact ? 1 : 0, integ ? 0 : 1);
    fflush(stdout);
  }
  fprintf(stderr, "failures: %lld\n", n_fail);
  return n_fail ? 1 : 0;
}
