// llmrt: kernels/phi3spec.cu generalized to llama-family models (phi3, Mistral-7B, SmolLM): bundles from
// scripts/llm_bundle.py with per-row scale / zero point (side tensors <name>.rs, .rsub, .radd), codes re-centered per
// row, huf8 (8-bit, one weight per symbol) or huf4p (4-bit, two weights per symbol, 64 x 512-weight tiles);
// grouped-query attention (n_kv_heads), head_dim 64 / 96 / 128, EOS ids from cfg "eos". phi3 bundles
// (scripts/rt_convert.py, q8_bundle.py: per-tensor s, z, raw codes) load unchanged (row side info synthesized).
// Huffman matrices are interleaved at load time, so they must be resident; a bundle written by `export-il` stores them
// interleaved (enc suffix _il, _ilc: compact layout) and then they stream like raw tensors (memory-constrained devices). Memory: compact interleaved layout
// (IlLayout: every lane padded to its tile's longest lane by default; LLMRT_COMPACT=1: lanes interleaved up to the
// tile's shortest lane, the rest in per-lane runs, no padding, about 10-17% slower lane reads on the Orin), embedding rows gathered on the host from the mmap'ed bundle (LLMRT_EMBED_DEVICE=1:
// resident), prefill scratch capped at LLMRT_SCRATCH_MB (default 16). Group-wise side info (g128): side tensors
// <name>.rs / .rsub / .radd of shape [N, G] (G groups of K / G weights per row; per-row bundles: G = 1), codes
// re-centered per group; embed.rs / embed.rsub likewise.
//
// phi3spec: fork of rt/phi3rt.cu (c8d3889) for multi-token fused Huffman GEMM and prompt-lookup speculative decoding.
// Changes against phi3rt:
//   - resident huf8 matrices are re-laid out at load time as interleaved lane words (word i of lane l at
//     tbase[t] + 32 i + l; kernels/huf2_bench.cu gemv5), device layout IlLayout
//   - k_gemvm_hufi<MT, B> (was k_gemvm_huf8i<MT>): fused decode + GEMM for m <= 8 tokens (MT = 1, 2, 4, 8): each weight decoded once and used
//     for every token; a block's 8 warps share one tile column, so x is staged once per block
//   - CLI `spec`: greedy generation, plain vs prompt-lookup speculation (draft = continuation of the latest earlier
//     match of the last n-gram, verified in one forward pass; output identical to greedy up to float ties);
//     `fwd-bench`: forward-pass time vs number of tokens m
//   PHI3SPEC_NO_IL=1 keeps the original per-lane layout and kernels (m > 1 then always decodes into scratch + GEMM).
//
// phi3rt: runtime for the uint8 (weight-only, per-tensor) phi3-mini ONNX model, built from a bundle written by
// scripts/rt_convert.py (format: ecw/rt/bundle.py). Targets the Jetson Nano (sm_53, CUDA 10.2) and builds for
// the A100 (sm_80) for testing. C++14, no external dependencies.
//
// The ONNX DequantizeLinear nodes are folded:
//   MatMul weights stay uint8 codes q with per-tensor (s, z): y_n = s * sum_k x_k (q_nk - z)
//     raw8: q stored [N, K]; huf8: q Huffman-coded in 64 x 256 tiles, 32 lanes per tile (decoded in registers)
//   embedding: codes gathered by row, dequantized like ORT (fp32 (q - z) * s, rounded to fp16)
//   norm weights, rotary cos/sin: constant-folded at conversion (f32)
// Activations are fp32, the KV cache fp16 (ORT runs the graph in fp16).
//
// Weights that do not fit in device memory (MemAvailable floor, or --budget-mb) are streamed from the bundle
// file each forward pass through two device staging buffers with a loader thread.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

#define CK(x)                                                                                  \
  do {                                                                                         \
    cudaError_t e_ = (x);                                                                      \
    if (e_ != cudaSuccess) {                                                                   \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
      exit(1);                                                                                 \
    }                                                                                          \
  } while (0)

static const int TN = 64, TK = 256, LMAX = 11, LUTN = 1 << LMAX;

// wall-clock seconds since the epoch (phase marks aligned with tegrastats power logs)
static double epoch_s() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

static double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static long long meminfo_kb(const char* key) {
  FILE* f = fopen("/proc/meminfo", "r");
  if (!f) return -1;
  char k[64];
  long long v = -1, x;
  while (fscanf(f, "%63s %lld kB", k, &x) == 2)
    if (!strncmp(k, key, strlen(key)) && k[strlen(key)] == ':') { v = x; break; }
  fclose(f);
  return v;
}

// ── kernels ─────────────────────────────────────────────────────────────────────────────────────────────

// fp16 load through the read-only cache
__device__ __forceinline__ float ld_h(const __half* p) {
  return __half2float(__ushort_as_half(__ldg(reinterpret_cast<const unsigned short*>(p))));
}

// float(b) for a byte b without I2F (quarter rate on Maxwell): 0x4B0000bb is 2^23 + b.
__device__ __forceinline__ float byte_f(uint32_t w, uint32_t sel) {
  return __uint_as_float(__byte_perm(w, 0x4B000000u, sel));
}

// K-quant side info (scripts/a2_kq_bundle.py): the GGUF side bytes of each 256-weight super-block, [N, K / 256, SB].
// kqt 1 Q4_K, 2 Q5_K (SB 16: fp16 d, dmin, 6-bit sc / m of 8 sub-blocks of 32), 3 Q6_K (SB 18: int8 sc of 16
// sub-blocks of 16, fp16 d), 4 IQ4_XS (SB 8: fp16 d, 6-bit ls of 8 sub-blocks of 32). weight = s f(q) - o with
// f(q) = q (Q4_K, Q5_K), q - 32 (Q6_K), kv[q] (IQ4_XS); codes are the format's own (not re-centered).
__constant__ float c_kv[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
__host__ __device__ constexpr int kq_sbytes(int kqt) { return kqt == 3 ? 18 : kqt == 4 ? 8 : 16; }
__host__ __device__ constexpr int kq_gw(int kqt) { return kqt == 3 ? 16 : 32; }   // weights per sub-block

__device__ __forceinline__ float kq_h(const uint8_t* p) {
  return __half2float(__ushort_as_half((unsigned short)(__ldg(p) | __ldg(p + 1) << 8)));
}

// (s, o) of sub-block j of the super-block whose side bytes start at p
template <int KQ>
__device__ __forceinline__ void kq_side(const uint8_t* p, int j, float& s, float& o) {
  if (KQ == 1 || KQ == 2) {   // llama.cpp get_scale_min_k4
    const uint8_t* q = p + 4;
    int sc, m;
    if (j < 4) {
      sc = __ldg(q + j) & 63;
      m = __ldg(q + j + 4) & 63;
    } else {
      sc = (__ldg(q + j + 4) & 15) | ((__ldg(q + j - 4) >> 6) << 4);
      m = (__ldg(q + j + 4) >> 4) | ((__ldg(q + j) >> 6) << 4);
    }
    s = kq_h(p) * (float)sc;
    o = kq_h(p + 2) * (float)m;
  } else if (KQ == 3) {
    s = kq_h(p + 16) * (float)(int8_t)__ldg(p + j);
    o = 0.f;
  } else {
    const int hi = __ldg(p + 2) | __ldg(p + 3) << 8;
    const int ls = ((__ldg(p + 4 + j / 2) >> (4 * (j & 1))) & 15) | (((hi >> (2 * j)) & 3) << 4);
    s = kq_h(p) * (float)(ls - 32);
    o = 0.f;
  }
}

// f(q); kv: the IQ4_XS grid (shared memory in the fused kernel, c_kv in the prefill GEMMs)
template <int KQ>
__device__ __forceinline__ float kq_f(uint32_t q, const float* kv) {
  if (KQ == 4) return kv[q];
  return __uint_as_float(0x4B000000u | q) - (KQ == 3 ? 8388640.f : 8388608.f);
}

// prefill GEMM weight load: 4 codes (bytes of v) at columns k..k+3 of row n (one sub-block) -> weights
template <int KQ>
__device__ __forceinline__ void kq_w4t(const uint8_t* side, int K, int n, int k, uint32_t v, float* w) {
  float s, o;
  kq_side<KQ>(side + ((size_t)n * (K / 256) + k / 256) * kq_sbytes(KQ), (k & 255) / kq_gw(KQ), s, o);
#pragma unroll
  for (int i = 0; i < 4; ++i) w[i] = s * kq_f<KQ>((v >> (8 * i)) & 255u, c_kv) - o;
}
__device__ __forceinline__ void kq_w4(int kqt, const uint8_t* side, int K, int n, int k, uint32_t v, float* w) {
  if (kqt == 1) kq_w4t<1>(side, K, n, k, v, w);
  else if (kqt == 2) kq_w4t<2>(side, K, n, k, v, w);
  else if (kqt == 3) kq_w4t<3>(side, K, n, k, v, w);
  else kq_w4t<4>(side, K, n, k, v, w);
}

// raw8 GEMV, M = 1: y[n] = s * sum_k x[k] (q[n,k] - z). 8 warps per block, 4 rows per warp, x staged in smem.
__global__ void k_gemv_raw8(const uint8_t* __restrict__ W, const float* __restrict__ x, float* __restrict__ y,
                            int N, int K, float s, float zc) {  // zc = 2^23 + z
  extern __shared__ float xs[];
  for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = x[i];
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int KW = K >> 2;
  for (int rr = 0; rr < 4; ++rr) {
    const int n = (blockIdx.x * 8 + warp) * 4 + rr;
    if (n >= N) return;
    const uint32_t* row = reinterpret_cast<const uint32_t*>(W + (size_t)n * K);
    float a0 = 0.f, a1 = 0.f;
#pragma unroll 4
    for (int w = lane; w < KW; w += 32) {
      const uint32_t v = __ldg(row + w);
      const float4 xv = reinterpret_cast<const float4*>(xs)[w];
      a0 = fmaf(xv.x, byte_f(v, 0x7440) - zc, a0);
      a1 = fmaf(xv.y, byte_f(v, 0x7441) - zc, a1);
      a0 = fmaf(xv.z, byte_f(v, 0x7442) - zc, a0);
      a1 = fmaf(xv.w, byte_f(v, 0x7443) - zc, a1);
    }
    float a = a0 + a1;
    for (int o = 16; o; o >>= 1) a += __shfl_down_sync(0xffffffffu, a, o);
    if (lane == 0) y[n] = s * a;
  }
}

// raw8 GEMM, M > 1: Y[m,n] = s * sum_k X[m,k] (q[n,k] - z). 64 (m) x 64 (n) tile, BK = 16, 4 x 4 per thread.
// grs != null: group side info (G groups per row, compact: fp16 scale, byte zero point):
// weight = grs[n G + g] (q - grz[n G + g]); s, zc unused.
__device__ __forceinline__ void wgroup(const __half* grs, const uint8_t* grz, int G, int K, int n, int k, float& zc,
                                       float& gs) {
  const size_t i = (size_t)n * G + k / (K / G);
  zc = __uint_as_float(0x4B000000u | (uint32_t)__ldg(grz + i));
  gs = ld_h(grs + i);
}

__global__ void k_gemm_raw8(const uint8_t* __restrict__ W, const float* __restrict__ X, float* __restrict__ Y,
                            int M, int N, int K, float s, float zc, int ldy, const __half* __restrict__ grs,
                            const uint8_t* __restrict__ grzc, int G, const uint8_t* __restrict__ kqs, int kqt) {
  __shared__ float Xs[16][64 + 4];
  __shared__ float Ws[16][64 + 4];
  const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
  const int m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
  float acc[4][4] = {};
  const int lr = tid >> 2, lk = (tid & 3) * 4;
  for (int k0 = 0; k0 < K; k0 += 16) {
    const uint32_t v = __ldg(reinterpret_cast<const uint32_t*>(W + (size_t)(n0 + lr) * K + k0 + lk));
    if (kqt) {
      float w[4];
      kq_w4(kqt, kqs, K, n0 + lr, k0 + lk, v, w);
      Ws[lk + 0][lr] = w[0];
      Ws[lk + 1][lr] = w[1];
      Ws[lk + 2][lr] = w[2];
      Ws[lk + 3][lr] = w[3];
    } else if (grs) {
      float gz, gs;
      wgroup(grs, grzc, G, K, n0 + lr, k0 + lk, gz, gs);
      Ws[lk + 0][lr] = gs * (byte_f(v, 0x7440) - gz);
      Ws[lk + 1][lr] = gs * (byte_f(v, 0x7441) - gz);
      Ws[lk + 2][lr] = gs * (byte_f(v, 0x7442) - gz);
      Ws[lk + 3][lr] = gs * (byte_f(v, 0x7443) - gz);
    } else {
      Ws[lk + 0][lr] = byte_f(v, 0x7440) - zc;
      Ws[lk + 1][lr] = byte_f(v, 0x7441) - zc;
      Ws[lk + 2][lr] = byte_f(v, 0x7442) - zc;
      Ws[lk + 3][lr] = byte_f(v, 0x7443) - zc;
    }
    float4 xv = make_float4(0.f, 0.f, 0.f, 0.f);
    if (m0 + lr < M) xv = *reinterpret_cast<const float4*>(X + (size_t)(m0 + lr) * K + k0 + lk);
    Xs[lk + 0][lr] = xv.x;
    Xs[lk + 1][lr] = xv.y;
    Xs[lk + 2][lr] = xv.z;
    Xs[lk + 3][lr] = xv.w;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      float a[4], b[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) { a[i] = Xs[k][ty * 4 + i]; b[i] = Ws[k][tx * 4 + i]; }
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int m = m0 + ty * 4 + i;
    if (m < M)
#pragma unroll
      for (int j = 0; j < 4; ++j) Y[(size_t)m * ldy + n0 + tx * 4 + j] = s * acc[i][j];
  }
}

// raw8 GEMM for larger M: 128 (m) x 128 (n) tile, BK = 8, 256 threads, 8 x 8 outputs per thread (rows/cols
// {4t..4t+3} and {64+4t..64+4t+3}, conflict-free float4 smem reads), next tile prefetched into registers.
__global__ void __launch_bounds__(256) k_gemm2_raw8(const uint8_t* __restrict__ W, const float* __restrict__ X,
                                                    float* __restrict__ Y, int M, int N, int K, float s, float zc,
                                                    int ldy, const __half* __restrict__ grs,
                                                    const uint8_t* __restrict__ grzc, int G,
                                                    const uint8_t* __restrict__ kqs, int kqt) {
  __shared__ __align__(16) float Xs[2][8][128];
  __shared__ __align__(16) float Ws[2][8][128];
  const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
  const int m0 = blockIdx.y * 128, n0 = blockIdx.x * 128;
  const int lr = tid >> 1, lk = (tid & 1) * 4;
  const bool xin = m0 + lr < M;
  const float* xp = X + (size_t)(m0 + lr) * K + lk;
  const uint32_t* wp = reinterpret_cast<const uint32_t*>(W + (size_t)(n0 + lr) * K + lk);
  float acc[8][8] = {};
  float4 xv = xin ? *reinterpret_cast<const float4*>(xp) : make_float4(0.f, 0.f, 0.f, 0.f);
  uint32_t wv = __ldg(wp);
  int buf = 0;
  for (int k0 = 0; k0 < K; k0 += 8) {
    Xs[buf][lk + 0][lr] = xv.x;
    Xs[buf][lk + 1][lr] = xv.y;
    Xs[buf][lk + 2][lr] = xv.z;
    Xs[buf][lk + 3][lr] = xv.w;
    if (kqt) {
      float w[4];
      kq_w4(kqt, kqs, K, n0 + lr, k0 + lk, wv, w);
      Ws[buf][lk + 0][lr] = w[0];
      Ws[buf][lk + 1][lr] = w[1];
      Ws[buf][lk + 2][lr] = w[2];
      Ws[buf][lk + 3][lr] = w[3];
    } else if (grs) {
      float gz, gs;
      wgroup(grs, grzc, G, K, n0 + lr, k0 + lk, gz, gs);
      Ws[buf][lk + 0][lr] = gs * (byte_f(wv, 0x7440) - gz);
      Ws[buf][lk + 1][lr] = gs * (byte_f(wv, 0x7441) - gz);
      Ws[buf][lk + 2][lr] = gs * (byte_f(wv, 0x7442) - gz);
      Ws[buf][lk + 3][lr] = gs * (byte_f(wv, 0x7443) - gz);
    } else {
      Ws[buf][lk + 0][lr] = byte_f(wv, 0x7440) - zc;
      Ws[buf][lk + 1][lr] = byte_f(wv, 0x7441) - zc;
      Ws[buf][lk + 2][lr] = byte_f(wv, 0x7442) - zc;
      Ws[buf][lk + 3][lr] = byte_f(wv, 0x7443) - zc;
    }
    __syncthreads();
    if (k0 + 8 < K) {
      xv = xin ? *reinterpret_cast<const float4*>(xp + k0 + 8) : make_float4(0.f, 0.f, 0.f, 0.f);
      wv = __ldg(wp + (k0 + 8) / 4);
    }
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const float4 a0 = *reinterpret_cast<const float4*>(&Xs[buf][k][ty * 4]);
      const float4 a1 = *reinterpret_cast<const float4*>(&Xs[buf][k][64 + ty * 4]);
      const float4 b0 = *reinterpret_cast<const float4*>(&Ws[buf][k][tx * 4]);
      const float4 b1 = *reinterpret_cast<const float4*>(&Ws[buf][k][64 + tx * 4]);
      const float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
      const float b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
      for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
    }
    buf ^= 1;
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int m = m0 + (i < 4 ? ty * 4 + i : 64 + ty * 4 + i - 4);
    if (m >= M) continue;
    float* yr = Y + (size_t)m * ldy + n0;
    *reinterpret_cast<float4*>(yr + tx * 4) = make_float4(s * acc[i][0], s * acc[i][1], s * acc[i][2], s * acc[i][3]);
    *reinterpret_cast<float4*>(yr + 64 + tx * 4) =
        make_float4(s * acc[i][4], s * acc[i][5], s * acc[i][6], s * acc[i][7]);
  }
}

// huf8 lane decoder state: a 96-bit window (w0, w1, w2) over the lane's bitstream, bp = bit position in w0.
struct Lane {
  const uint32_t* p;
  uint32_t w0, w1, w2, bp;
  __device__ __forceinline__ void init(const uint32_t* payload, uint32_t start) {
    p = payload + (start >> 5);
    bp = start & 31;
    w0 = __ldg(p);
    w1 = __ldg(p + 1);
    w2 = __ldg(p + 2);
    p += 3;
  }
  __device__ __forceinline__ uint32_t next(const uint16_t* lut) {  // returns lut entry: sym | len << 8
    const uint32_t e = lut[__funnelshift_r(w0, w1, bp) & (LUTN - 1)];
    bp += e >> 8;
    if (bp >= 32) {
      bp -= 32;
      w0 = w1;
      w1 = w2;
      w2 = __ldg(p++);
    }
    return e;
  }
};

// huf8 fused decode + GEMV, M = 1. One warp per tile (64 x 256); lane l owns tile rows l and l + 32, so each
// lane accumulates a whole row segment in a register (x is broadcast from smem). Split-K by atomics: y must be 0.
__global__ void k_gemv_huf8(const uint8_t* __restrict__ blob, const float* __restrict__ x, float* __restrict__ y,
                            int N, int K, float s, float zc) {
  __shared__ uint16_t lut[LUTN];
  __shared__ float xs[8][TK];
  const uint16_t* glut = reinterpret_cast<const uint16_t*>(blob);
  for (int i = threadIdx.x; i < LUTN; i += blockDim.x) lut[i] = glut[i];
  const int KT = K / TK, T = (N / TN) * KT;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int t = blockIdx.x * 8 + warp;
  const uint32_t* starts = reinterpret_cast<const uint32_t*>(blob + 2 * LUTN);
  const uint32_t* payload = starts + (size_t)T * 32;
  const int tr = t / KT, tc = t - tr * KT;
  if (t < T)
    for (int i = lane; i < TK; i += 32) xs[warp][i] = x[tc * TK + i];
  __syncthreads();
  if (t >= T) return;
  Lane L;
  L.init(payload, __ldg(starts + (size_t)t * 32 + lane));
  const float* xw = xs[warp];
#pragma unroll 1
  for (int half = 0; half < 2; ++half) {
    float a0 = 0.f, a1 = 0.f;
#pragma unroll 8
    for (int c = 0; c < TK; c += 2) {
      const uint32_t e0 = L.next(lut);
      a0 = fmaf(xw[c], __uint_as_float(0x4B000000u | (e0 & 255u)) - zc, a0);
      const uint32_t e1 = L.next(lut);
      a1 = fmaf(xw[c + 1], __uint_as_float(0x4B000000u | (e1 & 255u)) - zc, a1);
    }
    atomicAdd(y + tr * TN + half * 32 + lane, s * (a0 + a1));
  }
}

// huf8 decode to raw8 codes for tile rows [tr0, tr1): out[(n - tr0*64) * K + k].
__global__ void k_decode_huf8(const uint8_t* __restrict__ blob, uint8_t* __restrict__ out, int N, int K, int tr0,
                              int tr1) {
  __shared__ uint16_t lut[LUTN];
  const uint16_t* glut = reinterpret_cast<const uint16_t*>(blob);
  for (int i = threadIdx.x; i < LUTN; i += blockDim.x) lut[i] = glut[i];
  __syncthreads();
  const int KT = K / TK, T = (N / TN) * KT;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int t = tr0 * KT + blockIdx.x * 8 + warp;
  if (t >= tr1 * KT) return;
  const uint32_t* starts = reinterpret_cast<const uint32_t*>(blob + 2 * LUTN);
  const uint32_t* payload = starts + (size_t)T * 32;
  const int tr = t / KT, tc = t - tr * KT;
  Lane L;
  L.init(payload, __ldg(starts + (size_t)t * 32 + lane));
  for (int half = 0; half < 2; ++half) {
    uint32_t* row = reinterpret_cast<uint32_t*>(out + (size_t)((tr - tr0) * TN + half * 32 + lane) * K + tc * TK);
#pragma unroll 1
    for (int c = 0; c < TK / 16; ++c) {   // 16 symbols -> one 16-byte store
      uint32_t v[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        v[j] = L.next(lut) & 255u;
        v[j] |= (L.next(lut) & 255u) << 8;
        v[j] |= (L.next(lut) & 255u) << 16;
        v[j] |= (L.next(lut) & 255u) << 24;
      }
      reinterpret_cast<uint4*>(row)[c] = make_uint4(v[0], v[1], v[2], v[3]);
    }
  }
}

// Y[j, n] += c * sum_k X[j, k] (one block per row j)
__global__ void k_rowsum_add(const float* __restrict__ X, float* __restrict__ Y, int N, int K, int ldy, float c) {
  __shared__ float red[256];
  const int j = blockIdx.x;
  float a = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x) a += X[(size_t)j * K + k];
  red[threadIdx.x] = a;
  __syncthreads();
  for (int o = 128; o; o >>= 1) {
    if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
    __syncthreads();
  }
  const float v = c * red[0];
  for (int n = threadIdx.x; n < N; n += blockDim.x) Y[(size_t)j * ldy + n] += v;
}

// Compact interleaved layout of one Huffman matrix (T tiles, 32 lanes each):
//   [LUT 4 KB][tbase T x u32][wc T x u16, padded to even][ovf T x 32 x u16][words]
// Tile t's words start at words + tbase[t]: wc[t] (its shortest lane's length) interleaved words per lane (word i of
// lane l at 32 i + l), then the remaining words of each lane contiguous (lane l's at 32 wc[t] + ovf[t][l]). Reads
// past a lane's end land in the following words (or the zero tail), whose bits the decoder never uses.
struct IlLayout {
  const uint32_t* tbase;
  const uint16_t* wc;
  const uint16_t* ovf;
  const uint32_t* words;
  __device__ __forceinline__ IlLayout(const uint8_t* base, int T) {
    tbase = reinterpret_cast<const uint32_t*>(base + 2 * LUTN);
    wc = reinterpret_cast<const uint16_t*>(tbase + T);
    ovf = wc + ((T + 1) & ~1);
    words = reinterpret_cast<const uint32_t*>(ovf + (size_t)T * 32);
  }
};

// interleaved huf8 lane on the padded layout (wc[t] = the longest lane, no overflow runs): word i at 32 i + lane
struct LaneP {
  const uint32_t* p;
  uint32_t w0, w1, w2, bp;
  __device__ __forceinline__ void init(const IlLayout& g, int t, int lane) {
    p = g.words + __ldg(g.tbase + t) + lane;
    bp = 0;
    w0 = __ldg(p);
    w1 = __ldg(p + 32);
    w2 = __ldg(p + 64);
    p += 96;
  }
  __device__ __forceinline__ uint32_t next(const uint16_t* lut) {
    const uint32_t e = lut[__funnelshift_r(w0, w1, bp) & (LUTN - 1)];
    bp += e >> 8;
    if (bp >= 32) {
      bp -= 32;
      w0 = w1;
      w1 = w2;
      w2 = __ldg(p);
      p += 32;
    }
    return e;
  }
};

// interleaved huf8 lane on the compact layout: interleaved part (stride 32), then the lane's overflow run (stride 1)
struct LaneC {
  const uint32_t* tw;           // the tile's words
  uint32_t i, iend, ovs, st;    // word index, end of the lane's interleaved part, start of its overflow run, stride
  uint32_t w0, w1, w2, bp;
  __device__ __forceinline__ uint32_t fetch() {
    const uint32_t v = __ldg(tw + i);
    i += st;
    if (i == iend) {   // once: overflow indices can reach iend again
      i = ovs;
      st = 1;
      iend = ~0u;
    }
    return v;
  }
  __device__ __forceinline__ void init(const IlLayout& g, int t, int lane) {
    tw = g.words + __ldg(g.tbase + t);
    const uint32_t wc = __ldg(g.wc + t);
    ovs = 32 * wc + __ldg(g.ovf + (size_t)t * 32 + lane);
    iend = wc ? 32 * wc + lane : ~0u;
    i = wc ? lane : ovs;
    st = wc ? 32 : 1;
    bp = 0;
    w0 = fetch();
    w1 = fetch();
    w2 = fetch();
  }
  __device__ __forceinline__ uint32_t next(const uint16_t* lut) {
    const uint32_t e = lut[__funnelshift_r(w0, w1, bp) & (LUTN - 1)];
    bp += e >> 8;
    if (bp >= 32) {
      bp -= 32;
      w0 = w1;
      w1 = w2;
      w2 = fetch();
    }
    return e;
  }
};

// group side info (G > 1), compact: fp16 scale gs[n G + g], code zero point gz[n G + g] (byte); decoder add = gz + goff
// (goff = 2^(B-1) for re-centered Huffman codes; goff < 0: add 0, packed codes)
// packed (fixed-rate) lane: the A2 baselines pk4 / pk8 (no entropy coding) in the same kernels. A tile's codes are
// stored as 32-bit words, word i of lane l at t * 32 W + 32 i + l (W = words per lane, the same for every lane);
// next() returns the next byte (pk8: one code; pk4: two codes, low nibble first), as a Huffman symbol.
struct LaneR {
  const uint32_t* p;
  uint32_t w, sh;
  __device__ __forceinline__ void init(const uint32_t* words, int t, int lane, int W) {
    p = words + (size_t)t * 32 * W + lane;
    w = __ldg(p);
    p += 32;
    sh = 0;
  }
  __device__ __forceinline__ uint32_t next(const uint16_t*) {
    if (sh == 32) {
      w = __ldg(p);
      p += 32;
      sh = 0;
    }
    const uint32_t e = (w >> sh) & 255u;
    sh += 8;
    return e;
  }
  __device__ __forceinline__ uint32_t word() {   // next 4 symbols (word-aligned use only)
    if (sh) {
      w = __ldg(p);
      p += 32;
    }
    sh = 32;
    return w;
  }
};

// packed Q5_K / Q6_K codes at the GGUF width (LT 3: 5 bits, LT 4: 6 bits; pk8 bundles are repacked at load): a tile's
// low nibbles as LaneR words (lane l: rows l then l + 32, 8 codes per word, code 2 b in the low nibble of byte b), and
// after all tiles' nibble words the high bits (HB = 1 or 2 per code, code i at bits HB i of its word, 32 / HB codes
// per word), laid out the same way. next() / word() return codes as LaneR does for pk8, so every kernel reads it.
template <int HB>
struct LaneQ {
  const uint32_t *pn, *ph;
  uint32_t nw, hw;
  int nc, hc;   // codes taken from nw (of 8) and hw (of 32 / HB)
  __device__ __forceinline__ void init(const uint32_t* words, int T, int t, int lane) {
    constexpr int WN = 2 * TK / 8, WH = 2 * TK * HB / 32;   // words per lane per tile
    pn = words + (size_t)t * 32 * WN + lane;
    ph = words + (size_t)T * 32 * WN + (size_t)t * 32 * WH + lane;
    nc = 8;   // first next() / word() loads
    hc = 32 / HB;
  }
  // one K-quant sub-block of 4 G codes (Q5_K: G = 8, 32 codes; Q6_K: G = 4, 16 codes): G / 2 nibble words and one word
  // of high bits, as G words of 4 codes (code i at bits 8 i); no counters (do not mix with next() / word())
  template <int G>
  __device__ __forceinline__ void group(uint32_t (&u)[G]) {
    static_assert(4 * G * HB == 32, "one high word per sub-block");
    uint32_t n[G / 2];
#pragma unroll
    for (int i = 0; i < G / 2; ++i) n[i] = __ldg(pn + 32 * i);
    const uint32_t h = __ldg(ph);
    pn += 32 * (G / 2);
    ph += 32;
#pragma unroll
    for (int w = 0; w < G; ++w) {
      const uint32_t lo = n[w / 2] & 0x0F0F0F0Fu, hi = (n[w / 2] >> 4) & 0x0F0F0F0Fu;
      const uint32_t q = __byte_perm(lo, hi, w & 1 ? 0x7362 : 0x5140);
      const uint32_t x = h >> (4 * HB * w);
      const uint32_t hs = HB == 1 ? ((x & 15u) * 0x00204081u) & 0x01010101u
                                  : (x & 3u) | (x & 0xCu) << 6 | (x & 0x30u) << 12 | (x & 0xC0u) << 18;
      u[w] = q | hs << 4;
    }
  }
  __device__ __forceinline__ void refill() {
    if (nc == 8) { nw = __ldg(pn); pn += 32; nc = 0; }
    if (hc == 32 / HB) { hw = __ldg(ph); ph += 32; hc = 0; }
  }
  __device__ __forceinline__ uint32_t next(const uint16_t*) {
    refill();
    const uint32_t e = ((nw >> (4 * nc)) & 15u) | ((hw >> (HB * hc)) & ((1u << HB) - 1)) << 4;
    ++nc; ++hc;
    return e;
  }
  __device__ __forceinline__ uint32_t word() {   // next 4 codes (4-aligned use only), code i at bits 8 i
    refill();
    const uint32_t lo = nw & 0x0F0F0F0Fu, hi = (nw >> 4) & 0x0F0F0F0Fu;
    const uint32_t q = __byte_perm(lo, hi, nc ? 0x7362 : 0x5140);
    const uint32_t h = hw >> (HB * hc);
    uint32_t hs;
    if (HB == 1) hs = ((h & 15u) * 0x00204081u) & 0x01010101u;   // bit i -> bit 8 i
    else hs = (h & 3u) | (h & 0xCu) << 6 | (h & 0x30u) << 12 | (h & 0xC0u) << 18;   // bits 2i..2i+1 -> 8i..8i+1
    nc += 4; hc += 4;
    return q | hs << 4;
  }
};

// lane reader of a tile: LT 0 = Huffman padded layout (LaneP), 1 = Huffman compact (LaneC), 2 = packed (LaneR),
// 3 / 4 = packed Q5_K / Q6_K at the GGUF width (LaneQ)
template <int LT>
struct LaneOf { typedef LaneP type; };
template <>
struct LaneOf<1> { typedef LaneC type; };
template <>
struct LaneOf<2> { typedef LaneR type; };
template <>
struct LaneOf<3> { typedef LaneQ<1> type; };
template <>
struct LaneOf<4> { typedef LaneQ<2> type; };

template <int LT>
__device__ __forceinline__ void lane_init(typename LaneOf<LT>::type& L, const uint8_t* base, int T, int t, int lane,
                                          int W) {
  L.init(IlLayout(base, T), t, lane);
}
template <>
__device__ __forceinline__ void lane_init<2>(LaneR& L, const uint8_t* base, int, int t, int lane, int W) {
  L.init(reinterpret_cast<const uint32_t*>(base), t, lane, W);
}
template <>
__device__ __forceinline__ void lane_init<3>(LaneQ<1>& L, const uint8_t* base, int T, int t, int lane, int) {
  L.init(reinterpret_cast<const uint32_t*>(base), T, t, lane);
}
template <>
__device__ __forceinline__ void lane_init<4>(LaneQ<2>& L, const uint8_t* base, int T, int t, int lane, int) {
  L.init(reinterpret_cast<const uint32_t*>(base), T, t, lane);
}

template <int MT>
__device__ __forceinline__ void load_x(const float* xr, float (&v)[MT]) {
  if (MT % 4 == 0) {
#pragma unroll
    for (int j = 0; j < MT; j += 4) {
      const float4 f = *reinterpret_cast<const float4*>(xr + j);
      v[j] = f.x; v[j + 1] = f.y; v[j + 2] = f.z; v[j + 3] = f.w;
    }
  } else if (MT == 2) {
    const float2 f = *reinterpret_cast<const float2*>(xr);
    v[0] = f.x; v[MT - 1] = f.y;
  } else {
    v[0] = xr[0];
  }
}

// Side info on the device, per row (G = 1) or per group of K / G weights (GR, G = K / 128): rs[n G + g] scale,
// radd[n G + g] (the decoder's q = (r + radd) mod 2^B), rzc[n G + g] = 2^23 + z.
// Interleaved Huffman fused decode + GEMM for m <= MT tokens: Y[j, n] += sum_g rs[n, g] sum_(k in g) X[j, k] (q[n, k] - z_ng),
// Y zeroed. B = 8 (huf8): one weight per symbol, tile 64 x 256; B = 4 (huf4p): symbol = two nibbles (columns 2c,
// 2c + 1), tile 64 x 512 weights. Block = 8 warps on 8 tile rows of one tile column; x staged transposed (xs[k][j]).
template <int MT, int B, int LT, bool GR>
__global__ void __launch_bounds__(256) k_gemvm_hufi(const uint8_t* __restrict__ base, const float* __restrict__ X,
                                                    float* __restrict__ Y, int m, int N, int K,
                                                    const float* __restrict__ rs, const int* __restrict__ radd,
                                                    const float* __restrict__ rzc, int G,
                                                    const __half* __restrict__ gs, const uint8_t* __restrict__ gz,
                                                    int goff) {
  constexpr int TW = B == 8 ? TK : 2 * TK;   // tile width in weights
  __shared__ uint16_t lut[LUTN];
  __shared__ __align__(16) float xs[TW][MT];
  const int KT = K / TW, NT = N / TN, T = NT * KT;
  const int tc = blockIdx.x % KT, tr = (blockIdx.x / KT) * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  const uint16_t* glut = reinterpret_cast<const uint16_t*>(base);
  if (LT < 2)
    for (int i = threadIdx.x; i < LUTN; i += blockDim.x) lut[i] = glut[i];
  for (int i = threadIdx.x; i < TW * MT; i += blockDim.x) {
    const int j = i / TW, k = i - j * TW;
    xs[k][j] = j < m ? X[(size_t)j * K + tc * TW + k] : 0.f;
  }
  __syncthreads();
  if (tr >= NT) return;
  typename LaneOf<LT>::type L;
  lane_init<LT>(L, base, T, tr * KT + tc, lane, 2 * TK / 4);   // packed: 2 rows x TK bytes per lane
  const int SPG = GR ? (B == 8 ? K / G : K / G / 2) : TK;   // symbols per side-info group within the tile
  const int g0 = GR ? tc * TW / (K / G) : 0;
#pragma unroll 1
  for (int half = 0; half < 2; ++half) {
    const int n = tr * TN + half * 32 + lane;
    float tot[MT];
#pragma unroll
    for (int j = 0; j < MT; ++j) tot[j] = 0.f;
#pragma unroll 1
    for (int c0 = 0; c0 < TK; c0 += SPG) {
    const int gi = GR ? n * G + g0 + c0 / SPG : n;
    const uint32_t zq = GR ? (uint32_t)__ldg(gz + gi) : 0u;
    const uint32_t add = GR ? (goff < 0 ? 0u : zq + (uint32_t)goff) : (uint32_t)__ldg(radd + gi);
    const float zc = GR ? __uint_as_float(0x4B000000u | zq) : __ldg(rzc + gi);
    float a0[MT], a1[MT];
#pragma unroll
    for (int j = 0; j < MT; ++j) a0[j] = a1[j] = 0.f;
    if (B == 8) {
#pragma unroll 4
      for (int c = c0; c < c0 + SPG; c += 2) {
        const float w0 = __uint_as_float(0x4B000000u | ((L.next(lut) + add) & 255u)) - zc;
        const float w1 = __uint_as_float(0x4B000000u | ((L.next(lut) + add) & 255u)) - zc;
        float x0[MT], x1[MT];
        load_x<MT>(xs[c], x0);
        load_x<MT>(xs[c + 1], x1);
#pragma unroll
        for (int j = 0; j < MT; ++j) {
          a0[j] = fmaf(x0[j], w0, a0[j]);
          a1[j] = fmaf(x1[j], w1, a1[j]);
        }
      }
    } else {
#pragma unroll 4
      for (int c = c0; c < c0 + SPG; ++c) {
        const uint32_t e = L.next(lut);
        const float w0 = __uint_as_float(0x4B000000u | ((e + add) & 15u)) - zc;
        const float w1 = __uint_as_float(0x4B000000u | (((e >> 4) + add) & 15u)) - zc;
        float x0[MT], x1[MT];
        load_x<MT>(xs[2 * c], x0);
        load_x<MT>(xs[2 * c + 1], x1);
#pragma unroll
        for (int j = 0; j < MT; ++j) {
          a0[j] = fmaf(x0[j], w0, a0[j]);
          a1[j] = fmaf(x1[j], w1, a1[j]);
        }
      }
    }
    const float sc = GR ? ld_h(gs + gi) : __ldg(rs + gi);
#pragma unroll
    for (int j = 0; j < MT; ++j) tot[j] += sc * (a0[j] + a1[j]);
    }
#pragma unroll
    for (int j = 0; j < MT; ++j)
      if (j < m) atomicAdd(Y + (size_t)j * N + n, tot[j]);
  }
}

// interleaved Huffman decode for tile rows [tr0, tr1) to the codes q (prefill: raw8 GEMM with zero point 0 and
// scale 1, then k_rowaffine; q - z_n does not fit a byte for per-tensor 8-bit grids; G > 1: the GEMM applies the
// group side info). radd[n G + g]: group g = column / (K / G).
template <int B, int LT>
__global__ void k_decode_hufi(const uint8_t* __restrict__ base, uint8_t* __restrict__ out, int N, int K, int tr0,
                              int tr1, const int* __restrict__ radd, int G, const uint8_t* __restrict__ gz, int goff) {
  constexpr int TW = B == 8 ? TK : 2 * TK;
  __shared__ uint16_t lut[LUTN];
  const uint16_t* glut = reinterpret_cast<const uint16_t*>(base);
  if (LT < 2)
    for (int i = threadIdx.x; i < LUTN; i += blockDim.x) lut[i] = glut[i];
  __syncthreads();
  const int KT = K / TW, T = (N / TN) * KT;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int t = tr0 * KT + blockIdx.x * 8 + warp;
  if (t >= tr1 * KT) return;
  const int tr = t / KT, tc = t - tr * KT;
  typename LaneOf<LT>::type L;
  lane_init<LT>(L, base, T, t, lane, 2 * TK / 4);   // packed: 2 rows x TK bytes per lane
  for (int half = 0; half < 2; ++half) {
    const int n = tr * TN + half * 32 + lane;
    uint32_t add = G > 1 ? (goff < 0 ? 0u : (uint32_t)gz[(size_t)n * G] + goff) : (uint32_t)radd[n];
    uint32_t* row = reinterpret_cast<uint32_t*>(out + (size_t)((tr - tr0) * TN + half * 32 + lane) * K + tc * TW);
#pragma unroll 1
    for (int c = 0; c < TW / 16; ++c) {
      if (G > 1 && goff >= 0) add = (uint32_t)gz[(size_t)n * G + (tc * TW + c * 16) / (K / G)] + goff;
      uint32_t v[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        if (B == 8) {
          v[j] = 0;
#pragma unroll
          for (int i = 0; i < 4; ++i) v[j] |= ((L.next(lut) + add) & 255u) << (8 * i);
        } else {
          const uint32_t e0 = L.next(lut), e1 = L.next(lut);
          v[j] = ((e0 + add) & 15u) | (((e0 >> 4) + add) & 15u) << 8 | ((e1 + add) & 15u) << 16 |
                 (((e1 >> 4) + add) & 15u) << 24;
        }
      }
      reinterpret_cast<uint4*>(row)[c] = make_uint4(v[0], v[1], v[2], v[3]);
    }
  }
}

// the side bytes of one super-block as 32-bit words (byte i at bits 8 (i & 3) of w[i / 4]); Q6_K records (18 bytes)
// are only 2-byte aligned
template <int KQ>
__device__ __forceinline__ void kq_load(const uint8_t* p, uint32_t* w) {
  if (KQ == 3) {
    const uint16_t* h = reinterpret_cast<const uint16_t*>(p);
#pragma unroll
    for (int i = 0; i < 5; ++i) w[i] = __ldg(h + 2 * i) | (i < 4 ? (uint32_t)__ldg(h + 2 * i + 1) << 16 : 0u);
  } else {
#pragma unroll
    for (int i = 0; i < kq_sbytes(KQ) / 4; ++i) w[i] = __ldg(reinterpret_cast<const uint32_t*>(p) + i);
  }
}
__device__ __forceinline__ uint32_t kq_byte(const uint32_t* w, int i) { return (w[i >> 2] >> (8 * (i & 3))) & 255u; }
__device__ __forceinline__ float kq_hw(uint32_t v) { return __half2float(__ushort_as_half((unsigned short)v)); }

// kq_side on register words (j a compile-time constant after unrolling)
template <int KQ>
__device__ __forceinline__ void kq_side_r(const uint32_t* w, int j, float& s, float& o) {
  if (KQ == 1 || KQ == 2) {
    uint32_t sc, m;
    if (j < 4) {
      sc = kq_byte(w, 4 + j) & 63;
      m = kq_byte(w, 8 + j) & 63;
    } else {
      sc = (kq_byte(w, 8 + j) & 15) | ((kq_byte(w, j) >> 6) << 4);
      m = (kq_byte(w, 8 + j) >> 4) | ((kq_byte(w, 4 + j) >> 6) << 4);
    }
    s = kq_hw(w[0]) * (float)sc;
    o = kq_hw(w[0] >> 16) * (float)m;
  } else if (KQ == 3) {
    s = kq_hw(w[4]) * (float)(int8_t)kq_byte(w, j);
    o = 0.f;
  } else {
    const uint32_t ls = ((kq_byte(w, 4 + j / 2) >> (4 * (j & 1))) & 15) | (((w[0] >> (16 + 2 * j)) & 3) << 4);
    s = kq_hw(w[0]) * (float)((int)ls - 32);
    o = 0.f;
  }
}

// the next 4 symbols of a lane as one word (symbol i at bits 8 i); a packed lane returns its next word
template <class LN>
__device__ __forceinline__ uint32_t next4(LN& L, const uint16_t* lut) {
  uint32_t v = L.next(lut) & 255u;
  v |= (L.next(lut) & 255u) << 8;
  v |= (L.next(lut) & 255u) << 16;
  return v | (L.next(lut) & 255u) << 24;
}
__device__ __forceinline__ uint32_t next4(LaneR& L, const uint16_t*) { return L.word(); }
template <int HB>
__device__ __forceinline__ uint32_t next4(LaneQ<HB>& L, const uint16_t*) { return L.word(); }
// LaneQ::group where the sub-block matches one high word (C++14: no if constexpr for the Jetson Nano's CUDA 10.2)
template <bool G>
struct GroupFetch {
  template <class LN, int W>
  static __device__ __forceinline__ void f(LN&, uint32_t (&)[W]) {}
};
template <>
struct GroupFetch<true> {
  template <class LN, int W>
  static __device__ __forceinline__ void f(LN& L, uint32_t (&u)[W]) { L.template group<W>(u); }
};

// f(q) of byte I of u (K-quant 8-bit codes): one PRMT, no shift / mask (byte_f), or the IQ4_XS grid
template <int KQ, int I>
__device__ __forceinline__ float kq_fb(uint32_t u, const float* kv) {
  if (KQ == 4) return kv[(u >> (8 * I)) & 255u];
  return byte_f(u, 0x7440 + I) - (KQ == 3 ? 8388640.f : 8388608.f);
}

// f(q) of nibble I of u (4-bit K-quant codes, column I of the word): Q4_K a nibble mask and one PRMT; IQ4_XS the grid
// as int8 bytes (kq_kv8, defined below), biased to unsigned, one PRMT (exact: kv holds integers), or kv in smem
// (LLMRT_KV_LDS)
template <int KQ, int I>
__device__ __forceinline__ float kq_fn(uint32_t lo, uint32_t hi, int2 g, const float* kv) {
#ifdef LLMRT_KV_LDS
  if (KQ == 4) return kv[(I & 1 ? hi : lo) >> (8 * (I / 2)) & 15u];
#else
  if (KQ == 4) return byte_f((I < 4 ? g.x : g.y) ^ 0x80808080u, 0x7440 + (I & 3)) - 8388736.f;
#endif
  return byte_f(I & 1 ? hi : lo, 0x7440 + I / 2) - 8388608.f;
}
__device__ __forceinline__ int2 kq_kv8(uint32_t q);

// K-quant fused decode + GEMM for m <= MT tokens (layout and lanes as k_gemvm_hufi; codes not re-centered):
// Y[j, n] (row stride ldy) += sum_(sub-blocks g) s_ng sum_(k in g) X[j, k] f(q[n, k]) - o_ng sum_(k in g) X[j, k].
// The per-sub-block sums of x (Q4_K / Q5_K offsets) are formed once per block in smem. Partial sums per tile column
// -> P [K / TW, m, N], summed in order by k_sum_kt (as k_gemvm_kq8: no atomics, coded == packed twin bit for bit).
template <int MT, int B, int LT, int KQ>
__global__ void __launch_bounds__(256) k_gemvm_kq(const uint8_t* __restrict__ base, const float* __restrict__ X,
                                                  float* __restrict__ P, int m, int N, int K,
                                                  const uint8_t* __restrict__ side) {
  constexpr int TW = B == 8 ? TK : 2 * TK;
  constexpr int GW = kq_gw(KQ), NG = TW / GW, SPG = B == 8 ? GW : GW / 2, SB = kq_sbytes(KQ), SW = (SB + 3) / 4;
  __shared__ uint16_t lut[LUTN];
  __shared__ __align__(16) float xs[TW][MT];
  __shared__ float xg[NG][MT];
  __shared__ float kv[16];
  const int KT = K / TW, NT = N / TN, T = NT * KT;
  const int tc = blockIdx.x % KT, tr = (blockIdx.x / KT) * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  const uint16_t* glut = reinterpret_cast<const uint16_t*>(base);
  if (LT < 2)
    for (int i = threadIdx.x; i < LUTN; i += blockDim.x) lut[i] = glut[i];
  if (threadIdx.x < 16) kv[threadIdx.x] = c_kv[threadIdx.x];
  for (int i = threadIdx.x; i < TW * MT; i += blockDim.x) {
    const int j = i / TW, k = i - j * TW;
    xs[k][j] = j < m ? X[(size_t)j * K + tc * TW + k] : 0.f;
  }
  __syncthreads();
  if (KQ <= 2) {
    for (int i = threadIdx.x; i < NG * MT; i += blockDim.x) {
      const int g = i / MT, j = i - g * MT;
      float a = 0.f;
      for (int k = g * GW; k < (g + 1) * GW; ++k) a += xs[k][j];
      xg[g][j] = a;
    }
    __syncthreads();
  }
  if (tr >= NT) return;
  typename LaneOf<LT>::type L;
  lane_init<LT>(L, base, T, tr * KT + tc, lane, 2 * TK / 4);
#pragma unroll 1
  for (int half = 0; half < 2; ++half) {
    const int n = tr * TN + half * 32 + lane;
    const uint8_t* srow = side + ((size_t)n * (K / 256) + tc * TW / 256) * SB;
    float tot[MT];
#pragma unroll
    for (int j = 0; j < MT; ++j) tot[j] = 0.f;
#pragma unroll 1
    for (int sb = 0; sb < TW / 256; ++sb) {   // super-blocks of this tile row: side bytes once, into registers
      uint32_t sw[SW];
      kq_load<KQ>(srow + sb * SB, sw);
#pragma unroll
      for (int jj = 0; jj < 256 / GW; ++jj) {   // sub-blocks (unrolled: scale fields at constant positions)
        const int g = sb * (256 / GW) + jj;
        float s, o;
        kq_side_r<KQ>(sw, jj, s, o);
        float a0[MT], a1[MT];
#pragma unroll
        for (int j = 0; j < MT; ++j) a0[j] = a1[j] = 0.f;
        if (B == 8 && LT >= 2) {   // packed: 4 codes per word (LaneQ: the sub-block in one fetch), same FMA order
          constexpr int WPG = GW / 4;
          constexpr bool GRP = LT >= 3 && 4 * WPG * (LT - 2) == 32;
          uint32_t ug[WPG];
          GroupFetch<GRP>::f(L, ug);
#pragma unroll
          for (int w = 0; w < WPG; ++w) {
            const uint32_t u = GRP ? ug[w] : next4(L, lut);
            const int c = g * SPG + 4 * w;
            const float w0 = kq_fb<KQ, 0>(u, kv), w1 = kq_fb<KQ, 1>(u, kv), w2 = kq_fb<KQ, 2>(u, kv),
                        w3 = kq_fb<KQ, 3>(u, kv);
            float x0[MT], x1[MT], x2[MT], x3[MT];
            load_x<MT>(xs[c], x0);
            load_x<MT>(xs[c + 1], x1);
            load_x<MT>(xs[c + 2], x2);
            load_x<MT>(xs[c + 3], x3);
#pragma unroll
            for (int j = 0; j < MT; ++j) {
              a0[j] = fmaf(x0[j], w0, a0[j]);
              a1[j] = fmaf(x1[j], w1, a1[j]);
              a0[j] = fmaf(x2[j], w2, a0[j]);
              a1[j] = fmaf(x3[j], w3, a1[j]);
            }
          }
        } else if (B == 8) {   // Huffman lanes: a symbol at a time (next4 costs registers at MT = 8)
#pragma unroll 4
          for (int c = g * SPG; c < (g + 1) * SPG; c += 2) {
            const float w0 = kq_f<KQ>(L.next(lut) & 255u, kv);
            const float w1 = kq_f<KQ>(L.next(lut) & 255u, kv);
            float x0[MT], x1[MT];
            load_x<MT>(xs[c], x0);
            load_x<MT>(xs[c + 1], x1);
#pragma unroll
            for (int j = 0; j < MT; ++j) {
              a0[j] = fmaf(x0[j], w0, a0[j]);
              a1[j] = fmaf(x1[j], w1, a1[j]);
            }
          }
        } else if (LT >= 2) {   // packed 4-bit: 8 codes per word (byte b: columns 2b, 2b + 1), same FMA order
#pragma unroll
          for (int w = 0; w < SPG / 4; ++w) {
            const uint32_t u = next4(L, lut), lo = u & 0x0F0F0F0Fu, hi = (u >> 4) & 0x0F0F0F0Fu;
            const int2 gv = KQ == 4 ? kq_kv8(u) : make_int2(0, 0);
            const int c = 2 * (g * SPG + 4 * w);
            float f[8];
            f[0] = kq_fn<KQ, 0>(lo, hi, gv, kv); f[1] = kq_fn<KQ, 1>(lo, hi, gv, kv);
            f[2] = kq_fn<KQ, 2>(lo, hi, gv, kv); f[3] = kq_fn<KQ, 3>(lo, hi, gv, kv);
            f[4] = kq_fn<KQ, 4>(lo, hi, gv, kv); f[5] = kq_fn<KQ, 5>(lo, hi, gv, kv);
            f[6] = kq_fn<KQ, 6>(lo, hi, gv, kv); f[7] = kq_fn<KQ, 7>(lo, hi, gv, kv);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              float x0[MT], x1[MT];
              load_x<MT>(xs[c + 2 * i], x0);
              load_x<MT>(xs[c + 2 * i + 1], x1);
#pragma unroll
              for (int j = 0; j < MT; ++j) {
                a0[j] = fmaf(x0[j], f[2 * i], a0[j]);
                a1[j] = fmaf(x1[j], f[2 * i + 1], a1[j]);
              }
            }
          }
        } else {
#pragma unroll 4
          for (int c = g * SPG; c < (g + 1) * SPG; ++c) {
            const uint32_t e = L.next(lut);
            const float w0 = kq_f<KQ>(e & 15u, kv);
            const float w1 = kq_f<KQ>((e >> 4) & 15u, kv);
            float x0[MT], x1[MT];
            load_x<MT>(xs[2 * c], x0);
            load_x<MT>(xs[2 * c + 1], x1);
#pragma unroll
            for (int j = 0; j < MT; ++j) {
              a0[j] = fmaf(x0[j], w0, a0[j]);
              a1[j] = fmaf(x1[j], w1, a1[j]);
            }
          }
        }
#pragma unroll
        for (int j = 0; j < MT; ++j) tot[j] += KQ <= 2 ? s * (a0[j] + a1[j]) - o * xg[g][j] : s * (a0[j] + a1[j]);
      }
    }
#pragma unroll
    for (int j = 0; j < MT; ++j)
      if (j < m) P[((size_t)tc * m + j) * N + n] = tot[j];
  }
}

// int8 activations (sm_61+, llama.cpp q8_1 style): X [m, K] -> xq int8 per block of 32 (d = amax / 127),
// dx[m, K / 32] = d, sx[m, K / 32] = d sum q (the Q4_K / Q5_K offset term). One warp per block.
__global__ void k_quant_x8(const float* __restrict__ X, int8_t* __restrict__ xq, float* __restrict__ dx,
                           float* __restrict__ sx, int nb) {
  const int w = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (w >= nb) return;
  const float v = X[(size_t)w * 32 + lane];
  float a = fabsf(v);
#pragma unroll
  for (int o = 16; o; o >>= 1) a = fmaxf(a, __shfl_xor_sync(~0u, a, o));
  const float d = a / 127.f;
  const int q = d > 0.f ? __float2int_rn(v / d) : 0;
  int s = q;
#pragma unroll
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  xq[(size_t)w * 32 + lane] = (int8_t)q;
  if (lane == 0) {
    dx[w] = d;
    sx[w] = d * (float)s;
  }
}

// 8 nibbles (columns 0..7) -> int8 IQ4_XS grid values: .x columns 0..3, .y columns 4..7 (llama.cpp
// get_int_from_table_16 without its low / high nibble split; the grid c_kv as int8 in 4 words)
__device__ __forceinline__ int2 kq_kv8(uint32_t q) {
  constexpr uint32_t t0 = 0xBFAD9881u, t1 = 0xF6EADDCFu, t2 = 0x26190D01u, t3 = 0x71594535u;
  const uint32_t sel = 0x32103210u | ((q & 0x88888888u) >> 1);
  const uint32_t lo0 = __byte_perm(t0, t1, q), hi0 = __byte_perm(t2, t3, q);
  const uint32_t lo1 = __byte_perm(t0, t1, q >> 16), hi1 = __byte_perm(t2, t3, q >> 16);
  return make_int2((int)__byte_perm(lo0, hi0, sel), (int)__byte_perm(lo1, hi1, sel >> 16));
}

// k_gemvm_kq with int8 activations (k_quant_x8): per sub-block an integer dot product of the codes (as int8:
// q, q - 32, kv[q]) with xq via __dp4a, then Y[j, n] += s_ng dx_jb acc - o_ng sx_jb (b = the 32-block of g).
// Huffman-coded and packed lanes give the same integers. Partial sums per tile column -> P [K / TW, m, N].
// int -> float for |x| < 2^22 without I2F (quarter rate on sm_8x): x + 1.5 * 2^23 in the float's mantissa, exact, so
// the same value as (float)x. A group's dp4a sum is at most 32 x 127 x 128 < 2^22 in magnitude.
__device__ __forceinline__ float i2f22(int x) { return __int_as_float(x + 0x4B400000) - 12582912.f; }

template <int MT, int B, int LT, int KQ>
__global__ void __launch_bounds__(256) k_gemvm_kq8(const uint8_t* __restrict__ base, const int8_t* __restrict__ XQ,
                                                   const float* __restrict__ DX, const float* __restrict__ SX,
                                                   float* __restrict__ P, int m, int N, int K,
                                                   const uint8_t* __restrict__ side) {
#if __CUDA_ARCH__ >= 610
  constexpr int TW = B == 8 ? TK : 2 * TK;
  constexpr int GW = kq_gw(KQ), WPG = B == 8 ? GW / 4 : GW / 8, SB = kq_sbytes(KQ), SW = (SB + 3) / 4;
  __shared__ uint16_t lut[LUTN];
  __shared__ int xs[TW / 4][MT];
  __shared__ float xd[TW / 32][MT], xm[TW / 32][MT];
  const int KT = K / TW, NT = N / TN, T = NT * KT;
  const int tc = blockIdx.x % KT, tr = (blockIdx.x / KT) * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  const uint16_t* glut = reinterpret_cast<const uint16_t*>(base);
  if (LT < 2)
    for (int i = threadIdx.x; i < LUTN; i += blockDim.x) lut[i] = glut[i];
  for (int i = threadIdx.x; i < TW / 4 * MT; i += blockDim.x) {
    const int j = i / (TW / 4), c = i - j * (TW / 4);
    xs[c][j] = j < m ? reinterpret_cast<const int*>(XQ)[((size_t)j * K + tc * TW) / 4 + c] : 0;
  }
  for (int i = threadIdx.x; i < TW / 32 * MT; i += blockDim.x) {
    const int j = i / (TW / 32), g = i - j * (TW / 32);
    xd[g][j] = j < m ? DX[(size_t)j * (K / 32) + tc * TW / 32 + g] : 0.f;
    xm[g][j] = j < m ? SX[(size_t)j * (K / 32) + tc * TW / 32 + g] : 0.f;
  }
  __syncthreads();
  if (tr >= NT) return;
  typename LaneOf<LT>::type L;
  lane_init<LT>(L, base, T, tr * KT + tc, lane, 2 * TK / 4);
#pragma unroll 1
  for (int half = 0; half < 2; ++half) {
    const int n = tr * TN + half * 32 + lane;
    const uint8_t* srow = side + ((size_t)n * (K / 256) + tc * TW / 256) * SB;
    float tot[MT];
#pragma unroll
    for (int j = 0; j < MT; ++j) tot[j] = 0.f;
#pragma unroll 1
    for (int sb = 0; sb < TW / 256; ++sb) {
      uint32_t sw[SW];
      kq_load<KQ>(srow + sb * SB, sw);
      // packed Q6_K: groups in runs of 4 (one scale word each), so the compiler does not hoist all 64 word loads of
      // the super-block into registers (173 registers at MT = 8: one block per SM); same order of operations
      constexpr int NJ = 256 / GW, JU = (KQ == 3 && LT >= 2) ? 4 : NJ;
#pragma unroll 1
      for (int j0 = 0; j0 < NJ; j0 += JU) {
#pragma unroll
      for (int ji = 0; ji < JU; ++ji) {
        const int jj = j0 + ji;
        const int g = sb * (256 / GW) + jj, xb = g * GW / 32;
        float s, o;
        if (JU == NJ) {
          kq_side_r<KQ>(sw, jj, s, o);
        } else {   // kq_side_r<3> with the scale word selected without dynamic register indexing
          const uint32_t wq = j0 == 0 ? sw[0] : j0 == 4 ? sw[1] : j0 == 8 ? sw[2] : sw[3];
          s = kq_hw(sw[SW - 1]) * (float)(int8_t)((wq >> (8 * ji)) & 255u);
          o = 0.f;
        }
        int acc[MT];
#pragma unroll
        for (int j = 0; j < MT; ++j) acc[j] = 0;
        constexpr bool GRP = B == 8 && LT >= 3 && 4 * WPG * (LT - 2) == 32;   // LaneQ: one sub-block at a time
        uint32_t ug[WPG];
        GroupFetch<GRP>::f(L, ug);
#pragma unroll
        for (int w = 0; w < WPG; ++w) {
          const uint32_t u = GRP ? ug[w] : next4(L, lut);
          if (B == 8) {
            const int v = KQ == 3 ? (int)__vsub4(u, 0x20202020u) : (int)u;
            const int c = (g * GW + 4 * w) / 4;
#pragma unroll
            for (int j = 0; j < MT; ++j) acc[j] = __dp4a(v, xs[c][j], acc[j]);
          } else {
            int2 v;
            if (KQ == 4) {
              v = kq_kv8(u);
            } else {   // Q4_K: nibbles to bytes in column order
              const uint32_t lo = u & 0x0F0F0F0Fu, hi = (u >> 4) & 0x0F0F0F0Fu;
              v = make_int2((int)__byte_perm(lo, hi, 0x5140), (int)__byte_perm(lo, hi, 0x7362));
            }
            const int c = (g * GW + 8 * w) / 4;
#pragma unroll
            for (int j = 0; j < MT; ++j) acc[j] = __dp4a(v.y, xs[c + 1][j], __dp4a(v.x, xs[c][j], acc[j]));
          }
        }
#pragma unroll
        for (int j = 0; j < MT; ++j)
          tot[j] += KQ <= 2 ? s * xd[xb][j] * i2f22(acc[j]) - o * xm[xb][j] : s * xd[xb][j] * i2f22(acc[j]);
      }
      }
    }
#pragma unroll
    for (int j = 0; j < MT; ++j)
      if (j < m) P[((size_t)tc * m + j) * N + n] = tot[j];
  }
#endif
}

// Y[j, n] (row stride ldy) = sum over tile columns tc in order of P[tc, j, n]: deterministic (no atomics), so the
// Huffman-coded and the packed bundle give the same logits bit for bit
__global__ void k_sum_kt(const float* __restrict__ P, float* __restrict__ Y, int m, int N, int ldy, int KT) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m * N) return;
  const int j = i / N, n = i - j * N;
  float a = 0.f;
  for (int tc = 0; tc < KT; ++tc) a += P[((size_t)tc * m + j) * N + n];
  Y[(size_t)j * ldy + n] = a;
}

// sx[j] = sum_k X[j, k] (one block per row)
__global__ void k_rowsum(const float* __restrict__ X, float* __restrict__ sx, int K) {
  __shared__ float red[256];
  float a = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x) a += X[(size_t)blockIdx.x * K + k];
  red[threadIdx.x] = a;
  __syncthreads();
  for (int o = 128; o; o >>= 1) {
    if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
    __syncthreads();
  }
  if (threadIdx.x == 0) sx[blockIdx.x] = red[0];
}

// Y[j, n] = rs[n] (Y[j, n] - z_n sx[j]), z_n = rzc[n] - 2^23
__global__ void k_rowaffine(float* __restrict__ Y, int m, int N, int ldy, const float* __restrict__ rs,
                            const float* __restrict__ rzc, const float* __restrict__ sx) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < (size_t)m * N; i += (size_t)gridDim.x * blockDim.x) {
    const size_t j = i / N, n = i % N;
    float* y = Y + j * ldy + n;
    *y = rs[n] * (*y - (rzc[n] - 8388608.f) * sx[j]);
  }
}

// host: huf8 blob -> interleaved layout (IlLayout): padded (every lane to its tile's longest, read by LaneP) or
// compact (read by LaneC)
static std::vector<uint32_t> interleave_huf8(const uint8_t* blob, size_t len, int N, int K, bool compact) {
  const int T = (N / TN) * (K / TK);
  const uint32_t* starts = reinterpret_cast<const uint32_t*>(blob + 2 * LUTN);
  const uint32_t* pay = starts + (size_t)T * 32;
  const uint64_t pw = (len - 2 * LUTN - (size_t)T * 32 * 4) / 4;
  const size_t hdr = LUTN / 2 + T + ((T + 1) & ~1) / 2 + (size_t)T * 16;   // u32 words before the payload
  std::vector<uint32_t> out(hdr);
  memcpy(out.data(), blob, 2 * LUTN);
  uint16_t* wc = reinterpret_cast<uint16_t*>(out.data() + LUTN / 2 + T);
  for (int t = 0; t < T; ++t) {
    uint64_t st[33];
    for (int l = 0; l < 32; ++l) st[l] = starts[(size_t)t * 32 + l];
    st[32] = t + 1 < T ? starts[(size_t)(t + 1) * 32] : std::min<uint64_t>(st[31] + 512 * LMAX, (pw - 2) * 32);
    uint32_t w[32], W = ~0u, Wmax = 0, ov = 0;
    for (int l = 0; l < 32; ++l) {
      w[l] = (uint32_t)((st[l + 1] - st[l] + 31) / 32);
      W = std::min(W, w[l]);
      Wmax = std::max(Wmax, w[l]);
    }
    if (!compact) W = Wmax;
    if (W > 65535) { fprintf(stderr, "interleave: lane of %u words\n", W); exit(1); }
    wc = reinterpret_cast<uint16_t*>(out.data() + LUTN / 2 + T);   // out may have moved
    uint16_t* ovf = wc + ((T + 1) & ~1) + (size_t)t * 32;
    wc[t] = (uint16_t)W;
    for (int l = 0; l < 32; ++l) {
      if (ov > 65535) { fprintf(stderr, "interleave: overflow offset %u\n", ov); exit(1); }
      ovf[l] = (uint16_t)ov;
      ov += w[l] > W ? w[l] - W : 0;
    }
    const size_t b0 = out.size() - hdr;
    if (b0 > 0xffffffffull) { fprintf(stderr, "interleave: matrix too large\n"); exit(1); }
    out[LUTN / 2 + t] = (uint32_t)b0;
    out.resize(out.size() + (size_t)W * 32 + ov, 0);
    uint32_t* o = out.data() + hdr + b0;
    ovf = reinterpret_cast<uint16_t*>(out.data() + LUTN / 2 + T) + ((T + 1) & ~1) + (size_t)t * 32;
    for (int l = 0; l < 32; ++l) {
      const uint64_t nb = st[l + 1] - st[l];
      for (uint32_t i = 0; i * 32ull < nb; ++i) {
        const uint64_t b = st[l] + i * 32ull;
        const uint64_t x = pay[b >> 5] | ((uint64_t)pay[(b >> 5) + 1] << 32);
        uint32_t v = (uint32_t)(x >> (b & 31));
        const uint64_t left = nb - i * 32ull;
        if (left < 32) v &= (1u << left) - 1;
        o[i < W ? (size_t)i * 32 + l : (size_t)W * 32 + ovf[l] + (i - W)] = v;
      }
    }
  }
  out.resize(out.size() + 96, 0);   // tail: the last lanes' read-ahead
  return out;
}

// embedding gather: out[m, :] = fp16((E[id, :] - z) * s), s / z per row or per group of D / G (rs / rz [V, G]) or
// per tensor; gathered: E holds the m rows already (host-side embedding)
__global__ void k_embed(const uint8_t* __restrict__ E, const int* __restrict__ ids, float* __restrict__ out, int D,
                        float s, float z, const float* __restrict__ rs, const float* __restrict__ rz, int gathered,
                        int G) {
  const int m = blockIdx.x;
  const uint8_t* row = E + (size_t)(gathered ? m : ids[m]) * D;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    if (rs) {
      const size_t g = (size_t)ids[m] * G + i / (D / G);
      s = rs[g];
      z = rz[g];
    }
    out[(size_t)m * D + i] = __half2float(__float2half_rn(((float)row[i] - z) * s));
  }
}

__device__ __forceinline__ float block_sum(float v, float* red) {
  for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  __syncthreads();
  if (lane == 0) red[warp] = v;
  __syncthreads();
  float t = 0.f;
  for (int i = 0; i < (int)(blockDim.x >> 5); ++i) t += red[i];
  return t;
}

// (Skip)SimplifiedLayerNormalization: if d != null, h += d (residual); out = h / sqrt(mean(h^2) + eps) * g
__global__ void k_rmsnorm(float* __restrict__ h, const float* __restrict__ d, const float* __restrict__ g,
                          float* __restrict__ out, int D, float eps) {
  __shared__ float red[32];
  float* hr = h + (size_t)blockIdx.x * D;
  const float* dr = d ? d + (size_t)blockIdx.x * D : nullptr;
  float ss = 0.f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float v = hr[i];
    if (dr) { v += dr[i]; hr[i] = v; }
    ss += v * v;
  }
  const float r = rsqrtf(block_sum(ss, red) / D + eps);
  for (int i = threadIdx.x; i < D; i += blockDim.x) out[(size_t)blockIdx.x * D + i] = hr[i] * r * g[i];
}

// rotary (non-interleaved, rotate-half) on q (H heads) and k (Hk heads) in place, then k, v -> fp16 KV cache at
// position pos0 + m. Row layout of qkv: [q H x Dh | k Hk x Dh | v Hk x Dh].
__global__ void k_rope_cache(float* __restrict__ qkv, const float* __restrict__ cs, const float* __restrict__ sn,
                             __half* __restrict__ kc, __half* __restrict__ vc, const int* __restrict__ posp, int H,
                             int Hk, int Dh, int Tmax) {
  const int m = blockIdx.x, pos = *posp + m, half = Dh / 2;   // posp: device copy of the position (CUDA graphs)
  float* q = qkv + (size_t)m * (H + 2 * Hk) * Dh;
  float* k = q + H * Dh;
  const float* v = k + Hk * Dh;
  for (int i = threadIdx.x; i < H * half; i += blockDim.x) {
    const int h = i / half, j = i - h * half;
    const float c = cs[pos * half + j], s = sn[pos * half + j];
    float* qh = q + h * Dh;
    const float q1 = qh[j], q2 = qh[j + half];
    qh[j] = q1 * c - q2 * s;
    qh[j + half] = q2 * c + q1 * s;
  }
  for (int i = threadIdx.x; i < Hk * half; i += blockDim.x) {
    const int h = i / half, j = i - h * half;
    const float c = cs[pos * half + j], s = sn[pos * half + j];
    const float* kh = k + h * Dh;
    const float k1 = kh[j], k2 = kh[j + half];
    kc[((size_t)h * Tmax + pos) * Dh + j] = __float2half_rn(k1 * c - k2 * s);
    kc[((size_t)h * Tmax + pos) * Dh + j + half] = __float2half_rn(k2 * c + k1 * s);
  }
  for (int i = threadIdx.x; i < Hk * Dh; i += blockDim.x) {
    const int h = i / Dh, j = i - h * Dh;
    vc[((size_t)h * Tmax + pos) * Dh + j] = __float2half_rn(v[i]);
  }
}

// causal attention for query m (absolute position pos0 + m) and query head h (KV head h / (H / Hk)); DH / 32 values
// per lane; block of 128 threads (>= DH). window > 0: sliding window over the last `window` keys.
template <int DH>
__global__ void k_attn(const float* __restrict__ qkv, const __half* __restrict__ kc, const __half* __restrict__ vc,
                       float* __restrict__ out, int pos0, int H, int Hk, int Tmax, float scale, int window) {
  extern __shared__ float sc[];
  __shared__ float red[32];
  __shared__ float qs[DH];
  constexpr int R = DH / 32;
  const int m = blockIdx.x, h = blockIdx.y, kh = h / (H / Hk), T = pos0 + m + 1;
  const int t0 = window > 0 ? max(0, T - window) : 0;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
  if (threadIdx.x < DH) qs[threadIdx.x] = qkv[(size_t)m * (H + 2 * Hk) * DH + h * DH + threadIdx.x];
  __syncthreads();
  const __half* K = kc + (size_t)kh * Tmax * DH;
  const __half* V = vc + (size_t)kh * Tmax * DH;
  float mx = -INFINITY;
  for (int t = t0 + warp; t < T; t += nw) {
    const __half* kt = K + (size_t)t * DH;
    float a = 0.f;
#pragma unroll
    for (int r = 0; r < R; ++r) a += qs[lane + 32 * r] * __half2float(kt[lane + 32 * r]);
    for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
    a *= scale;
    if (lane == 0) sc[t] = a;
    mx = fmaxf(mx, a);
  }
  if (lane == 0) red[warp] = mx;
  __syncthreads();
  mx = -INFINITY;
  for (int i = 0; i < nw; ++i) mx = fmaxf(mx, red[i]);
  __syncthreads();
  float sum = 0.f;
  for (int t = t0 + threadIdx.x; t < T; t += blockDim.x) {
    const float e = __expf(sc[t] - mx);
    sc[t] = e;
    sum += e;
  }
  sum = block_sum(sum, red);
  __syncthreads();
  if (threadIdx.x < DH) {
    float a = 0.f;
    for (int t = t0; t < T; ++t) a = fmaf(sc[t], __half2float(V[(size_t)t * DH + threadIdx.x]), a);
    out[(size_t)m * H * DH + h * DH + threadIdx.x] = a / sum;
  }
}

// Multi-query causal attention for m <= MQ query tokens (verify passes and plain decode): one block per query head
// reads every key / value row once for all m queries (k_attn reads them once per query). Query i sits at absolute
// position pos0 + i and sees keys [t0_i, pos0 + i]. Each query's arithmetic (dot products, max, sum, value
// accumulation order) is the same for every m (keys one per lane, q . k summed over d in order), so a verify pass gives the same logits as m = 1 passes. 256 threads:
// keys one per lane over 8 warps, then values as DH-wide rows over 256 / DH key slices. smem: scores sc[m][pos0 + m]
// + slice partials (sized per call: a large score buffer at m = 1 would take L1 from the K reads).
template <int DH, int MQ>
__global__ void __launch_bounds__(256) k_attn_mq(const float* __restrict__ qkv, const __half* __restrict__ kc,
                                                 const __half* __restrict__ vc, float* __restrict__ out, int m,
                                                 const int* __restrict__ posp, int H, int Hk, int Tmax, float scale,
                                                 int window) {
  extern __shared__ float sm[];
  const int pos0 = *posp;   // device copy of the position, so that a CUDA graph replays at any position
  constexpr int NW = 8, S = 256 / DH;
  const int TS = pos0 + m;                       // score row stride: keys up to the last query
  float* sc = sm;                                // [m][TS]
  float* part = sm + (size_t)m * TS;             // [S][MQ][DH]
  __shared__ float qs[MQ][DH];
  __shared__ float red[NW][MQ];
  __shared__ float mxs[MQ], sums[MQ];
  const int h = blockIdx.x, kh = h / (H / Hk), warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int i = threadIdx.x; i < MQ * DH; i += 256)
    qs[i / DH][i % DH] = i / DH < m ? qkv[(size_t)(i / DH) * (H + 2 * Hk) * DH + h * DH + i % DH] : 0.f;
  __syncthreads();
  const __half* K = kc + (size_t)kh * Tmax * DH;
  const __half* V = vc + (size_t)kh * Tmax * DH;
  auto lo = [&](int i) { return window > 0 ? max(0, pos0 + i + 1 - window) : 0; };
  const int tlo = lo(0), thi = pos0 + m;   // union of the key ranges
  float mx[MQ];
#pragma unroll
  for (int i = 0; i < MQ; ++i) mx[i] = -INFINITY;
  for (int t = tlo + warp * 32 + lane; t - lane < thi; t += NW * 32) {   // one key per lane: q . k over d in order
    float a[MQ];
#pragma unroll
    for (int i = 0; i < MQ; ++i) a[i] = 0.f;
    if (t < thi) {
      const uint4* kt = (const uint4*)(K + (size_t)t * DH);
#pragma unroll 2
      for (int c = 0; c < DH / 8; ++c) {
        const uint4 u = kt[c];
        const __half2* k2 = (const __half2*)&u;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 kf = __half22float2(k2[j]);
#pragma unroll
          for (int i = 0; i < MQ; ++i) {
            a[i] = fmaf(qs[i][8 * c + 2 * j], kf.x, a[i]);
            a[i] = fmaf(qs[i][8 * c + 2 * j + 1], kf.y, a[i]);
          }
        }
      }
#pragma unroll
      for (int i = 0; i < MQ; ++i)
        if (i < m && t >= lo(i) && t <= pos0 + i) {
          sc[(size_t)i * TS + t] = a[i] * scale;
          mx[i] = fmaxf(mx[i], a[i] * scale);
        }
    }
  }
#pragma unroll
  for (int i = 0; i < MQ; ++i)
    for (int o = 16; o; o >>= 1) mx[i] = fmaxf(mx[i], __shfl_xor_sync(0xffffffffu, mx[i], o));
  if (lane == 0)
#pragma unroll
    for (int i = 0; i < MQ; ++i) red[warp][i] = mx[i];
  __syncthreads();
  if (threadIdx.x < MQ) {
    float v = -INFINITY;
    for (int w = 0; w < NW; ++w) v = fmaxf(v, red[w][threadIdx.x]);
    mxs[threadIdx.x] = v;
  }
  __syncthreads();
  float ps[MQ];
#pragma unroll
  for (int i = 0; i < MQ; ++i) {
    ps[i] = 0.f;
    if (i >= m) continue;
    for (int t = lo(i) + threadIdx.x; t <= pos0 + i; t += 256) {
      const float e = __expf(sc[(size_t)i * TS + t] - mxs[i]);
      sc[(size_t)i * TS + t] = e;
      ps[i] += e;
    }
    for (int o = 16; o; o >>= 1) ps[i] += __shfl_xor_sync(0xffffffffu, ps[i], o);
  }
  __syncthreads();
  if (lane == 0)
#pragma unroll
    for (int i = 0; i < MQ; ++i) red[warp][i] = ps[i];
  __syncthreads();
  if (threadIdx.x < MQ) {
    float v = 0.f;
    for (int w = 0; w < NW; ++w) v += red[w][threadIdx.x];
    sums[threadIdx.x] = v;
  }
  const int d = threadIdx.x % DH, sl = threadIdx.x / DH;
  float acc[MQ];
#pragma unroll
  for (int i = 0; i < MQ; ++i) acc[i] = 0.f;
  if (sl < S) {
#pragma unroll 4
    for (int t = tlo + sl; t < thi; t += S) {
      const float v = __half2float(V[(size_t)t * DH + d]);
#pragma unroll
      for (int i = 0; i < MQ; ++i)
        if (i < m && t >= lo(i) && t <= pos0 + i) acc[i] = fmaf(sc[(size_t)i * TS + t], v, acc[i]);
    }
#pragma unroll
    for (int i = 0; i < MQ; ++i) part[((size_t)sl * MQ + i) * DH + d] = acc[i];
  }
  __syncthreads();
  for (int x = threadIdx.x; x < m * DH; x += 256) {
    const int i = x / DH, dd = x % DH;
    float a = 0.f;
    for (int q = 0; q < S; ++q) a += part[((size_t)q * MQ + i) * DH + dd];
    out[(size_t)i * H * DH + h * DH + dd] = a / sums[i];
  }
}

// MLP activation: g = g * sigmoid(g) * u
__global__ void k_silu_mul(float* __restrict__ g, const float* __restrict__ u, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const float a = g[i];
    g[i] = a * (1.f / (1.f + __expf(-a))) * u[i];
  }
}

// ── model ───────────────────────────────────────────────────────────────────────────────────────────────

struct Tensor {
  std::string name, enc;
  int N = 0, K = 0, zp = 0;
  float scale = 1.f;
  size_t off = 0, nbytes = 0;
  uint8_t* dev = nullptr;   // resident copy
  int stream_idx = -1;      // index in the streamed list, -1 if resident
  float zadj = 0.f;          // zero-point offset (PHI3SPEC_ROUND_BITS drafter)
  bool il = false;          // huf8 re-laid out as interleaved lane words (dev = IlLayout)
  bool pre_il = false;      // stored in the bundle already interleaved (enc suffix _il / _ilc, `export-il`)
  bool compact = false;     // compact IlLayout (LaneC) instead of padded (LaneP)
  int lt = 0;               // lane reader: 0 LaneP, 1 LaneC (Huffman), 2 LaneR (pk4 / pk8: packed, no entropy coding),
                            // 3 / 4 LaneQ (pk8 Q5_K / Q6_K repacked to 5 / 6 bits)
  int bits = 8;             // 4: huf4p (two weights per symbol)
  int G = 1;                // side-info groups per row (g128: K / 128); G > 1: gs / gz [N, G] instead of rs / radd / rzc
  __half* gs = nullptr;     // device group scales (fp16) and code zero points; decoder add = gz + goff
  uint8_t* gz = nullptr;
  int goff = 0;
  float* rs = nullptr;      // device per-row scale, decoder add, 2^23 + zero point (Huffman matrices; embed: rs, rz)
  int* radd = nullptr;
  float* rzc = nullptr;
  float* rz = nullptr;
  std::vector<int> h_radd, h_z;   // host copies (check)
  uint8_t* kqs = nullptr;   // K-quant side bytes [N, K / 256, SB] on the device (bundle tensor <name>.kqs)
  int kqt = 0;              // 0: none, 1 Q4_K, 2 Q5_K, 3 Q6_K, 4 IQ4_XS
};

struct Model {
  std::map<std::string, std::string> cfg;
  std::vector<Tensor> ts;
  std::map<std::string, int> by_name;
  int fd = -1;
  int L, D, H, Hk, Dh, I, V, Tmax, max_batch;
  std::vector<int> eos;
  float eps, attn_scale;
  // activations
  float *h, *xn, *qkv, *att, *o, *g, *u, *logits;
  int* ids;
  int* posd;   // device copy of pos (read by k_rope_cache / k_attn_mq; a CUDA graph replays at any position)
  int* ids_host = nullptr;   // pinned staging: ids [max_batch] then pos
  // CUDA graphs (LLMRT_GRAPHS=1, build with --default-stream per-thread): one graph per decode / verify pass shape
  // (m <= 8 tokens, all_logits, attention score buffer bucket graph_T), captured the second time a shape occurs
  std::map<long long, cudaGraphExec_t> graphs;
  std::set<long long> graph_seen;
  int graph_T = 0;   // capturing: attention smem sized for graph_T keys (>= pos + m) instead of pos + m
  int n_graph_launch = 0, n_graph_capture = 0;
  __half *kc, *vc;
  uint8_t* scratch = nullptr;
  float* sx = nullptr;      // [max_batch] row sums of a matmul input (prefill epilogue)
  size_t scratch_bytes = 0;
  // int8 activations for the K-quant decode GEMV (sm_61+; LLMRT_FP_ACT=1: fp32 path): X quantized per 32
  bool act8 = false;
  int8_t* xq = nullptr;
  float *xqd = nullptr, *xqs = nullptr, *xqp = nullptr;   // xqp: GEMV partials [K / TW, m, N] (both kernels)
  int pos = 0;
  int window = 0;   // 0: full attention (ONNX semantics); 2047: HF Phi-3 sliding window
  std::vector<char> skip;   // self-speculation drafter: layers skipped by rt_forward (empty: none)
  bool verbose = false;
  // streaming: weights that do not fit are read from the bundle every forward pass into two pinned, mapped
  // host buffers (zero-copy on Tegra: the GPU reads them in place), O_DIRECT so the page cache is not involved
  // host-side embedding (default; LLMRT_EMBED_DEVICE=1 keeps it resident): the raw8 rows stay in the bundle file,
  // mmap'ed (page cache, reclaimable), gathered per token on the CPU into a pinned buffer and copied to emb_rows
  const uint8_t* emb_map = nullptr;
  size_t emb_map_len = 0, emb_lead = 0;
  uint8_t* emb_pin = nullptr;
  uint8_t* emb_rows = nullptr;
  std::vector<int> stream_list;   // tensor indices in forward order
  uint8_t* pin[2] = {nullptr, nullptr};
  uint8_t* pin_dev[2] = {nullptr, nullptr};   // integrated GPU (Tegra): device alias of pin, read in place
  uint8_t* dbuf[2] = {nullptr, nullptr};      // discrete GPU: device copies (PCIe zero-copy would be slow)
  bool zero_copy = false;
  size_t lead[2] = {0, 0};
  size_t pin_bytes = 0;
  int fd_direct = -1;
  cudaEvent_t done[2];
  std::thread loader;
  std::mutex mu;
  std::condition_variable cv;
  long long loaded[2] = {-1, -1};   // stream sequence number held by each buffer
  long long consumed = 0;           // next sequence number the GPU side will use
  long long released = 0;           // sequence numbers whose kernels have finished on the GPU
  int cur_slot = -1;
  bool stop = false;
  // stats
  double t_stream_wait = 0, t_stream_bytes = 0;
  size_t resident_bytes = 0, streamed_bytes = 0;
  long long avail_after_kb = 0;

  Tensor& T(const std::string& n) {
    auto it = by_name.find(n);
    if (it == by_name.end()) { fprintf(stderr, "missing tensor %s\n", n.c_str()); exit(1); }
    return ts[it->second];
  }
  int ci(const char* k) { return atoi(cfg[k].c_str()); }
};

static void pread_all(int fd, void* dst, size_t n, size_t off) {
  uint8_t* d = (uint8_t*)dst;
  while (n) {
    ssize_t r = pread(fd, d, n, off);
    if (r <= 0) { perror("pread"); exit(1); }
    d += r; n -= r; off += r;
  }
}

static void loader_main(Model* M) {
  long long seq = 0;
  const long long n = (long long)M->stream_list.size();
  for (;;) {
    const int slot = seq & 1;
    {
      std::unique_lock<std::mutex> lk(M->mu);
      M->cv.wait(lk, [&] { return M->stop || seq < M->released + 2; });   // buffer's previous tensor finished
      if (M->stop) return;
    }
    const Tensor& t = M->ts[M->stream_list[seq % n]];
    size_t lead = 0;
    bool ok = false;
    if (M->fd_direct >= 0) {
      const size_t a = t.off & ~(size_t)4095;
      lead = t.off - a;
      const size_t len = (lead + t.nbytes + 4095) & ~(size_t)4095;
      size_t got = 0;
      while (got < len) {
        const ssize_t r = pread(M->fd_direct, M->pin[slot] + got, len - got, a + got);
        if (r <= 0) break;
        got += r;
      }
      ok = got >= lead + t.nbytes;
      if (!ok) { close(M->fd_direct); M->fd_direct = -1; }   // O_DIRECT refused (or short at EOF): buffered
    }
    if (!ok) {
      lead = 0;
      pread_all(M->fd, M->pin[slot], t.nbytes, t.off);
    }
    {
      std::lock_guard<std::mutex> lk(M->mu);
      M->lead[slot] = lead;
      M->loaded[slot] = seq;
    }
    M->cv.notify_all();
    ++seq;
  }
}

// device pointer for tensor t (streamed tensors: wait until the loader has filled the next buffer)
static const uint8_t* weight_ptr(Model* M, Tensor& t) {
  if (t.dev) return t.dev;
  const long long seq = M->consumed;
  const int slot = seq & 1;
  if (M->stream_list[seq % M->stream_list.size()] != (int)(&t - &M->ts[0])) {
    fprintf(stderr, "streamed tensor %s used out of forward order\n", t.name.c_str());
    exit(1);
  }
  // All CUDA synchronization stays on this thread (a loader blocked on a CUDA event can wait on work this thread
  // has not flushed yet). Once the previous streamed tensor's kernels are done, its buffer goes back to the loader,
  // which then reads the tensor after this one while the GPU works on this one.
  if (seq >= 1) {
    CK(cudaEventSynchronize(M->done[slot ^ 1]));
    {
      std::lock_guard<std::mutex> lk(M->mu);
      M->released = seq;
    }
    M->cv.notify_all();
  }
  const double t0 = now_s();
  {
    std::unique_lock<std::mutex> lk(M->mu);
    M->cv.wait(lk, [&] { return M->loaded[slot] == seq; });
  }
  M->t_stream_wait += now_s() - t0;
  M->t_stream_bytes += t.nbytes;
  M->consumed = seq + 1;
  M->cur_slot = slot;
  if (M->zero_copy) return M->pin_dev[slot] + M->lead[slot];
  CK(cudaMemcpyAsync(M->dbuf[slot], M->pin[slot] + M->lead[slot], t.nbytes, cudaMemcpyHostToDevice));
  return M->dbuf[slot];
}

// after the kernels reading a streamed tensor are enqueued: mark the point its buffer is free again
static void release_ptr(Model* M, const Tensor& t) {
  if (t.dev) return;
  CK(cudaEventRecord(M->done[M->cur_slot]));
}

// PHI3RT_SYNC=1: synchronize after every kernel and print its time (debugging / per-op profiling)
static int g_sync = -1;
static bool g_attn_old = getenv("LLMRT_ATTN_OLD") && atoi(getenv("LLMRT_ATTN_OLD"));   // per-query k_attn
// graph-bench: rt_forward is being captured into a CUDA graph (no host copies, no synchronization; needs a build with
// --default-stream per-thread so that kernel launches go to a capturable stream)
static bool g_capture = false;
// LLMRT_NO_LAYER_SYNC=1: skip the per-layer drain (graph-bench separates its cost from launch overhead)
static bool g_no_layer_sync = getenv("LLMRT_NO_LAYER_SYNC") && atoi(getenv("LLMRT_NO_LAYER_SYNC"));
// LLMRT_GRAPHS=1: decode and verify passes (m <= 8) replay CUDA graphs (saves ~4 us of GPU idle per kernel launch on
// the Orin; same kernels and arguments, so the same logits). Needs --default-stream per-thread.
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
static bool g_graphs = getenv("LLMRT_GRAPHS") && atoi(getenv("LLMRT_GRAPHS"));
#else
static bool g_graphs = false;
#endif
static void ksync(const char* what) {
  if (g_sync < 0) g_sync = getenv("PHI3RT_SYNC") ? atoi(getenv("PHI3RT_SYNC")) : 0;
  if (!g_sync) return;
  static double last = now_s();
  cudaError_t e = cudaDeviceSynchronize();
  const double t = now_s();
  fprintf(stderr, "[sync] %-12s %8.3f ms %s\n", what, (t - last) * 1e3, e == cudaSuccess ? "" : cudaGetErrorString(e));
  if (e != cudaSuccess) exit(1);
  last = now_s();
}

// interleaved Huffman decode of tile rows [tr0, tr1) of t (nb blocks) to the codes q
// Huffman-coded (interleaved) or packed matrix
static bool coded(const Tensor& t) { return t.enc == "huf8" || t.enc == "huf4p" || t.enc == "pk4" || t.enc == "pk8"; }

static void decode_il(const Tensor& t, const uint8_t* w, uint8_t* out, int tr0, int tr1, int nb) {
#define DEC(B, LT) k_decode_hufi<B, LT><<<nb, 256>>>(w, out, t.N, t.K, tr0, tr1, t.radd, t.G, t.gz, t.goff)
  if (t.bits == 8) { if (t.lt == 0) DEC(8, 0); else if (t.lt == 1) DEC(8, 1); else if (t.lt == 3) DEC(8, 3); else if (t.lt == 4) DEC(8, 4); else DEC(8, 2); }
  else { if (t.lt == 0) DEC(4, 0); else if (t.lt == 1) DEC(4, 1); else DEC(4, 2); }
#undef DEC
}

// host: packed codes [N, K * B / 8] (pk4: two codes per byte, low nibble first) -> LaneR words: tile t (64 rows x TK
// bytes), word i of lane l at t * 32 W + 32 i + l, lane l = rows l (i < W / 2) and l + 32, W = 2 TK / 4
static std::vector<uint32_t> interleave_packed(const uint8_t* blob, int N, int Kb) {
  const int KT = Kb / TK, NT = N / TN, W = 2 * TK / 4;
  std::vector<uint32_t> out((size_t)NT * KT * 32 * W);
  for (int tr = 0; tr < NT; ++tr)
    for (int tc = 0; tc < KT; ++tc) {
      uint32_t* o = out.data() + ((size_t)tr * KT + tc) * 32 * W;
      for (int l = 0; l < 32; ++l)
        for (int half = 0; half < 2; ++half) {
          const uint8_t* row = blob + (size_t)(tr * TN + half * 32 + l) * Kb + tc * TK;
          for (int i = 0; i < W / 2; ++i) {
            uint32_t v;
            memcpy(&v, row + 4 * i, 4);
            o[(size_t)(half * W / 2 + i) * 32 + l] = v;
          }
        }
    }
  return out;
}

// host: pk8 Q5_K / Q6_K codes [N, K] (one byte each, < 2^(4 + HB)) -> LaneQ<HB> words: per tile as interleave_packed
// the low nibbles (two codes per byte, code 2 b low), then after all tiles the high bits, HB per code
static std::vector<uint32_t> interleave_packed_q(const uint8_t* codes, int N, int K, int HB) {
  const int KT = K / TK, NT = N / TN, T = NT * KT, WN = 2 * TK / 8, WH = 2 * TK * HB / 32, CPW = 32 / HB;
  std::vector<uint32_t> out((size_t)T * 32 * (WN + WH));
  uint32_t* hbase = out.data() + (size_t)T * 32 * WN;
  for (int tr = 0; tr < NT; ++tr)
    for (int tc = 0; tc < KT; ++tc) {
      const size_t t = (size_t)tr * KT + tc;
      for (int l = 0; l < 32; ++l)
        for (int half = 0; half < 2; ++half) {
          const uint8_t* row = codes + (size_t)(tr * TN + half * 32 + l) * K + tc * TK;
          for (int i = 0; i < WN / 2; ++i) {
            uint32_t v = 0;
            for (int c = 0; c < 8; ++c) v |= (uint32_t)(row[8 * i + c] & 15) << (4 * c);
            out[(t * WN + half * WN / 2 + i) * 32 + l] = v;
          }
          for (int i = 0; i < WH / 2; ++i) {
            uint32_t v = 0;
            for (int c = 0; c < CPW; ++c) v |= (uint32_t)(row[CPW * i + c] >> 4) << (HB * c);
            hbase[(t * WH + half * WH / 2 + i) * 32 + l] = v;
          }
        }
    }
  return out;
}

// grs != null: group side info [N, G] (rows of this call), weight = grs (q - (grzc - 2^23)); s, zc unused
// kqt > 0: K-quant side bytes kqs [N, K / 256, SB] (rows of this call), weight = s f(q) - o; s = 1, zc unused
static void gemm_raw8(const uint8_t* w, const float* X, int m, float* Y, int N, int K, float s, float zc, int ldy,
                      const __half* grs = nullptr, const uint8_t* grzc = nullptr, int G = 1,
                      const uint8_t* kqs = nullptr, int kqt = 0) {
  if (m > 64 && N % 128 == 0 && ldy % 4 == 0) {
    dim3 grid(N / 128, (m + 127) / 128);
    k_gemm2_raw8<<<grid, 256>>>(w, X, Y, m, N, K, s, zc, ldy, grs, grzc, G, kqs, kqt);
  } else {
    dim3 grid(N / 64, (m + 63) / 64);
    k_gemm_raw8<<<grid, 256>>>(w, X, Y, m, N, K, s, zc, ldy, grs, grzc, G, kqs, kqt);
  }
}

// Y[M, N] (row stride ldy) = X[M, K] @ dequant(W)^T
static void matmul(Model* M, Tensor& t, const float* X, int m, float* Y, int ldy, int path = 0) {
  const uint8_t* w = weight_ptr(M, t);
  const float zc = 8388608.f + (float)t.zp;
  if (t.enc == "raw8") {
    if (m == 1 && path != 2) {
      const int blocks = (t.N + 31) / 32;
      k_gemv_raw8<<<blocks, 256, t.K * sizeof(float)>>>(w, X, Y, t.N, t.K, t.scale, zc);
    } else {
      gemm_raw8(w, X, m, Y, t.N, t.K, t.scale, zc, ldy);
    }
    // fractional zero-point offset (not representable next to 2^23): Y[j, :] += -s zadj sum_k X[j, k]
    if (t.zadj != 0.f) k_rowsum_add<<<m, 256>>>(X, Y, t.N, t.K, ldy, -t.scale * t.zadj);
  } else if (t.il && t.kqt) {
    const int TW = t.bits == 8 ? TK : 2 * TK;
    if (m <= 8 && path != 2) {
      const int blocks = (t.K / TW) * ((t.N / TN + 7) / 8);
      if (M->act8) {
        const int nb = m * t.K / 32;
        k_quant_x8<<<(nb + 7) / 8, 256>>>(X, M->xq, M->xqd, M->xqs, nb);
      }
#define KV(MT, B, LT, KQ) do { if (M->act8) k_gemvm_kq8<MT, B, LT, KQ><<<blocks, 256>>>(w, M->xq, M->xqd, M->xqs, M->xqp, m, t.N, t.K, t.kqs); \
                              else k_gemvm_kq<MT, B, LT, KQ><<<blocks, 256>>>(w, X, M->xqp, m, t.N, t.K, t.kqs); } while (0)
#define KVL(MT, B, KQ) do { if (t.lt == 0) KV(MT, B, 0, KQ); else if (t.lt == 1) KV(MT, B, 1, KQ); else KV(MT, B, 2, KQ); } while (0)
#define KVL8(MT, KQ) do { if (t.lt == 3) KV(MT, 8, 3, KQ); else if (t.lt == 4) KV(MT, 8, 4, KQ); else KVL(MT, 8, KQ); } while (0)
#define KVQ(MT) do { if (t.kqt == 1) KVL(MT, 4, 1); else if (t.kqt == 2) KVL8(MT, 2); \
                     else if (t.kqt == 3) KVL8(MT, 3); else KVL(MT, 4, 4); } while (0)
      if (m == 1) KVQ(1); else if (m == 2) KVQ(2); else if (m <= 4) KVQ(4); else KVQ(8);
#undef KVQ
#undef KVL8
#undef KVL
#undef KV
      k_sum_kt<<<(m * t.N + 255) / 256, 256>>>(M->xqp, Y, m, t.N, ldy, t.K / TW);
    } else {   // decode the codes into scratch (no re-centering: add 0), GEMM with the side bytes
      const int KT = t.K / TW, NT = t.N / TN;
      const size_t rowb = (size_t)(t.K / 256) * kq_sbytes(t.kqt);
      const int rows_fit = std::max(1, (int)(M->scratch_bytes / ((size_t)TN * t.K)));
      for (int tr0 = 0; tr0 < NT; tr0 += rows_fit) {
        const int tr1 = std::min(NT, tr0 + rows_fit);
        const int nb = ((tr1 - tr0) * KT + 7) / 8;
#define DEC(B, LT) k_decode_hufi<B, LT><<<nb, 256>>>(w, M->scratch, t.N, t.K, tr0, tr1, nullptr, 2, nullptr, -1)
        if (t.bits == 8) { if (t.lt == 0) DEC(8, 0); else if (t.lt == 1) DEC(8, 1); else if (t.lt == 3) DEC(8, 3); else if (t.lt == 4) DEC(8, 4); else DEC(8, 2); }
        else { if (t.lt == 0) DEC(4, 0); else if (t.lt == 1) DEC(4, 1); else DEC(4, 2); }
#undef DEC
        ksync("  decode");
        gemm_raw8(M->scratch, X, m, Y + tr0 * TN, (tr1 - tr0) * TN, t.K, 1.f, 0.f, ldy, nullptr, nullptr, 1,
                  t.kqs + (size_t)tr0 * TN * rowb, t.kqt);
        ksync("  gemm");
      }
    }
  } else if (t.il) {
    const int TW = t.bits == 8 ? TK : 2 * TK;
    if (m <= 8 && path != 2) {
      CK(cudaMemsetAsync(Y, 0, (size_t)m * t.N * sizeof(float)));
      if (ldy != t.N) { fprintf(stderr, "%s: ldy %d != N\n", t.name.c_str(), ldy); exit(1); }
      const int blocks = (t.K / TW) * ((t.N / TN + 7) / 8);
#define GV(MT, B, LT, GR) \
  k_gemvm_hufi<MT, B, LT, GR><<<blocks, 256>>>(w, X, Y, m, t.N, t.K, t.rs, t.radd, t.rzc, t.G, t.gs, t.gz, t.goff)
#define GEMVM(MT, B)                                                    \
  do {                                                                  \
    if (t.G > 1) {                                                      \
      if (t.lt == 0) GV(MT, B, 0, true); else if (t.lt == 1) GV(MT, B, 1, true); else GV(MT, B, 2, true); \
    } else {                                                            \
      if (t.lt == 0) GV(MT, B, 0, false); else if (t.lt == 1) GV(MT, B, 1, false); else GV(MT, B, 2, false); \
    }                                                                   \
  } while (0)
      if (t.bits == 8) {
        if (m == 1) GEMVM(1, 8); else if (m == 2) GEMVM(2, 8); else if (m <= 4) GEMVM(4, 8); else GEMVM(8, 8);
      } else {
        if (m == 1) GEMVM(1, 4); else if (m == 2) GEMVM(2, 4); else if (m <= 4) GEMVM(4, 4); else GEMVM(8, 4);
      }
#undef GEMVM
#undef GV
    } else {
      const int KT = t.K / TW, NT = t.N / TN;
      const int rows_fit = std::max(1, (int)(M->scratch_bytes / ((size_t)TN * t.K)));
      for (int tr0 = 0; tr0 < NT; tr0 += rows_fit) {
        const int tr1 = std::min(NT, tr0 + rows_fit);
        const int nb = ((tr1 - tr0) * KT + 7) / 8;
        decode_il(t, w, M->scratch, tr0, tr1, nb);
        ksync("  decode");
        if (t.G > 1) {
          const size_t g0 = (size_t)tr0 * TN * t.G;
          gemm_raw8(M->scratch, X, m, Y + tr0 * TN, (tr1 - tr0) * TN, t.K, 1.f, 0.f, ldy, t.gs + g0, t.gz + g0, t.G);
        } else {
          gemm_raw8(M->scratch, X, m, Y + tr0 * TN, (tr1 - tr0) * TN, t.K, 1.f, 8388608.f, ldy);
        }
        ksync("  gemm");
      }
      if (t.G == 1) {
        k_rowsum<<<m, 256>>>(X, M->sx, t.K);
        k_rowaffine<<<std::min(1024, (m * t.N + 255) / 256), 256>>>(Y, m, t.N, ldy, t.rs, t.rzc, M->sx);
      }
    }
  } else {
    const int T = (t.N / TN) * (t.K / TK);
    if ((m == 1 && path != 2) || path == 1) {
      for (int r = 0; r < m; ++r) {
        CK(cudaMemsetAsync(Y + (size_t)r * ldy, 0, t.N * sizeof(float)));
        k_gemv_huf8<<<(T + 7) / 8, 256>>>(w, X + (size_t)r * t.K, Y + (size_t)r * ldy, t.N, t.K, t.scale, zc);
      }
    } else {   // decode tile rows into scratch, then raw8 GEMM, chunk by chunk
      const int KT = t.K / TK, NT = t.N / TN;
      const int rows_fit = std::max(1, (int)(M->scratch_bytes / ((size_t)TN * t.K)));
      for (int tr0 = 0; tr0 < NT; tr0 += rows_fit) {
        const int tr1 = std::min(NT, tr0 + rows_fit);
        k_decode_huf8<<<((tr1 - tr0) * KT + 7) / 8, 256>>>(w, M->scratch, t.N, t.K, tr0, tr1);
        ksync("  decode");
        gemm_raw8(M->scratch, X, m, Y + tr0 * TN, (tr1 - tr0) * TN, t.K, t.scale, zc, ldy);
        ksync("  gemm");
      }
    }
  }
  CK(cudaGetLastError());
  release_ptr(M, t);
}

static void* dalloc(size_t n) {
  void* p;
  CK(cudaMalloc(&p, n));
  return p;
}

extern "C" {

void rt_close(void* hm);

// budget_mb < 0: as much as the MemAvailable floor allows; 0: stream everything that can be streamed
void* rt_open(const char* dir, int max_ctx, int max_batch, long long budget_mb, int floor_mb, int verbose) {
  Model* M = new Model();
  M->verbose = verbose;
  std::ifstream mf(std::string(dir) + "/manifest.txt");
  if (!mf) { fprintf(stderr, "cannot open %s/manifest.txt\n", dir); return nullptr; }
  std::string line;
  while (std::getline(mf, line)) {
    std::istringstream is(line);
    std::string tag;
    is >> tag;
    if (tag == "cfg") {
      std::string k, v;
      is >> k >> v;
      M->cfg[k] = v;
    } else if (tag == "t") {
      Tensor t;
      is >> t.name >> t.enc >> t.N >> t.K >> t.scale >> t.zp >> t.off >> t.nbytes;
      for (const char* x : {"_ilc", "_il"}) {   // pre-interleaved (export-il): the device layout as stored
        const size_t n = strlen(x);
        if (t.enc.size() > n && !t.enc.compare(t.enc.size() - n, n, x)) {
          t.enc.resize(t.enc.size() - n);
          t.pre_il = t.il = true;
          t.compact = x[3] == 'c' && t.enc[0] != 'p';
          t.lt = t.enc[0] == 'p' ? 2 : t.compact ? 1 : 0;
          break;
        }
      }
      M->by_name[t.name] = (int)M->ts.size();
      M->ts.push_back(t);
    }
  }
  M->fd = open((std::string(dir) + "/weights.bin").c_str(), O_RDONLY);
  if (M->fd < 0) { perror("weights.bin"); return nullptr; }
  M->L = M->ci("n_layers"); M->D = M->ci("hidden"); M->H = M->ci("n_heads"); M->Dh = M->ci("head_dim");
  M->I = M->ci("inter"); M->V = M->ci("vocab");
  M->Hk = M->cfg.count("n_kv_heads") ? M->ci("n_kv_heads") : M->H;
  {
    std::stringstream es(M->cfg.count("eos") ? M->cfg["eos"] : "32000,32007,32001");   // phi3 default
    std::string x;
    while (std::getline(es, x, ',')) M->eos.push_back(atoi(x.c_str()));
  }
  for (auto& t : M->ts)
    if (t.enc == "huf4p" || t.enc == "pk4") t.bits = 4;
  M->eps = (float)atof(M->cfg["eps"].c_str());
  M->attn_scale = (float)atof(M->cfg["attn_scale"].c_str());
  M->Tmax = std::min(max_ctx, M->ci("rope_rows"));
  M->max_batch = max_batch;
  CK(cudaFree(0));
  const long long avail0 = meminfo_kb("MemAvailable");
  // activations, KV cache
  const size_t B = max_batch, D = M->D;
  M->h = (float*)dalloc(B * D * 4); M->xn = (float*)dalloc(B * D * 4); M->att = (float*)dalloc(B * D * 4);
  M->o = (float*)dalloc(B * D * 4); M->qkv = (float*)dalloc(B * (M->H + 2 * M->Hk) * M->Dh * 4);
  M->g = (float*)dalloc(B * M->I * 4); M->u = (float*)dalloc(B * M->I * 4);
  M->logits = (float*)dalloc(B * (size_t)M->V * 4);
  M->ids = (int*)dalloc((B + 1) * 4);
  M->posd = M->ids + B;
  CK(cudaMallocHost(&M->ids_host, (B + 1) * 4));
  M->sx = (float*)dalloc(B * 4);
  const size_t kv = (size_t)M->L * M->Hk * M->Tmax * M->Dh;
  M->kc = (__half*)dalloc(kv * 2); M->vc = (__half*)dalloc(kv * 2);
  // scratch for prefill decode of huf8 weights: the largest non-lm_head matrix, capped at LLMRT_SCRATCH_MB
  // (default 16; prefill then decodes + multiplies a matrix in tile-row chunks); at least one tile row of any matrix
  size_t min_scratch = 0;
  for (auto& t : M->ts)
    if (coded(t)) {
      if (t.name != "lm_head") M->scratch_bytes = std::max(M->scratch_bytes, (size_t)t.N * t.K);
      min_scratch = std::max(min_scratch, (size_t)TN * t.K);
    }
  {
    const double cap_mb = getenv("LLMRT_SCRATCH_MB") ? atof(getenv("LLMRT_SCRATCH_MB")) : 16.0;
    if (cap_mb > 0) M->scratch_bytes = std::min(M->scratch_bytes, std::max(min_scratch, (size_t)(cap_mb * 1048576.0)));
  }
  if (M->scratch_bytes) M->scratch = (uint8_t*)dalloc(M->scratch_bytes);
  {
    Tensor& E = M->T("embed");
    if (E.enc == "raw8" && !(getenv("LLMRT_EMBED_DEVICE") && atoi(getenv("LLMRT_EMBED_DEVICE")))) {
      const size_t pg = (size_t)sysconf(_SC_PAGESIZE), a = E.off & ~(pg - 1);
      M->emb_lead = E.off - a;
      M->emb_map_len = M->emb_lead + E.nbytes;
      void* p = mmap(nullptr, M->emb_map_len, PROT_READ, MAP_SHARED, M->fd, (off_t)a);
      if (p == MAP_FAILED) { perror("mmap embed"); return nullptr; }
      M->emb_map = (const uint8_t*)p;
      CK(cudaHostAlloc((void**)&M->emb_pin, B * D, cudaHostAllocDefault));
      M->emb_rows = (uint8_t*)dalloc(B * D);
    }
  }
  // residency: f32 + embedding first, then matmuls in forward order, while the budget and the floor allow
  std::vector<int> order;
  auto side = [](const std::string& n) {   // side tensors: read on the host (below), not kept as device tensors
    for (const char* x : {".rs", ".rsub", ".radd", ".kqs"})
      if (n.size() > strlen(x) && !n.compare(n.size() - strlen(x), strlen(x), x)) return true;
    return false;
  };
  for (size_t i = 0; i < M->ts.size(); ++i)
    if ((M->ts[i].enc == "f32" && !side(M->ts[i].name)) || (M->ts[i].name == "embed" && !M->emb_map)) order.push_back(i);
  std::vector<int> mats;
  for (int l = 0; l < M->L; ++l)
    for (const char* s : {"qkv", "q", "k", "v", "o", "gate", "up", "down"}) {
      const std::string n = "l" + std::to_string(l) + "." + s;
      if (M->by_name.count(n)) mats.push_back(M->by_name[n]);
    }
  mats.push_back(M->by_name["lm_head"]);
  size_t max_stream = 0;
  for (int i : mats) max_stream = std::max(max_stream, M->ts[i].nbytes);
  std::vector<uint8_t> host;
  long long used = 0;
  const long long floor_kb = (long long)floor_mb * 1024;
  auto try_resident = [&](Tensor& t, bool must) {
    const long long need_kb = (long long)(t.nbytes >> 10) + 1;
    const long long reserve_kb = must ? 0 : (long long)((2 * max_stream + 16384) >> 10);  // pinned stream buffers
    const bool budget_ok = must || budget_mb < 0 || used + (long long)t.nbytes <= budget_mb * 1048576LL;
    if (!budget_ok || meminfo_kb("MemAvailable") - need_kb - reserve_kb < floor_kb) {
      if (must) { fprintf(stderr, "cannot place %s above the MemAvailable floor\n", t.name.c_str()); exit(1); }
      return false;
    }
    if (!t.pre_il && ((t.enc == "huf8" && !getenv("PHI3SPEC_NO_IL")) || t.enc == "huf4p" || t.enc == "pk4" ||
                      t.enc == "pk8")) {
      std::vector<uint8_t> blob(t.nbytes);
      pread_all(M->fd, blob.data(), t.nbytes, t.off);
      static const bool compact = getenv("LLMRT_COMPACT") && atoi(getenv("LLMRT_COMPACT"));
      const bool pk = t.enc[0] == 'p';
      // pk8 Q5_K / Q6_K twin: repacked to the GGUF width (5 / 6 bits per code) unless LLMRT_PK8_WIDE=1
      int hb = 0;
      static const bool wide = getenv("LLMRT_PK8_WIDE") && atoi(getenv("LLMRT_PK8_WIDE"));
      if (pk && t.enc == "pk8" && !wide && M->by_name.count(t.name + ".kqs")) {
        const std::string& se = M->ts[M->by_name[t.name + ".kqs"]].enc;
        hb = se == "kq_Q5_K" ? 1 : se == "kq_Q6_K" ? 2 : 0;
      }
      const std::vector<uint32_t> iw = hb ? interleave_packed_q(blob.data(), t.N, t.K, hb)
                                     : pk ? interleave_packed(blob.data(), t.N, t.K / (8 / t.bits))
                                          : interleave_huf8(blob.data(), t.nbytes, t.N, t.K / (8 / t.bits), compact);
      t.compact = compact && !pk;
      t.lt = hb ? 2 + hb : pk ? 2 : t.compact ? 1 : 0;
      const size_t nb = iw.size() * 4;
      if (cudaMalloc(&t.dev, nb) != cudaSuccess) { cudaGetLastError(); t.dev = nullptr; return false; }
      CK(cudaMemcpy(t.dev, iw.data(), nb, cudaMemcpyHostToDevice));
      t.il = true;
      used += nb;
      M->resident_bytes += nb;
      return true;
    }
    if (cudaMalloc(&t.dev, t.nbytes) != cudaSuccess) { cudaGetLastError(); t.dev = nullptr; return false; }
    const size_t chunk = 64 << 20;
    if (host.size() < std::min(chunk, t.nbytes)) host.resize(std::min(chunk, t.nbytes));
    for (size_t o = 0; o < t.nbytes; o += chunk) {
      const size_t n = std::min(chunk, t.nbytes - o);
      pread_all(M->fd, host.data(), n, t.off + o);
      // PHI3SPEC_ROUND_BITS=b (raw8 MatMuls): the drafter of a split-code format, r = q - z split into
      // high = floor(r / 2^b) (Huffman-coded) and low = r mod 2^b (raw); the drafter dequantizes high at its bin
      // midpoint, r_d = 2^b high + (2^b - 1) / 2 (stored as code z + 2^b high, zero point shifted by -(2^b - 1) / 2)
      static const int rb = getenv("PHI3SPEC_ROUND_BITS") ? atoi(getenv("PHI3SPEC_ROUND_BITS")) : 0;
      if (rb && t.enc == "raw8" && t.name != "embed") {
        t.zadj = -0.5f * ((1 << rb) - 1);
        for (size_t i = 0; i < n; ++i) {
          const int r = (int)host[i] - t.zp;
          const int v = t.zp + (r >> rb) * (1 << rb);   // arithmetic shift: floor
          host[i] = (uint8_t)std::min(255, std::max(0, v));
        }
      }
      CK(cudaMemcpy(t.dev + o, host.data(), n, cudaMemcpyHostToDevice));
#ifdef POSIX_FADV_DONTNEED
      posix_fadvise(M->fd, t.off + o, n, POSIX_FADV_DONTNEED);
#endif
    }
    used += t.nbytes;
    M->resident_bytes += t.nbytes;
    return true;
  };
  const double t0 = now_s();
  for (int i : order) try_resident(M->ts[i], true);
  bool streaming = false;
  for (int i : mats) {
    Tensor& t = M->ts[i];
    if (!streaming && try_resident(t, false)) continue;
    streaming = true;   // keep forward order contiguous: everything after the first miss is streamed
    t.stream_idx = (int)M->stream_list.size();
    M->stream_list.push_back(i);
    M->streamed_bytes += t.nbytes;
    M->pin_bytes = std::max(M->pin_bytes, t.nbytes);
  }
  std::vector<uint8_t>().swap(host);
  if (!M->stream_list.empty()) {
    M->pin_bytes = (M->pin_bytes + 8192 + 4095) & ~(size_t)4095;
    int integrated = 0;
    CK(cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, 0));
    M->zero_copy = integrated && !getenv("PHI3RT_NO_ZERO_COPY");
    for (int s = 0; s < 2; ++s) {
      CK(cudaHostAlloc((void**)&M->pin[s], M->pin_bytes, cudaHostAllocMapped));
      CK(cudaHostGetDevicePointer((void**)&M->pin_dev[s], M->pin[s], 0));
      if (!M->zero_copy) M->dbuf[s] = (uint8_t*)dalloc(M->pin_bytes);
      CK(cudaEventCreateWithFlags(&M->done[s], cudaEventDisableTiming));
    }
    if (!getenv("PHI3RT_NO_DIRECT")) M->fd_direct = open((std::string(dir) + "/weights.bin").c_str(), O_RDONLY | O_DIRECT);
    M->loader = std::thread(loader_main, M);
  }
  // per-row side info: Huffman matrices (bundle tensors <name>.rs / .rsub / .radd, else the per-tensor s, z with
  // raw codes as symbols) and the embedding (embed.rs / embed.rsub, else per tensor)
  auto host_f32 = [&](const std::string& n, std::vector<float>& v) {
    if (!M->by_name.count(n)) return false;
    Tensor& x = M->ts[M->by_name[n]];
    v.resize((size_t)x.N * x.K);
    pread_all(M->fd, v.data(), v.size() * 4, x.off);
    return true;
  };
  for (auto& t : M->ts) {
    if (t.enc == "raw8" && t.name == "embed") {
      std::vector<float> a, b;
      if (host_f32("embed.rs", a) && host_f32("embed.rsub", b)) {
        t.G = (int)(a.size() / t.N);
        t.rs = (float*)dalloc(a.size() * 4);
        t.rz = (float*)dalloc(b.size() * 4);
        CK(cudaMemcpy(t.rs, a.data(), a.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(t.rz, b.data(), b.size() * 4, cudaMemcpyHostToDevice));
      }
      continue;
    }
    if (!coded(t)) continue;
    if (M->by_name.count(t.name + ".kqs")) {   // K-quant side bytes, resident as they are
      const Tensor& x = M->ts[M->by_name[t.name + ".kqs"]];
      static const char* kn[] = {"", "kq_Q4_K", "kq_Q5_K", "kq_Q6_K", "kq_IQ4_XS"};
      for (int i = 1; i <= 4; ++i)
        if (x.enc == kn[i]) t.kqt = i;
      if (!t.kqt || x.N != t.N || (size_t)x.K != (size_t)(t.K / 256) * kq_sbytes(t.kqt) || !t.il ||
          t.bits != (t.kqt == 1 || t.kqt == 4 ? 4 : 8)) {
        fprintf(stderr, "%s: bad K-quant side tensor %s (or matrix not resident)\n", t.name.c_str(), x.enc.c_str());
        exit(1);
      }
      std::vector<uint8_t> b(x.nbytes);
      pread_all(M->fd, b.data(), x.nbytes, x.off);
      t.G = 2;      // decode_il: raw codes (add 0)
      t.goff = -1;
      t.kqs = (uint8_t*)dalloc(x.nbytes);
      CK(cudaMemcpy(t.kqs, b.data(), x.nbytes, cudaMemcpyHostToDevice));
      M->resident_bytes += x.nbytes;
      continue;
    }
    std::vector<float> rs, rsub, radd;
    const bool row = host_f32(t.name + ".rs", rs) && host_f32(t.name + ".rsub", rsub) && host_f32(t.name + ".radd", radd);
    if (!row) {
      rs.assign(t.N, t.scale);
      rsub.assign(t.N, (float)t.zp);
      radd.assign(t.N, 0.f);   // raw codes: q = symbol
    }
    if ((row || t.bits == 4 || t.enc[0] == 'p') && !t.il) {
      fprintf(stderr, "%s: re-centered Huffman matrices must be resident\n", t.name.c_str());
      exit(1);
    }
    t.G = (int)(rs.size() / t.N);
    if (t.G > 1 && (t.K % t.G || (t.bits == 8 ? TK : 2 * TK) % (t.K / t.G))) {   // groups must tile a tile column
      fprintf(stderr, "%s: %d side-info groups per row of %d\n", t.name.c_str(), t.G, t.K);
      exit(1);
    }
    const size_t ng = (size_t)t.N * t.G;
    std::vector<float> zc(ng);
    t.h_radd.resize(ng);
    t.h_z.resize(ng);
    for (size_t i = 0; i < ng; ++i) {
      zc[i] = 8388608.f + rsub[i];
      t.h_radd[i] = (int)radd[i];
      t.h_z[i] = (int)rsub[i];
    }
    if (t.G > 1) {   // compact on the device: fp16 scale + byte zero point, add = z + goff
      const bool add0 = std::all_of(t.h_radd.begin(), t.h_radd.end(), [](int v) { return v == 0; });   // packed
      t.goff = add0 ? -1 : (t.h_radd[0] - t.h_z[0]) & ((1 << t.bits) - 1);
      std::vector<__half> hs(ng);
      std::vector<uint8_t> hz(ng);
      for (size_t i = 0; i < ng; ++i) {
        hs[i] = __float2half_rn(rs[i]);
        hz[i] = (uint8_t)t.h_z[i];
        if (__half2float(hs[i]) != rs[i] || t.h_z[i] < 0 || t.h_z[i] >= (1 << t.bits) ||
            (!add0 && ((t.h_radd[i] - t.h_z[i]) & ((1 << t.bits) - 1)) != t.goff)) {
          fprintf(stderr, "%s: group side info not fp16 scale + b-bit zero point (group %zu)\n", t.name.c_str(), i);
          exit(1);
        }
      }
      t.gs = (__half*)dalloc(ng * 2);
      t.gz = (uint8_t*)dalloc(ng);
      CK(cudaMemcpy(t.gs, hs.data(), ng * 2, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(t.gz, hz.data(), ng, cudaMemcpyHostToDevice));
      M->resident_bytes += ng * 3;
      continue;
    }
    t.rs = (float*)dalloc(ng * 4);
    t.rzc = (float*)dalloc(ng * 4);
    t.radd = (int*)dalloc(ng * 4);
    CK(cudaMemcpy(t.rs, rs.data(), ng * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(t.rzc, zc.data(), ng * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(t.radd, t.h_radd.data(), ng * 4, cudaMemcpyHostToDevice));
  }
  {
    int maj = 0, mnr = 0, kmax = 0;
    size_t pmax = 0;
    CK(cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, 0));
    CK(cudaDeviceGetAttribute(&mnr, cudaDevAttrComputeCapabilityMinor, 0));
    for (auto& t : M->ts)
      if (t.il && t.kqt) {
        kmax = std::max(kmax, t.K);
        pmax = std::max(pmax, (size_t)(t.K / (t.bits == 8 ? TK : 2 * TK)) * t.N);
      }
    M->act8 = kmax > 0 && maj * 10 + mnr >= 61 && !(getenv("LLMRT_FP_ACT") && atoi(getenv("LLMRT_FP_ACT")));
    if (M->act8) {
      M->xq = (int8_t*)dalloc((size_t)8 * kmax);
      M->xqd = (float*)dalloc((size_t)8 * kmax / 32 * sizeof(float));
      M->xqs = (float*)dalloc((size_t)8 * kmax / 32 * sizeof(float));
    }
    if (kmax > 0) M->xqp = (float*)dalloc(8 * pmax * sizeof(float));
  }
  M->avail_after_kb = meminfo_kb("MemAvailable");
  if (verbose)
    fprintf(stderr, "[phi3rt] %s: resident %.1f MB, streamed %.1f MB (%zu tensors), embed %s, scratch %.1f MB, ctx %d, "
                    "batch %d, load %.1f s, MemAvailable %lld -> %lld MB\n",
            dir, M->resident_bytes / 1e6, M->streamed_bytes / 1e6, M->stream_list.size(),
            M->emb_map ? "host (mmap)" : "device", M->scratch_bytes / 1e6, M->Tmax, max_batch, now_s() - t0,
            avail0 / 1024, M->avail_after_kb / 1024);
  return M;
}

void rt_close(void* hm) {
  Model* M = (Model*)hm;
  if (M->loader.joinable()) {
    {
      std::lock_guard<std::mutex> lk(M->mu);
      M->stop = true;
    }
    M->cv.notify_all();
    M->loader.join();
  }
  for (auto& t : M->ts)
    if (t.dev) cudaFree(t.dev);
  for (void* p : {(void*)M->h, (void*)M->xn, (void*)M->att, (void*)M->o, (void*)M->qkv, (void*)M->g, (void*)M->u,
                  (void*)M->logits, (void*)M->ids, (void*)M->kc, (void*)M->vc, (void*)M->scratch})
    if (p) cudaFree(p);
  if (M->emb_map) munmap((void*)M->emb_map, M->emb_map_len);
  if (M->emb_pin) cudaFreeHost(M->emb_pin);
  if (M->ids_host) cudaFreeHost(M->ids_host);
  if (M->n_graph_launch) fprintf(stderr, "[graphs] %d captured, %d launches\n", M->n_graph_capture, M->n_graph_launch);
  for (auto& kv : M->graphs) cudaGraphExecDestroy(kv.second);
  if (M->emb_rows) cudaFree(M->emb_rows);
  for (int s = 0; s < 2; ++s) {
    if (M->pin[s]) cudaFreeHost(M->pin[s]);
    if (M->dbuf[s]) cudaFree(M->dbuf[s]);
  }
  if (M->fd_direct >= 0) close(M->fd_direct);
  close(M->fd);
  delete M;
}

void rt_reset(void* hm) { ((Model*)hm)->pos = 0; }
void rt_set_window(void* hm, int w) { ((Model*)hm)->window = std::max(0, w); }
// rewind to an earlier position (the KV cache below it is kept; used to share a prompt prefix)
void rt_set_pos(void* hm, int p) { ((Model*)hm)->pos = std::max(0, std::min(p, ((Model*)hm)->pos)); }
int rt_pos(void* hm) { return ((Model*)hm)->pos; }
int rt_vocab(void* hm) { return ((Model*)hm)->V; }

double rt_stat(void* hm, const char* k) {
  Model* M = (Model*)hm;
  const std::string s(k);
  if (s == "resident_bytes") return (double)M->resident_bytes;
  if (s == "streamed_bytes") return (double)M->streamed_bytes;
  if (s == "stream_wait_s") return M->t_stream_wait;
  if (s == "stream_bytes_moved") return M->t_stream_bytes;
  if (s == "mem_available_mb") return M->avail_after_kb / 1024.0;
  if (s == "n_streamed") return (double)M->stream_list.size();
  if (s == "graph_launches") return M->n_graph_launch;
  if (s == "graph_captures") return M->n_graph_capture;
  return -1;
}

// Forward n tokens at the current position (chunks of max_batch). logits: n x V if all_logits, else V (last).
int rt_forward(void* hm, const int* ids, int n, float* logits, int all_logits) {
  Model* M = (Model*)hm;
  if (M->pos + n > M->Tmax) { fprintf(stderr, "context overflow (%d + %d > %d)\n", M->pos, n, M->Tmax); return -1; }
  if (g_graphs && !g_capture && n <= 8 && n <= M->max_batch && M->stream_list.empty() && M->skip.empty() &&
      !g_attn_old && !g_sync) {
    // score buffer bucket: graphs for positions up to Tb share one shared-memory size (the kernels' arithmetic does
    // not depend on it)
    const int Tb = std::min(M->Tmax, (M->pos + n + 255) / 256 * 256);
    const long long key = ((long long)Tb << 8) | (n << 1) | (all_logits ? 1 : 0);
    auto it = M->graphs.find(key);
    if (it != M->graphs.end() || M->graph_seen.count(key)) {   // first occurrence runs directly (sets attributes)
      memcpy(M->ids_host, ids, n * 4);
      M->ids_host[M->max_batch] = M->pos;
      CK(cudaMemcpyAsync(M->ids, M->ids_host, (M->max_batch + 1) * 4, cudaMemcpyHostToDevice, cudaStreamPerThread));
      if (M->emb_map) {
        for (int i = 0; i < n; ++i) memcpy(M->emb_pin + (size_t)i * M->D, M->emb_map + M->emb_lead + (size_t)ids[i] * M->D, M->D);
        CK(cudaMemcpyAsync(M->emb_rows, M->emb_pin, (size_t)n * M->D, cudaMemcpyHostToDevice, cudaStreamPerThread));
      }
      if (it == M->graphs.end()) {
        const int p0 = M->pos;
        cudaGraph_t g;
        g_capture = true;
        M->graph_T = Tb;
        CK(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeRelaxed));
        rt_forward(hm, ids, n, nullptr, all_logits);
        CK(cudaStreamEndCapture(cudaStreamPerThread, &g));
        g_capture = false;
        M->graph_T = 0;
        M->pos = p0;
        cudaGraphExec_t ge;
#if CUDART_VERSION >= 11040
        CK(cudaGraphInstantiate(&ge, g, 0));
#else
        CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
#endif
        CK(cudaGraphDestroy(g));
        it = M->graphs.emplace(key, ge).first;
        ++M->n_graph_capture;
      }
      CK(cudaGraphLaunch(it->second, cudaStreamPerThread));
      ++M->n_graph_launch;
      M->pos += n;
      if (logits) {
        const size_t nl = all_logits ? (size_t)n * M->V : (size_t)M->V;
        CK(cudaMemcpyAsync(logits, M->logits, nl * 4, cudaMemcpyDeviceToHost, cudaStreamPerThread));
      }
      CK(cudaStreamSynchronize(cudaStreamPerThread));
      return 0;
    }
    M->graph_seen.insert(key);
  }
  const int D = M->D, H = M->H, Dh = M->Dh, I = M->I;
  const int Hk = M->Hk;
  const size_t kvl = (size_t)Hk * M->Tmax * Dh;
  for (int c0 = 0; c0 < n; c0 += M->max_batch) {
    const int m = std::min(M->max_batch, n - c0);
    if (!g_capture) {   // capture: ids and pos are uploaded before each launch
      memcpy(M->ids_host, ids + c0, m * 4);
      M->ids_host[M->max_batch] = M->pos;
      CK(cudaMemcpy(M->ids, M->ids_host, (M->max_batch + 1) * 4, cudaMemcpyHostToDevice));
    }
    const Tensor& E = M->T("embed");
    ksync("start");
    if (M->emb_map && !g_capture) {   // capture: the rows were copied by the direct pass before it
      for (int i = 0; i < m; ++i) memcpy(M->emb_pin + (size_t)i * D, M->emb_map + M->emb_lead + (size_t)ids[c0 + i] * D, D);
      CK(cudaMemcpy(M->emb_rows, M->emb_pin, (size_t)m * D, cudaMemcpyHostToDevice));
    }
    k_embed<<<m, 256>>>(M->emb_map ? M->emb_rows : E.dev, M->ids, M->h, D, E.scale, (float)E.zp, E.rs, E.rz,
                        M->emb_map != nullptr, E.G);
    ksync("embed");
    auto kept = [&](int l) { while (l < M->L && !M->skip.empty() && M->skip[l]) ++l; return l; };
    if (!M->skip.empty() && !M->stream_list.empty()) { fprintf(stderr, "layer skipping needs resident weights\n"); exit(1); }
    k_rmsnorm<<<m, 256>>>(M->h, nullptr, (const float*)M->T("l" + std::to_string(kept(0)) + ".ln1").dev, M->xn, D,
                          M->eps);
    ksync("rmsnorm");
    for (int l = kept(0); l < M->L; l = kept(l + 1)) {
      const std::string p = "l" + std::to_string(l) + ".";
      if (M->by_name.count(p + "qkv")) {
        matmul(M, M->T(p + "qkv"), M->xn, m, M->qkv, (H + 2 * Hk) * Dh);
      } else {   // q, k, v of different formats (K-quant bundles): three matmuls into the qkv rows
        matmul(M, M->T(p + "q"), M->xn, m, M->qkv, (H + 2 * Hk) * Dh);
        matmul(M, M->T(p + "k"), M->xn, m, M->qkv + H * Dh, (H + 2 * Hk) * Dh);
        matmul(M, M->T(p + "v"), M->xn, m, M->qkv + (H + Hk) * Dh, (H + 2 * Hk) * Dh);
      }
      ksync("qkv");
      k_rope_cache<<<m, 256>>>(M->qkv, (const float*)M->T("cos").dev, (const float*)M->T("sin").dev, M->kc + l * kvl,
                               M->vc + l * kvl, M->posd, H, Hk, Dh, M->Tmax);
      ksync("rope");
      if (m <= 8 && !g_attn_old) {   // decode and verify passes: all queries of a head in one block
        const int Ts = M->graph_T ? M->graph_T : M->pos + m;   // score row stride is pos + m either way
        const size_t smem = ((size_t)m * Ts + (size_t)(256 / Dh) * 8 * Dh) * sizeof(float);
        const size_t smem_max = ((size_t)8 * M->Tmax + (size_t)(256 / Dh) * 8 * Dh) * sizeof(float);
#define ATTN(DH) do { static bool attr = false; \
                      if (!attr) { CK(cudaFuncSetAttribute(k_attn_mq<DH, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, \
                                                           (int)smem_max)); attr = true; } \
                      k_attn_mq<DH, 8><<<H, 256, smem>>>(M->qkv, M->kc + l * kvl, M->vc + l * kvl, M->att, m, M->posd, \
                                                         H, Hk, M->Tmax, M->attn_scale, M->window); } while (0)
        if (Dh == 64) ATTN(64); else if (Dh == 96) ATTN(96); else if (Dh == 128) ATTN(128);
        else { fprintf(stderr, "head_dim %d unsupported\n", Dh); exit(1); }
#undef ATTN
      } else {
        dim3 ag(m, H);
        const size_t smem = (M->pos + m) * sizeof(float);
#define ATTN(DH) k_attn<DH><<<ag, 128, smem>>>(M->qkv, M->kc + l * kvl, M->vc + l * kvl, M->att, M->pos, H, Hk, \
                                              M->Tmax, M->attn_scale, M->window)
        if (Dh == 64) ATTN(64); else if (Dh == 96) ATTN(96); else if (Dh == 128) ATTN(128);
        else { fprintf(stderr, "head_dim %d unsupported\n", Dh); exit(1); }
#undef ATTN
      }
      ksync("attn");
      matmul(M, M->T(p + "o"), M->att, m, M->o, D);   // att: [m, H Dh]
      ksync("o");
      k_rmsnorm<<<m, 256>>>(M->h, M->o, (const float*)M->T(p + "ln2").dev, M->xn, D, M->eps);
      ksync("rmsnorm");
      matmul(M, M->T(p + "gate"), M->xn, m, M->g, I);
      ksync("gate");
      matmul(M, M->T(p + "up"), M->xn, m, M->u, I);
      ksync("up");
      k_silu_mul<<<std::min(1024, (m * I + 255) / 256), 256>>>(M->g, M->u, (size_t)m * I);
      ksync("silu");
      matmul(M, M->T(p + "down"), M->g, m, M->o, D);
      ksync("down");
      // Tegra (nvgpu) kills a channel whose queued work runs longer than its watchdog (~5 s); a long prefill
      // queued asynchronously trips it, so drain the queue once per layer (costs microseconds).
      if (!g_capture && !g_no_layer_sync) CK(cudaStreamSynchronize(0));
      const int ln = kept(l + 1);
      const std::string nn = ln < M->L ? "l" + std::to_string(ln) + ".ln1" : "lnf";
      k_rmsnorm<<<m, 256>>>(M->h, M->o, (const float*)M->T(nn).dev, M->xn, D, M->eps);
    }
    M->pos += m;
    const bool last = c0 + m == n;
    Tensor& lm = M->T("lm_head");
    if (!all_logits && !last && lm.dev) continue;   // intermediate chunk: no logits needed
    if (all_logits) {
      matmul(M, lm, M->xn, m, M->logits, M->V);
      if (logits && !g_capture) CK(cudaMemcpy(logits + (size_t)c0 * M->V, M->logits, (size_t)m * M->V * 4, cudaMemcpyDeviceToHost));
    } else {
      // only the last token's logits are needed; streamed lm_head must still be consumed once per chunk
      matmul(M, lm, M->xn + (size_t)(m - 1) * D, 1, M->logits, M->V);
      ksync("lm_head");
      if (last && logits && !g_capture) CK(cudaMemcpy(logits, M->logits, (size_t)M->V * 4, cudaMemcpyDeviceToHost));
    }
  }
  if (!g_capture) CK(cudaDeviceSynchronize());
  return 0;
}

// ── tests / benchmarks ──────────────────────────────────────────────────────────────────────────────────

// Y[M, N] = X[M, K] @ dequant(W)^T on the device kernels; path 0 auto, 1 fused GEMV per row, 2 decode + GEMM
int rt_test_matmul(void* hm, const char* name, const float* X, int m, float* Y, int path) {
  Model* M = (Model*)hm;
  Tensor& t = M->T(name);
  if (!t.dev) return -1;
  float *dx, *dy;
  CK(cudaMalloc(&dx, (size_t)m * t.K * 4));
  CK(cudaMalloc(&dy, (size_t)m * t.N * 4));
  CK(cudaMemcpy(dx, X, (size_t)m * t.K * 4, cudaMemcpyHostToDevice));
  if (path == 1 && t.enc == "raw8") {
    for (int r = 0; r < m; ++r) matmul(M, t, dx + (size_t)r * t.K, 1, dy + (size_t)r * t.N, t.N, 0);
  } else {
    matmul(M, t, dx, m, dy, t.N, path);
  }
  CK(cudaMemcpy(Y, dy, (size_t)m * t.N * 4, cudaMemcpyDeviceToHost));
  cudaFree(dx);
  cudaFree(dy);
  return 0;
}

// decode a resident tensor to host codes [N, K] (raw8: copy)
int rt_test_decode(void* hm, const char* name, uint8_t* out) {
  Model* M = (Model*)hm;
  Tensor& t = M->T(name);
  if (!t.dev) return -1;
  if (t.enc == "raw8") {
    CK(cudaMemcpy(out, t.dev, t.nbytes, cudaMemcpyDeviceToHost));
    return 0;
  }
  // codes q
  const int KT = t.K / (t.bits == 8 ? TK : 2 * TK), NT = t.N / TN;
  uint8_t* d;
  CK(cudaMalloc(&d, (size_t)t.N * t.K));
  if (t.il) decode_il(t, t.dev, d, 0, NT, (NT * KT + 7) / 8);
  else k_decode_huf8<<<(NT * KT + 7) / 8, 256>>>(t.dev, d, t.N, t.K, 0, NT);
  CK(cudaGetLastError());
  CK(cudaMemcpy(out, d, (size_t)t.N * t.K, cudaMemcpyDeviceToHost));
  cudaFree(d);
  return 0;
}

// GPU decode of every resident huf8 matrix into scratch (median of reps); returns seconds per full pass
double rt_gpu_decode_all(void* hm, int reps) {
  Model* M = (Model*)hm;
  std::vector<double> ts;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  for (int r = 0; r < reps; ++r) {
    CK(cudaEventRecord(e0));
    for (auto& t : M->ts) {
      if ((t.enc != "huf8" && t.enc != "huf4p") || !t.dev) continue;
      const int KT = t.K / (t.bits == 8 ? TK : 2 * TK), NT = t.N / TN;
      const int rows_fit = std::max(1, (int)(M->scratch_bytes / ((size_t)TN * t.K)));
      for (int tr0 = 0; tr0 < NT; tr0 += rows_fit) {
        const int tr1 = std::min(NT, tr0 + rows_fit);
        const int nb = ((tr1 - tr0) * KT + 7) / 8;
        if (t.il) decode_il(t, t.dev, M->scratch, tr0, tr1, nb);
        else k_decode_huf8<<<((tr1 - tr0) * KT + 7) / 8, 256>>>(t.dev, M->scratch, t.N, t.K, tr0, tr1);
      }
    }
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    ts.push_back(ms / 1e3);
  }
  std::sort(ts.begin(), ts.end());
  return ts[ts.size() / 2];
}

// CPU parallel decode of one huf8 blob (host memory) into out [N, K]; tiles interleaved across threads
// (the paper's setup: Huffman segments decoded on the CPU cores). Returns decode seconds.
static double cpu_decode(const uint8_t* blob, int N, int K, int threads, uint8_t* out) {
  const int KT = K / TK, T = (N / TN) * KT;
  const uint16_t* lut = (const uint16_t*)blob;
  const uint32_t* starts = (const uint32_t*)(blob + 2 * LUTN);
  const uint8_t* payload = (const uint8_t*)(starts + (size_t)T * 32);
  auto work = [&](int tid) {
    for (int tt = tid; tt < T; tt += threads) {
      const int tr = tt / KT, tc = tt - tr * KT;
      for (int lane = 0; lane < 32; ++lane) {
        uint64_t bp = starts[(size_t)tt * 32 + lane];
        for (int half = 0; half < 2; ++half) {
          uint8_t* row = out + (size_t)(tr * TN + half * 32 + lane) * K + tc * TK;
          for (int c = 0; c < TK; ++c) {
            uint64_t w;
            memcpy(&w, payload + ((bp >> 5) << 2), 8);
            const uint16_t e = lut[(w >> (bp & 31)) & (LUTN - 1)];
            row[c] = (uint8_t)e;
            bp += e >> 8;
          }
        }
      }
    }
  };
  const double t0 = now_s();
  std::vector<std::thread> th;
  for (int i = 0; i < threads; ++i) th.emplace_back(work, i);
  for (auto& x : th) x.join();
  return now_s() - t0;
}

int rt_test_cpu_decode(void* hm, const char* name, int threads, uint8_t* out) {
  Model* M = (Model*)hm;
  Tensor& t = M->T(name);
  if (!coded(t) || t.pre_il) return -1;   // symbols [N, K / (8 / bits)] (pre-interleaved bundles: not supported)
  if (t.enc[0] == 'p') {       // packed: the stored bytes are the symbols
    pread_all(M->fd, out, t.nbytes, t.off);
    return 0;
  }
  std::vector<uint8_t> blob(t.nbytes + 16);
  pread_all(M->fd, blob.data(), t.nbytes, t.off);
  cpu_decode(blob.data(), t.N, t.K / (8 / t.bits), threads, out);
  return 0;
}

// CPU decode of every huf8 matrix; blobs are read from the bundle first (not timed). Returns decode seconds.
double rt_cpu_decode_all(void* hm, int threads, long long* symbols) {
  Model* M = (Model*)hm;
  double total = 0;
  long long nsym = 0;
  std::vector<uint8_t> blob, out;
  for (auto& t : M->ts) {
    if ((t.enc != "huf8" && t.enc != "huf4p") || t.pre_il) continue;
    blob.resize(t.nbytes + 16);
    pread_all(M->fd, blob.data(), t.nbytes, t.off);
    out.resize((size_t)t.N * t.K);
    total += cpu_decode(blob.data(), t.N, t.K / (8 / t.bits), threads, out.data());
    nsym += (long long)t.N * t.K / (8 / t.bits);
  }
  if (symbols) *symbols = nsym;
  return total;
}

}  // extern "C"

// ── CLI ─────────────────────────────────────────────────────────────────────────────────────────────────
#ifdef PHI3RT_MAIN
static std::vector<int> parse_ids(const std::string& s) {
  std::vector<int> v;
  std::istringstream is(s);
  int x;
  while (is >> x) v.push_back(x);
  return v;
}

// layer list for --skip: "odd", "even", "a-b" ranges and single layers, comma-separated (e.g. "odd", "12-19,25")
static std::vector<int> parse_skip(const std::string& s, int L) {
  std::vector<int> v;
  std::stringstream ss(s);
  std::string it;
  while (std::getline(ss, it, ',')) {
    if (it == "odd" || it == "even") {
      for (int l = it == "odd"; l < L; l += 2) v.push_back(l);
    } else if (it.find('-') != std::string::npos) {
      for (int l = atoi(it.c_str()); l <= atoi(it.c_str() + it.find('-') + 1); ++l) v.push_back(l);
    } else if (!it.empty()) {
      v.push_back(atoi(it.c_str()));
    }
  }
  return v;
}

// --drafter ngramcache: llama.cpp's prompt-lookup drafter (common/ngram-cache.{h,cpp} at 5d7793d, as llama-lookup
// runs it with no static / dynamic cache files), ported so both runtimes draft alike: a context cache of n-grams
// (n = 1..4) of the prompt and every accepted token -> next-token counts; each draft token from the longest n-gram whose
// most frequent continuation passes the lax thresholds (sample size / share). Same containers and hashes as llama.cpp,
// so equal counts break ties the same way.
struct NgKey {
  int t[4];
  bool operator==(const NgKey& o) const { return t[0] == o.t[0] && t[1] == o.t[1] && t[2] == o.t[2] && t[3] == o.t[3]; }
};
struct NgTokHash {
  size_t operator()(int tok) const { return tok * 11400714819323198485llu; }
};
struct NgKeyHash {
  size_t operator()(const NgKey& k) const {
    size_t h = NgTokHash{}(k.t[0]);
    for (int i = 1; i < 4; ++i) h ^= NgTokHash{}(k.t[i]);
    return h;
  }
};
typedef std::unordered_map<int, int32_t> NgPart;
typedef std::unordered_map<NgKey, NgPart, NgKeyHash> NgCache;

static NgKey ng_key(const int* x, int n) {
  NgKey k;
  for (int i = 0; i < 4; ++i) k.t[i] = i < n ? x[i] : -1;
  return k;
}

static void ng_update(NgCache& c, const std::vector<int>& inp, long S, int nnew) {   // common_ngram_cache_update on
  // inp[0, S), n = 1..4
  for (long n = 1; n <= 4; ++n)
    for (long i = std::max(S - nnew, n); i < S; ++i) ++c[ng_key(&inp[i - n], (int)n)][inp[i]];
}

static void ng_draft(const NgCache& c, const std::vector<int>& inp, std::vector<int>& d, int kmax) {
  static const int min_n[4] = {2, 2, 1, 1}, min_pct[4] = {66, 50, 50, 50};   // draft_min_*_lax
  const int S = (int)inp.size();
  if (S < 2) return;
  auto at = [&](int i) { return i < S ? inp[i] : d[i - S]; };
  while ((int)d.size() < kmax) {
    int tok = -1;
    for (int n = 4; n >= 1 && tok < 0; --n) {
      int g[4];
      for (int j = 0; j < n; ++j) g[j] = at(S + (int)d.size() - n + j);
      const auto it = c.find(ng_key(g, n));
      if (it == c.end()) continue;
      int best = 0, sum = 0, arg = -1;
      for (const auto& tc : it->second) {   // empty static cache: count_static = 1 for every token
        if (tc.second > best) { best = tc.second; arg = tc.first; }
        sum += tc.second;
      }
      if (sum < min_n[n - 1] || 100 * best < min_pct[n - 1] * sum) continue;
      tok = arg;
    }
    if (tok < 0) break;
    d.push_back(tok);
  }
}

static int argmax(const float* x, int n) {
  int b = 0;
  for (int i = 1; i < n; ++i)
    if (x[i] > x[b]) b = i;
  return b;
}

static void usage() {
  fprintf(stderr,
          "usage: phi3rt <bundle> <cmd> [options]\n"
          "  serve                 read prompt token ids (one line per prompt) on stdin, print generated ids\n"
          "  bench                 Table-3 style timings: prefill, first token, per-token generation\n"
          "  decode-bench          GPU and CPU (--threads) decode of every huf8 matrix\n"
          "  check                 every huf8 matrix: GPU decode == CPU reference decode; fused m-token GEMM vs\n"
          "                        decode + GEMM for m = 1..8\n"
          "  fwd-bench             forward-pass time for m = 1..8 tokens at position --prompt-len\n"
          "  graph-bench           fwd-bench direct vs CUDA-graph replay (build with --default-stream per-thread)\n"
          "  graph-check           LLMRT_GRAPHS=1 logits == direct logits, bit for bit, m = 1..8\n"
          "  export-il OUT [padded] copy of the bundle with coded matrices stored interleaved (compact unless padded),\n"
          "                        so they can be streamed (bundles larger than memory)\n"
          "  spec                  stdin lines \"<max_new> <id> ...\": greedy plain vs speculative, one JSON line per\n"
          "                        prompt; --drafter lookup (--ngram N) | self (--skip odd|a-b,...) | model\n"
          "                        (--draft-bundle DIR); --draft K max draft tokens, --tau p early stop\n"
          "options: --ctx N (512) --batch N (128) --budget-mb N (-1 = floor-limited) --floor-mb N (384)\n"
          "         --prompt-len N (128) --gen N (64) --threads N (4) --reps N (3)\n"
          "         --window N (0 = full attention as in the ONNX graph; 2047 = Phi-3's trained sliding window)\n");
}

// export-il: copy of a bundle with every coded matrix stored in its device layout (interleaved lane words; huf8 /
// huf4p compact unless "padded"), every tensor 4096-aligned (O_DIRECT streaming reads it in place). Host only.
static int export_il(const char* dir, const char* out, bool compact) {
  std::ifstream mf(std::string(dir) + "/manifest.txt");
  if (!mf) { fprintf(stderr, "cannot open %s/manifest.txt\n", dir); return 1; }
  if (mkdir(out, 0755) && errno != EEXIST) { perror(out); return 1; }
  const int fd = open((std::string(dir) + "/weights.bin").c_str(), O_RDONLY);
  FILE* fw = fopen((std::string(out) + "/weights.bin").c_str(), "wb");
  std::ofstream mo(std::string(out) + "/manifest.txt");
  if (fd < 0 || !fw || !mo) { perror("export-il"); return 1; }
  size_t off = 0, in_b = 0, out_b = 0;
  std::vector<uint8_t> blob;
  static const char zero[4096] = {0};
  for (std::string line; std::getline(mf, line);) {
    std::istringstream is(line);
    std::vector<std::string> f;
    for (std::string x; is >> x;) f.push_back(x);
    if (f.size() != 9 || f[0] != "t") { mo << line << "\n"; continue; }
    Tensor t;
    t.enc = f[2]; t.N = atoi(f[3].c_str()); t.K = atoi(f[4].c_str());
    t.off = strtoull(f[7].c_str(), nullptr, 10); t.nbytes = strtoull(f[8].c_str(), nullptr, 10);
    const int bits = t.enc == "huf4p" || t.enc == "pk4" ? 4 : 8;
    blob.resize(t.nbytes);
    pread_all(fd, blob.data(), t.nbytes, t.off);
    const uint8_t* data = blob.data();
    size_t n = t.nbytes;
    std::vector<uint32_t> iw;
    if (coded(t)) {
      const bool pk = t.enc[0] == 'p';
      iw = pk ? interleave_packed(blob.data(), t.N, t.K / (8 / bits))
              : interleave_huf8(blob.data(), t.nbytes, t.N, t.K / (8 / bits), compact);
      data = (const uint8_t*)iw.data();
      n = iw.size() * 4;
      f[2] = t.enc + (compact && !pk ? "_ilc" : "_il");
      in_b += t.nbytes; out_b += n;
    }
    fwrite(data, 1, n, fw);
    f[7] = std::to_string(off); f[8] = std::to_string(n);
    off += n;
    const size_t pad = (4096 - off % 4096) % 4096;
    fwrite(zero, 1, pad, fw);
    off += pad;
    for (size_t i = 0; i < f.size(); ++i) mo << f[i] << (i + 1 < f.size() ? " " : "\n");
  }
  fclose(fw);
  close(fd);
  fprintf(stderr, "export-il: coded matrices %.1f MB -> %.1f MB (%s), weights.bin %.1f MB\n", in_b / 1e6, out_b / 1e6,
          compact ? "compact" : "padded", off / 1e6);
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 3) { usage(); return 2; }
  const char* dir = argv[1];
  const std::string cmd = argv[2];
  if (cmd == "export-il") {
    if (argc < 4) { usage(); return 2; }
    return export_il(dir, argv[3], !(argc > 4 && std::string(argv[4]) == "padded"));
  }
  int ctx = 512, batch = 128, floor_mb = 384, plen = 128, gen = 64, threads = 4, reps = 3, window = 0, draft = 7,
      ngram = 3;
  long long budget = -1;
  std::string drafter = "lookup", skip_spec, draft_bundle;
  double tau = 0;
  for (int i = 3; i + 1 < argc; i += 2) {
    const std::string k = argv[i];
    if (k == "--drafter") { drafter = argv[i + 1]; continue; }
    if (k == "--skip") { skip_spec = argv[i + 1]; continue; }
    if (k == "--draft-bundle") { draft_bundle = argv[i + 1]; continue; }
    if (k == "--tau") { tau = atof(argv[i + 1]); continue; }
    const long long v = atoll(argv[i + 1]);
    if (k == "--ctx") ctx = v; else if (k == "--batch") batch = v; else if (k == "--budget-mb") budget = v;
    else if (k == "--floor-mb") floor_mb = v; else if (k == "--prompt-len") plen = v; else if (k == "--gen") gen = v;
    else if (k == "--threads") threads = v; else if (k == "--reps") reps = v; else if (k == "--window") window = v;
    else if (k == "--draft") draft = v; else if (k == "--ngram") ngram = v;
    else { usage(); return 2; }
  }
  const double t_open = now_s();
  void* m = rt_open(dir, ctx, batch, budget, floor_mb, 1);
  if (!m) return 1;
  const double load_s = now_s() - t_open;
  Model* M = (Model*)m;
  rt_set_window(m, window);
  const int V = M->V;
  std::vector<float> logits(V);
  auto is_eos = [&](int t) { return std::find(M->eos.begin(), M->eos.end(), t) != M->eos.end(); };
  if (cmd == "serve") {
    printf("READY\n");
    fflush(stdout);
    std::string line;
    while (std::getline(std::cin, line)) {
      // line: "<max_new> <id> <id> ..."; generation is greedy; ids are streamed back as they are produced
      std::vector<int> v = parse_ids(line);
      if (v.empty()) continue;
      const int max_new = v[0];
      std::vector<int> ids(v.begin() + 1, v.end());
      rt_reset(m);
      const double t0 = now_s();
      if (rt_forward(m, ids.data(), (int)ids.size(), logits.data(), 0)) { printf("ERR\n"); fflush(stdout); continue; }
      const double t1 = now_s();
      int n = 0;
      double tg = 0;
      for (; n < max_new; ++n) {
        int tok = argmax(logits.data(), V);
        printf("%d ", tok);
        fflush(stdout);
        if (is_eos(tok) || M->pos >= M->Tmax) { ++n; break; }
        const double a = now_s();
        rt_forward(m, &tok, 1, logits.data(), 0);
        tg += now_s() - a;
      }
      printf("\nSTATS prefill_s=%.3f prompt_tokens=%zu new_tokens=%d gen_s_per_token=%.4f\n", t1 - t0, ids.size(), n,
             n > 1 ? tg / (n - 1) : 0.0);
      fflush(stdout);
    }
  } else if (cmd == "bench") {
    // prompt: deterministic token ids (content does not affect timing); greedy generation
    std::vector<int> ids(plen);
    for (int i = 0; i < plen; ++i) ids[i] = i == 0 ? 1 : 1000 + (i * 7919) % 28000;
    std::vector<double> pre, first, per;
    for (int r = 0; r < reps; ++r) {
      rt_reset(m);
      const double t0 = now_s();
      rt_forward(m, ids.data(), plen, logits.data(), 0);
      const double t1 = now_s();
      int tok = argmax(logits.data(), V);
      rt_forward(m, &tok, 1, logits.data(), 0);
      const double t2 = now_s();
      for (int i = 1; i < gen; ++i) {
        tok = argmax(logits.data(), V);
        rt_forward(m, &tok, 1, logits.data(), 0);
      }
      const double t3 = now_s();
      pre.push_back(t1 - t0);
      first.push_back(t2 - t0);
      per.push_back((t3 - t2) / std::max(1, gen - 1));
    }
    auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
    printf("{\"bundle\": \"%s\", \"enc\": \"%s\", \"prompt_len\": %d, \"gen\": %d, \"reps\": %d, \"load_s\": %.3f, "
           "\"prefill_s\": %.4f, \"first_token_s\": %.4f, \"token_gen_s\": %.5f, \"resident_bytes\": %.0f, "
           "\"streamed_bytes\": %.0f, \"n_streamed\": %.0f, \"stream_wait_s_total\": %.3f, \"mem_available_mb\": %.0f, "
           "\"ctx\": %d, \"batch\": %d}\n",
           dir, M->cfg["enc"].c_str(), plen, gen, reps, load_s, med(pre), med(first), med(per), rt_stat(m, "resident_bytes"),
           rt_stat(m, "streamed_bytes"), rt_stat(m, "n_streamed"), rt_stat(m, "stream_wait_s"),
           rt_stat(m, "mem_available_mb"), ctx, batch);
  } else if (cmd == "decode-bench") {
    const double g = rt_gpu_decode_all(m, reps);
    long long ns = 0;
    const double c = rt_cpu_decode_all(m, threads, &ns);
    printf("{\"bundle\": \"%s\", \"gpu_decode_all_s\": %.4f, \"cpu_decode_all_s\": %.4f, \"cpu_threads\": %d, "
           "\"symbols\": %lld}\n", dir, g, c, threads, ns);
  } else if (cmd == "check") {
    int bad = 0;
    for (auto& t : M->ts) {
      if (!coded(t) || !t.dev) continue;
      std::vector<uint8_t> a((size_t)t.N * t.K), b((size_t)t.N * t.K);
      rt_test_decode(m, t.name.c_str(), a.data());
      rt_test_cpu_decode(m, t.name.c_str(), threads, b.data());
      if (t.il) {   // CPU reference symbols -> codes q (as the GPU decode)
        std::vector<uint8_t> sym(b.begin(), b.begin() + (size_t)t.N * t.K / (8 / t.bits));
        const int Ks = t.K / (8 / t.bits), A = 1 << t.bits;
        for (int n = 0; n < t.N; ++n)
          for (int k = 0; k < Ks; ++k) {
            const int e = sym[(size_t)n * Ks + k];
            for (int i = 0; i < 8 / t.bits; ++i) {
              const int r = t.bits == 8 ? e : (e >> (4 * i)) & 15;
              const int col = k * (8 / t.bits) + i;
              const int q = t.kqt ? r : (r + t.h_radd[(size_t)n * t.G + col / (t.K / t.G)]) & (A - 1);
              b[(size_t)n * t.K + k * (8 / t.bits) + i] = (uint8_t)q;
            }
          }
      }
      const bool ok = a == b;
      bad += !ok;
      double worst = 0;
      if (t.name == "l0.qkv" || t.name == "l0.v" || t.name == "l0.q" || t.name == "l7.down" || t.name == "l" + std::to_string(M->L - 1) + ".gate" ||
          t.name == "lm_head") {
        std::vector<float> X((size_t)8 * t.K), Y0((size_t)8 * t.N), Y1((size_t)8 * t.N);
        srand(7);
        for (auto& v : X) v = (float)rand() / RAND_MAX - 0.5f;
        rt_test_matmul(m, t.name.c_str(), X.data(), 8, Y0.data(), 2);
        for (int mm = 1; mm <= 8; ++mm) {
          rt_test_matmul(m, t.name.c_str(), X.data(), mm, Y1.data(), 0);
          double num = 0, den = 0;
          for (size_t i = 0; i < (size_t)mm * t.N; ++i) {
            num = std::max(num, (double)fabsf(Y1[i] - Y0[i]));
            den = std::max(den, (double)fabsf(Y0[i]));
          }
          worst = std::max(worst, num / den);
        }
      }
      printf("{\"tensor\": \"%s\", \"interleaved\": %d, \"decode_exact\": %d, \"gemm_m1_8_max_rel_err\": %.3g}\n",
             t.name.c_str(), (int)t.il, (int)ok, worst);
    }
    printf("{\"check_failures\": %d}\n", bad);
  } else if (cmd == "attn-check") {
    // k_attn_mq vs the per-query k_attn: stdin line "<max_new> <id> ..." (spec format), prefill, then m = 1..8 tokens of
    // the prompt's tail re-fed at the end; max |logit difference| and argmax agreement per m.
    std::string line;
    std::getline(std::cin, line);
    std::vector<int> v = parse_ids(line);
    const std::vector<int> prompt(v.begin() + 1, v.end());
    const int P = (int)prompt.size() - 8;
    std::vector<float> a((size_t)8 * V), b((size_t)8 * V);
    rt_reset(m);
    rt_forward(m, prompt.data(), P, logits.data(), 0);
    for (int mm = 1; mm <= 8; ++mm) {
      g_attn_old = true;
      rt_set_pos(m, P);
      rt_forward(m, prompt.data() + P, mm, a.data(), 1);
      g_attn_old = false;
      rt_set_pos(m, P);
      rt_forward(m, prompt.data() + P, mm, b.data(), 1);
      double dmax = 0, lmax = 0;
      int agree = 0;
      for (int i = 0; i < mm; ++i) {
        for (int j = 0; j < V; ++j) {
          dmax = std::max(dmax, (double)fabsf(a[(size_t)i * V + j] - b[(size_t)i * V + j]));
          lmax = std::max(lmax, (double)fabsf(a[(size_t)i * V + j]));
        }
        agree += argmax(&a[(size_t)i * V], V) == argmax(&b[(size_t)i * V], V);
      }
      printf("{\"m\": %d, \"max_abs_diff\": %.3e, \"max_abs_logit\": %.2f, \"argmax_agree\": %d}\n", mm, dmax, lmax,
             agree);
    }
  } else if (cmd == "fwd-bench") {
    std::vector<int> ids(plen + 8);
    for (int i = 0; i < plen + 8; ++i) ids[i] = i == 0 ? 1 : 1000 + (i * 7919) % 28000;
    std::vector<float> lg((size_t)8 * V);
    rt_reset(m);
    rt_forward(m, ids.data(), plen, logits.data(), 0);
    for (int mm = 1; mm <= 8; ++mm) {
      std::vector<double> ts;
      for (int r = 0; r < reps + 1; ++r) {
        rt_set_pos(m, plen);
        const double a = now_s();
        rt_forward(m, ids.data() + plen, mm, lg.data(), 1);
        if (r) ts.push_back(now_s() - a);   // first rep: warm-up
      }
      std::sort(ts.begin(), ts.end());
      printf("{\"bundle\": \"%s\", \"m\": %d, \"pos\": %d, \"forward_s\": %.5f, \"per_token_s\": %.5f, \"act8\": %d}\n",
             dir, mm, plen, ts[ts.size() / 2], ts[ts.size() / 2] / mm, (int)((Model*)m)->act8);
      fflush(stdout);
    }
  } else if (cmd == "graph-bench") {
    // launch overhead: one forward pass of m tokens launched kernel by kernel (with and without the per-layer drain)
    // vs replayed as a CUDA graph. Timing only: a graph keeps the position and shapes it was captured with.
    if (!M->stream_list.empty()) { fprintf(stderr, "graph-bench needs resident weights\n"); return 1; }
    std::vector<int> ids(plen + 8);
    for (int i = 0; i < plen + 8; ++i) ids[i] = i == 0 ? 1 : 1000 + (i * 7919) % 28000;
    rt_reset(m);
    rt_forward(m, ids.data(), plen, logits.data(), 0);
    auto med = [](std::vector<double>& v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
    for (int mm = 1; mm <= 8; ++mm) {
      std::vector<double> td, tn, tg;
      for (int r = 0; r < reps + 1; ++r) {
        for (int ns = 0; ns < 2; ++ns) {
          g_no_layer_sync = ns;
          rt_set_pos(m, plen);
          const double a = now_s();
          rt_forward(m, ids.data() + plen, mm, nullptr, 1);
          if (r) (ns ? tn : td).push_back(now_s() - a);   // first rep: warm-up
        }
      }
      g_no_layer_sync = false;
      rt_set_pos(m, plen);
      CK(cudaMemcpy(M->ids, ids.data() + plen, mm * 4, cudaMemcpyHostToDevice));
      cudaGraph_t g;
      cudaGraphExec_t ge;
      g_capture = true;
      CK(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeRelaxed));
      rt_forward(m, ids.data() + plen, mm, nullptr, 1);
      CK(cudaStreamEndCapture(cudaStreamPerThread, &g));
      g_capture = false;
      size_t nodes = 0;
      CK(cudaGraphGetNodes(g, nullptr, &nodes));
#if CUDART_VERSION >= 11040
      CK(cudaGraphInstantiate(&ge, g, 0));
#else
      CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
#endif
      for (int r = 0; r < reps + 1; ++r) {
        const double a = now_s();
        CK(cudaGraphLaunch(ge, cudaStreamPerThread));
        CK(cudaStreamSynchronize(cudaStreamPerThread));
        if (r) tg.push_back(now_s() - a);
      }
      CK(cudaGraphExecDestroy(ge));
      CK(cudaGraphDestroy(g));
      rt_set_pos(m, plen);
      printf("{\"bundle\": \"%s\", \"m\": %d, \"pos\": %d, \"kernels\": %zu, \"direct_s\": %.5f, "
             "\"no_layer_sync_s\": %.5f, \"graph_s\": %.5f, \"act8\": %d}\n",
             dir, mm, plen, nodes, med(td), med(tn), med(tg), (int)((Model*)m)->act8);
      fflush(stdout);
    }
  } else if (cmd == "graph-check") {
    // LLMRT_GRAPHS path == direct path, bit for bit: m = 1..8 at --prompt-len and 300 positions later (another
    // score-buffer bucket), all_logits on and off; each shape runs direct, then 3 graph-mode passes (first
    // direct, capture + replay, replay)
#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
    fprintf(stderr, "graph-check needs a build with --default-stream per-thread\n");
    return 1;
#else
    const int P2 = std::min(M->Tmax - 9, plen + 300);
    std::vector<int> ids(P2 + 8);
    for (int i = 0; i < P2 + 8; ++i) ids[i] = i == 0 ? 1 : 1000 + (i * 7919) % 28000;
    std::vector<float> a((size_t)8 * V), b((size_t)8 * V);
    rt_reset(m);
    rt_forward(m, ids.data(), P2, logits.data(), 0);
    int bad = 0, shapes = 0;
    for (int P : {P2, plen})   // descending: rt_set_pos only rewinds
      for (int al = 0; al < 2; ++al)
        for (int mm = 1; mm <= 8; ++mm) {
          g_graphs = false;
          rt_set_pos(m, P);
          rt_forward(m, ids.data() + P, mm, a.data(), al);
          g_graphs = !(getenv("LLMRT_GC_DIRECT") && atoi(getenv("LLMRT_GC_DIRECT")));   // check the check
          const size_t nl = al ? (size_t)mm * V : (size_t)V;
          int diff = 0;
          for (int r = 0; r < 3; ++r) {
            rt_set_pos(m, P);
            std::fill(b.begin(), b.end(), NAN);
            rt_forward(m, ids.data() + P, mm, b.data(), al);
            diff += memcmp(a.data(), b.data(), nl * 4) != 0;
          }
          bad += diff > 0;
          ++shapes;
          printf("{\"pos\": %d, \"m\": %d, \"all_logits\": %d, \"mismatching_passes\": %d}\n", P, mm, al, diff);
        }
    printf("{\"bundle\": \"%s\", \"shapes\": %d, \"bit_exact\": %d, \"graph_captures\": %d, "
           "\"graph_launches\": %d}\n", dir, shapes, shapes - bad, M->n_graph_capture, M->n_graph_launch);
    if (bad) return 2;
#endif
  } else if (cmd == "spec") {
    // drafters: lookup (prompt lookup), self (this model with layers --skip skipped), model (--draft-bundle, e.g. the
    // same bundle opened with PHI3SPEC_ROUND_BITS). Model drafters draft up to --draft tokens greedily and stop early
    // when the drafter's top probability drops below --tau.
    std::vector<char> skipmask;
    if (drafter == "self") {
      skipmask.assign(M->L, 0);
      for (int l : parse_skip(skip_spec, M->L)) skipmask[l] = 1;
    }
    void* dm = nullptr;
    if (drafter == "model") {
      if (draft_bundle.empty()) { usage(); return 2; }
      dm = rt_open(draft_bundle.c_str(), ctx, batch, -1, floor_mb, 1);
      if (!dm) return 1;
      rt_set_window(dm, window);
    }
    int nskip = 0;
    for (char c : skipmask) nskip += c;
    std::string line;
    std::vector<float> lg((size_t)(draft + 1) * V), dl(V);
    auto top_p = [&](const float* x, int* arg) {   // argmax and its softmax probability
      const int k = argmax(x, V);
      double z = 0;
      for (int i = 0; i < V; ++i) z += exp((double)x[i] - x[k]);
      *arg = k;
      return 1.0 / z;
    };
    while (std::getline(std::cin, line)) {
      std::vector<int> v = parse_ids(line);
      if (v.size() < 2) continue;
      const int max_new = v[0];
      const std::vector<int> prompt(v.begin() + 1, v.end());
      // plain greedy
      rt_reset(m);
      const double e_pre0 = epoch_s();
      double t0 = now_s();
      rt_forward(m, prompt.data(), (int)prompt.size(), logits.data(), 0);
      const double prefill = now_s() - t0;
      const double e_pre1 = epoch_s();
      std::vector<int> plain;
      t0 = now_s();
      int tok = argmax(logits.data(), V);
      while ((int)plain.size() < max_new) {
        plain.push_back(tok);
        if (is_eos(tok) || M->pos >= M->Tmax - 1) break;
        rt_forward(m, &tok, 1, logits.data(), 0);
        tok = argmax(logits.data(), V);
      }
      const double plain_s = now_s() - t0;
      const double e_plain1 = epoch_s();
      // speculative
      rt_reset(m);
      rt_forward(m, prompt.data(), (int)prompt.size(), logits.data(), 0);
      std::vector<int> seq = prompt, out, dseq;   // dseq: tokens the model drafter has ingested
      NgCache ngc;   // --drafter ngramcache: prompt n-grams now, each accepted token as it is appended
      size_t ng_done = seq.size();
      if (drafter == "ngramcache") ng_update(ngc, seq, (long)seq.size(), (int)seq.size());
      if (dm) {
        rt_reset(dm);
        rt_forward(dm, prompt.data(), (int)prompt.size(), nullptr, 0);
        dseq = prompt;
      }
      const double e_spec0 = epoch_s();
      t0 = now_s();
      double t_draft = 0;
      tok = argmax(logits.data(), V);
      int passes = 0, drafted = 0, accepted = 0, dfwd = 0, dfwd_tok = 0;
      bool done = false;
      while (!done && (int)out.size() < max_new) {
        out.push_back(tok);
        seq.push_back(tok);
        if (is_eos(tok) || (int)out.size() >= max_new) break;
        const int pos0 = M->pos;
        const int room = std::min(max_new - (int)out.size(), M->Tmax - 1 - M->pos);
        const int kmax = std::max(0, std::min(draft, room - 1));
        std::vector<int> d;
        const double td = now_s();
        if (drafter == "ngramcache") {
          for (; ng_done < seq.size(); ++ng_done) ng_update(ngc, seq, (long)ng_done + 1, 1);   // one token at a time,
                                                                                              // as llama-lookup does
          ng_draft(ngc, seq, d, kmax);
        } else if (drafter == "lookup") {
          const int S = (int)seq.size();
          for (int n = std::min(ngram, S - 1); n >= 1 && d.empty(); --n)
            for (int p = S - n - 1; p >= 0; --p) {
              if (!std::equal(seq.begin() + p, seq.begin() + p + n, seq.end() - n)) continue;
              for (int i = p + n; i < S && (int)d.size() < kmax; ++i) d.push_back(seq[i]);
              break;
            }
        } else if (kmax > 0) {
          void* h = dm ? dm : m;
          std::vector<int> feed;
          if (dm) {   // catch the drafter up: rewind to the common prefix, feed the rest of seq
            size_t c = 0;
            while (c < dseq.size() && c < seq.size() && dseq[c] == seq[c]) ++c;
            rt_set_pos(dm, (int)c);
            dseq.resize(c);
            feed.assign(seq.begin() + c, seq.end());
          } else {
            feed.assign(1, tok);
            M->skip = skipmask;
          }
          while ((int)d.size() < kmax) {
            rt_forward(h, feed.data(), (int)feed.size(), dl.data(), 0);
            ++dfwd;
            dfwd_tok += (int)feed.size();
            if (dm) dseq.insert(dseq.end(), feed.begin(), feed.end());
            int a;
            const double pr = top_p(dl.data(), &a);
            if (tau > 0 && pr < tau && !d.empty()) break;
            d.push_back(a);
            if (is_eos(a)) break;
            feed.assign(1, a);
          }
          if (!dm) {
            M->skip.clear();
            rt_set_pos(m, pos0);
          }
        }
        t_draft += now_s() - td;
        std::vector<int> in(1, tok);
        in.insert(in.end(), d.begin(), d.end());
        rt_forward(m, in.data(), (int)in.size(), lg.data(), 1);
        ++passes;
        drafted += (int)d.size();
        int a = 0;
        tok = argmax(lg.data(), V);
        while (a < (int)d.size() && d[a] == tok) {
          out.push_back(tok);
          seq.push_back(tok);
          ++a;
          if (is_eos(tok)) { done = true; break; }
          tok = argmax(lg.data() + (size_t)a * V, V);
        }
        accepted += a;
        rt_set_pos(m, pos0 + 1 + a);
      }
      const double spec_s = now_s() - t0;
      const double e_spec1 = epoch_s();
      if ((int)out.size() > max_new) out.resize(max_new);
      int agree = 0;
      while (agree < (int)std::min(out.size(), plain.size()) && out[agree] == plain[agree]) ++agree;
      printf("{\"act8\": %d, \"bundle\": \"%s\", \"drafter\": \"%s\", \"skip\": \"%s\", \"n_skip\": %d, \"draft_bundle\": \"%s\", "
             "\"round_bits\": \"%s\", \"tau\": %.3f, \"prompt_tokens\": %zu, \"draft\": %d, \"ngram\": %d, "
             "\"prefill_s\": %.3f, \"plain_tokens\": %zu, \"plain_s_per_token\": %.5f, \"spec_tokens\": %zu, "
             "\"spec_s_per_token\": %.5f, \"draft_s\": %.3f, \"passes\": %d, \"tokens_per_pass\": %.3f, \"drafted\": %d, "
             "\"accepted\": %d, \"accept_rate\": %.3f, \"draft_forwards\": %d, \"draft_forward_tokens\": %d, "
             "\"speedup\": %.3f, \"identical_prefix\": %d, \"identical\": %d, \"t_prefill\": [%.3f, %.3f], "
             "\"t_plain\": [%.3f, %.3f], \"t_spec\": [%.3f, %.3f], \"out\": [",
             (int)((Model*)m)->act8, dir, drafter.c_str(), skip_spec.c_str(), nskip, draft_bundle.c_str(),
             getenv("PHI3SPEC_ROUND_BITS") ? getenv("PHI3SPEC_ROUND_BITS") : "", tau, prompt.size(), draft, ngram,
             prefill, plain.size(), plain_s / plain.size(), out.size(), spec_s / out.size(), t_draft, passes,
             (double)(out.size() - 1) / std::max(1, passes), drafted, accepted, (double)accepted / std::max(1, drafted),
             dfwd, dfwd_tok, (plain_s / plain.size()) / (spec_s / out.size()), agree, (int)(out == plain), e_pre0,
             e_pre1, e_pre1, e_plain1, e_spec0, e_spec1);
      for (size_t i = 0; i < out.size(); ++i) printf(i ? ", %d" : "%d", out[i]);
      printf("]}\n");
      fflush(stdout);
    }
    if (dm) rt_close(dm);
  } else {
    usage();
    return 2;
  }
  rt_close(m);
  return 0;
}
#endif
