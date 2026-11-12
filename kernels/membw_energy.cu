// A5 energy model input: energy per byte read from DRAM (memory controller + PHY + DRAM), measured on a Jetson as the
// difference of two slopes of module input power vs read bandwidth: the same streaming read kernel on a buffer far
// larger than L2 (every byte from DRAM) and on a buffer that stays in L2, each throttled to several bandwidths with
// __nanosleep (sm_70+; a clock64 spin on the Jetson Nano, whose 256 KiB L2 takes a 0.125 MiB buffer) (same
// instructions per byte). Loads are __ldcg (L2 only, no L1), 16 bytes per thread per iteration.
//   membw_energy <buffer MiB, fractional ok> <sleep ns> <seconds>   -> one JSON line (wall-clock start / end for the tegrastats log)
#include <cuda_runtime.h>
#include <sys/time.h>

#include <cstdio>
#include <cstdlib>

__global__ void rd(const uint4* __restrict__ p, size_t n, int iters, unsigned sleep_ns, unsigned* out) {
  uint4 acc = make_uint4(0, 0, 0, 0);
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (int k = 0; k < iters; ++k) {
    const uint4 v = __ldcg(p + i);
    acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    i += stride;
    if (i >= n) i -= n;
    if (sleep_ns) {
#if __CUDA_ARCH__ >= 700
      __nanosleep(sleep_ns);
#else   // Jetson Nano (sm_53): spin on the SM clock (~1 cycle per ns at the 921.6 MHz MAXN GPU clock)
      const long long t0 = clock64();
      while (clock64() - t0 < (long long)sleep_ns) {}
#endif
    }
  }
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x9e3779b9u) out[0] = 1;   // keep the loads
}

static double now() {
  timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + 1e-6 * tv.tv_usec;
}

int main(int argc, char** argv) {
  const double mib = argc > 1 ? atof(argv[1]) : 512;
  const unsigned sl = argc > 2 ? atoi(argv[2]) : 0;
  const double secs = argc > 3 ? atof(argv[3]) : 5.0;
  const size_t n = (size_t)(mib * (1 << 20)) / sizeof(uint4);
  uint4* p;
  unsigned* o;
  cudaMalloc(&p, n * sizeof(uint4));
  cudaMalloc(&o, 4);
  cudaMemset(p, 0x5a, n * sizeof(uint4));
  int sms;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  const int blocks = sms * 8, threads = 256, iters = 4096;
  const double bytes_per_launch = (double)blocks * threads * iters * sizeof(uint4);
  rd<<<blocks, threads>>>(p, n, iters, sl, o);   // warm-up
  cudaDeviceSynchronize();
  const double t0 = now();
  long launches = 0;
  while (now() - t0 < secs) {
    for (int j = 0; j < 8; ++j) rd<<<blocks, threads>>>(p, n, iters, sl, o);
    cudaDeviceSynchronize();
    launches += 8;
  }
  const double t1 = now();
  const cudaError_t e = cudaGetLastError();
  printf("{\"buffer_mib\": %g, \"sleep_ns\": %u, \"t_start\": %.3f, \"t_end\": %.3f, \"bytes\": %.6e, "
         "\"gbps\": %.3f, \"err\": \"%s\"}\n",
         mib, sl, t0, t1, bytes_per_launch * launches, bytes_per_launch * launches / (t1 - t0) / 1e9,
         cudaGetErrorString(e));
  return e != cudaSuccess;
}
