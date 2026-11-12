// Verilator test of hw/huff/fpga/nexys_top.sv huff_fpga_core through its UART pins (CPB from the build, e.g. 4):
// the same command sequence as scripts/fpga_host.py on one job file (hw/huff/tb/slice_main.cpp); checks the decoded
// symbols and the cycles / stalls / words against the C model's.   tb_core <job> -> one JSON line, rc 0 if ok.
#include "Vhuff_fpga_core.h"
#include "verilated.h"
#ifdef TRACE
#include "Vhuff_fpga_core___024root.h"
#endif

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <vector>

#ifndef CPB
#define CPB 4
#endif

static Vhuff_fpga_core* top;
static uint64_t cyc = 0;
static std::deque<int> rxq;          // bits to send (one entry per bit)
static int rx_cnt = 0;
static std::vector<uint8_t> got;     // bytes received from the core
static int tx_state = 0, tx_cnt = 0, tx_bit = 0, tx_byte = 0;

static void tick() {
  // drive the core's RX line
  if (rx_cnt == 0 && !rxq.empty()) rx_cnt = CPB;
  top->uart_rx_i = rxq.empty() ? 1 : rxq.front();
  if (rx_cnt && --rx_cnt == 0) rxq.pop_front();
  top->clk = 1; top->eval(); top->clk = 0; top->eval();
  ++cyc;
  // sample the core's TX line
  int tx = top->uart_tx_o;
  if (tx_state == 0) { if (!tx) { tx_state = 1; tx_cnt = CPB + CPB / 2; tx_bit = 0; tx_byte = 0; } }
  else if (--tx_cnt == 0) {
    if (tx_bit < 8) { tx_byte |= tx << tx_bit; ++tx_bit; tx_cnt = CPB; }
    else { got.push_back((uint8_t)tx_byte); tx_state = 0; }
  }
}
static void send(const void* p, size_t n) {
  const uint8_t* b = (const uint8_t*)p;
  for (size_t i = 0; i < n; ++i) {
    rxq.push_back(0);
    for (int k = 0; k < 8; ++k) rxq.push_back((b[i] >> k) & 1);
    rxq.push_back(1);
  }
}
static void send8(uint8_t v) { send(&v, 1); }
static void send32(uint32_t v) { send(&v, 4); }
static bool expect(size_t n, uint64_t limit = 400000000ull) {
  size_t want = got.size() + n;
  uint64_t c0 = cyc;
  while (got.size() < want && cyc - c0 < limit) tick();
  return got.size() >= want;
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  FILE* f = fopen(argv[1], "rb");
  std::vector<uint8_t> J;
  for (int c; (c = fgetc(f)) != EOF;) J.push_back((uint8_t)c);
  fclose(f);
  auto r32 = [&](size_t o) { uint32_t v; memcpy(&v, &J[o], 4); return v; };
  uint32_t nt = r32(8), nw = r32(12), cc = r32(16), cst = r32(20), cw = r32(24);
  size_t o = 36;
  const uint8_t* lut = &J[o]; o += 4096;
  const uint8_t* srt = &J[o]; o += 256;
  const uint8_t* cmp = &J[o]; o += 66;
  const uint8_t* desc = &J[o]; o += (size_t)nt * 32 * 8;
  const uint8_t* words = &J[o]; o += (size_t)nw * 4;
  const uint8_t* expv = &J[o];
  top = new Vhuff_fpga_core;
  top->rst_in = 1; top->uart_rx_i = 1;
  for (int i = 0; i < 10; ++i) tick();
  top->rst_in = 0;
  for (int i = 0; i < 10; ++i) tick();
  got.clear(); tx_state = 0;   // anything before the first command is line noise (the host flushes it too)
  bool ok = true;
  auto ack = [&](const char* what) {
    size_t i = got.size();
    bool a = expect(1, rxq.size() * CPB + 100000);
    fprintf(stderr, "%-6s %s  cycle %llu\n", what, a && got[i] == 0xAC ? "ack" : "NO ACK", (unsigned long long)cyc);
    if (!a || got[i] != 0xAC) { ok = false; }
  };
  send8(0x08); if (!expect(2, 100000) || got[0] != 0x5A) { fprintf(stderr, "ping failed (%zu bytes)\n", got.size()); return 1; }
  fprintf(stderr, "ping   ok   cycle %llu\n", (unsigned long long)cyc);
  send8(0x01); send(lut, 4096); ack("lut");
  send8(0x02); send(srt, 256); ack("srt");
  send8(0x03); send(cmp, 66); ack("cmp");
  send8(0x04); send8((uint8_t)nt); send(desc, (size_t)nt * 32 * 8); ack("desc");
  send8(0x05); send32(nw); send(words, (size_t)nw * 4); ack("words");
  send8(0x06); send8((uint8_t)nt);
  size_t i0 = got.size();
#ifdef TRACE   // per cycle of the run: state, busy, requests / responses / outputs, lane 0's tile and descriptor
  {
    auto* r = top->rootp;
    int on = 0;
    for (int k = 0; k < 40000 && got.size() == i0; ++k) {
      tick();
      if (r->huff_fpga_core__DOT__run) ++on;
      static long long ng[32], nr[32], ngr[32]; static int lastbad = -1;
      if (r->huff_fpga_core__DOT__run) {
        uint8_t rv = r->huff_fpga_core__DOT__req_valid; uint64_t rl = r->huff_fpga_core__DOT__req_lane;
        uint32_t seen = 0;
        for (int p = 0; p < 8; ++p) if ((rv >> p) & 1) { int l = (rl >> (5 * p)) & 31; ++ng[l]; if ((seen >> l) & 1) fprintf(stderr, "c%d: lane %d granted twice\n", on, l); seen |= 1u << l; }
        uint32_t g = r->huff_fpga_core__DOT__u_eng__DOT__grant;
        if (g != seen && lastbad < 0) { lastbad = on; fprintf(stderr, "c%d: grant %08x != req lanes %08x\n", on, g, seen); }
        for (int l = 0; l < 32; ++l) { nr[l] += (r->huff_fpga_core__DOT__resp_valid >> l) & 1; ngr[l] += (g >> l) & 1; }
      }
      if (on == 5000) {
#define LANE(k) { auto* R = r; int act = R->huff_fpga_core__DOT__u_eng__DOT__g_lane__BRA__##k##__KET____DOT__u__DOT__active; \
      if (act) fprintf(stderr, "stuck lane %d: tile %d rtile %d nb %d done %d narr %d cnt %d ds %u de %u\n", k, \
        (int)R->huff_fpga_core__DOT__u_eng__DOT__g_lane__BRA__##k##__KET____DOT__u__DOT__tile, \
        (int)R->huff_fpga_core__DOT__u_eng__DOT__g_lane__BRA__##k##__KET____DOT__u__DOT__rtile, \
        (int)R->huff_fpga_core__DOT__u_eng__DOT__g_lane__BRA__##k##__KET____DOT__u__DOT__nb, \
        (int)R->huff_fpga_core__DOT__u_eng__DOT__g_lane__BRA__##k##__KET____DOT__u__DOT__done, \
        (int)R->huff_fpga_core__DOT__u_eng__DOT__g_lane__BRA__##k##__KET____DOT__u__DOT__narr, \
        (int)R->huff_fpga_core__DOT__u_eng__DOT__g_lane__BRA__##k##__KET____DOT__u__DOT__cnt, \
        R->huff_fpga_core__DOT__ds[k][0], R->huff_fpga_core__DOT__de[k][0]); }
    LANE(0);
    LANE(1);
    LANE(2);
    LANE(3);
    LANE(4);
    LANE(5);
    LANE(6);
    LANE(7);
    LANE(8);
    LANE(9);
    LANE(10);
    LANE(11);
    LANE(12);
    LANE(13);
    LANE(14);
    LANE(15);
    LANE(16);
    LANE(17);
    LANE(18);
    LANE(19);
    LANE(20);
    LANE(21);
    LANE(22);
    LANE(23);
    LANE(24);
    LANE(25);
    LANE(26);
    LANE(27);
    LANE(28);
    LANE(29);
    LANE(30);
    LANE(31);
      }
      if (on == 3000) { for (int l = 0; l < 32; ++l) if (ng[l] != nr[l] || ng[l] != ngr[l]) fprintf(stderr, "lane %d: req %lld grant %lld resp %lld\n", l, ng[l], ngr[l], nr[l]); }
      static uint32_t everout = 0, everresp = 0;
      if (r->huff_fpga_core__DOT__run) { everout |= r->huff_fpga_core__DOT__out_valid; everresp |= r->huff_fpga_core__DOT__resp_valid; }
      if (on == 300) {
        fprintf(stderr, "lanes never output: ");
        for (int l = 0; l < 32; ++l) if (!((everout >> l) & 1)) fprintf(stderr, "%d(resp %d ds %u de %u) ", l, (everresp >> l) & 1,
            r->huff_fpga_core__DOT__desc_start[l], r->huff_fpga_core__DOT__desc_end[l]);
        fprintf(stderr, "\n");
      }
      if (r->huff_fpga_core__DOT__run && (on < 12 || on % 1000 == 0))
        fprintf(stderr, "c%5d st %d busy %d req %2d resp %2d out %2d rtile0 %d ds0 %u de0 %u stall %d wrd %u\n", on,
                (int)r->huff_fpga_core__DOT__st, (int)r->huff_fpga_core__DOT__busy,
                __builtin_popcount(r->huff_fpga_core__DOT__req_valid), __builtin_popcount(r->huff_fpga_core__DOT__resp_valid),
                __builtin_popcount(r->huff_fpga_core__DOT__out_valid), (int)(r->huff_fpga_core__DOT__rtile[0] & 0xFFFF),
                r->huff_fpga_core__DOT__desc_start[0], r->huff_fpga_core__DOT__desc_end[0],
                (int)r->huff_fpga_core__DOT__n_stall, r->huff_fpga_core__DOT__wrd);
    }
  }
#endif
  if (!expect(14) || got[i0] != 0xA5) { fprintf(stderr, "run: no reply\n"); return 1; }
  uint32_t hc, hs, hw; memcpy(&hc, &got[i0 + 1], 4); memcpy(&hs, &got[i0 + 5], 4); memcpy(&hw, &got[i0 + 9], 4);
  int tmo = got[i0 + 13];
  send8(0x07); send8((uint8_t)nt);
  size_t n = (size_t)32 * nt * 512, b0 = got.size();
  if (!expect(n)) { fprintf(stderr, "read: short\n"); return 1; }
  long long mm = 0;
  for (size_t i = 0; i < n; ++i) mm += got[b0 + i] != expv[i];
  ok = ok && mm == 0 && hc == cc && hs == cst && hw == cw && !tmo;
  printf("{\"nt\": %u, \"mismatch\": %lld, \"cycles\": %u, \"c_cycles\": %u, \"stalls\": %u, \"c_stalls\": %u, "
         "\"words\": %u, \"c_words\": %u, \"timeout\": %d, \"ok\": %d, \"sim_cycles\": %llu}\n", nt, mm, hc, cc, hs, cst,
         hw, cw, tmo, (int)ok, (unsigned long long)cyc);
  top->final();
  delete top;
  return ok ? 0 : 1;
}
