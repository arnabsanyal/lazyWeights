"""ZCU102 host (runs on the board's Cortex-A53s, as root for /dev/mem): drives the Huffman engines in the fabric
(hw/huff/fpga/zcu_core.sv, register map in its header) at 0x8000_0000 over the LPD AXI port. For every HMAT file
(hw/huff/tb/chunks_main.cpp: a whole matrix cut into chunks) it loads the tables into every engine, then runs the
chunks NE at a time (engines started in the same cycle) and checks each chunk's per-lane CRC-32s and symbol counts
against the C model's, and its cycles / stalls / words. One JSON line per matrix on stdout.

  sudo python3 zcu_host.py zjobs/*.hmat > results_zcu.jsonl
  sudo python3 zcu_host.py --power 20 zjobs/<file>.hmat     (run 4 chunks back to back for 20 s, sample the rails)
"""
import argparse
import glob
import json
import mmap
import os
import struct
import sys
import time

import numpy as np

BASE, SIZE = 0x80000000, 1 << 24
NTM = 16


class Dev:
    def __init__(self):
        fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
        self.mm = mmap.mmap(fd, SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=BASE)
        self.r = np.frombuffer(self.mm, dtype=np.uint32)
        if self.rd(0xE00000) != 0x32465548:
            sys.exit("no HUF2 design at 0x%x (id %08x): load the bitstream" % (BASE, self.rd(0xE00000)))
        c = self.rd(0xE00004)
        self.ne, self.aw, self.ntm, self.lat = c & 0xFF, (c >> 8) & 0xFF, (c >> 16) & 0xFF, c >> 24
        self.khz = self.rd(0xE0000C)
        # measured fabric clock (free-running counter over 0.5 s): the Si570 must be at its 300 MHz default
        c0, t0 = self.rd(0xE00010), time.time()
        time.sleep(0.5)
        c1, t1 = self.rd(0xE00010), time.time()
        self.mhz_meas = ((c1 - c0) & 0xFFFFFFFF) / (t1 - t0) / 1e6

    def rd(self, a):
        return int(self.r[a >> 2])

    def wr(self, a, v):
        self.r[a >> 2] = v

    def wblk(self, a, arr):
        # glibc's aarch64 memcpy ends a copy whose length is not a multiple of 16 B with an overlapping 16 B store at a
        # 4 B-aligned address: an alignment fault on Device memory (SIGBUS). Copy whole 16 B groups, then the tail words.
        assert a % 16 == 0
        n = len(arr) & ~3
        self.r[a >> 2:(a >> 2) + n] = arr[:n]
        for i in range(n, len(arr)):
            self.r[(a >> 2) + i] = arr[i]

    def rblk(self, a, n):
        return np.array(self.r[a >> 2:(a >> 2) + n])


def load_hmat(path):
    B = open(path, "rb").read()
    assert B[:4] == b"HMAT", path
    ver, nch, ntt, bits, N, K = struct.unpack_from("<6I", B, 4)
    o = 28
    srt = np.frombuffer(B, np.uint8, 256, o + 4096).astype(np.uint32)
    cmp_ = np.frombuffer(B, np.uint16, 33, o + 4096 + 256).astype(np.uint32).reshape(11, 3)
    o += 4096 + 256 + 66
    chunks = []
    for _ in range(nch):
        t0, nt, nw, cc, cs, cw = struct.unpack_from("<6I", B, o)
        o += 24
        desc = np.frombuffer(B, np.uint32, nt * 64, o).reshape(nt, 32, 2)
        o += nt * 256
        words = np.frombuffer(B, np.uint32, nw, o)
        o += nw * 4
        crc = np.frombuffer(B, np.uint32, 32, o)
        o += 128
        chunks.append(dict(t0=t0, nt=nt, nw=nw, cc=cc, cs=cs, cw=cw, desc=desc, words=words, crc=crc))
    return dict(nch=nch, ntt=ntt, bits=bits, N=N, K=K, srt=srt, cmp=cmp_, chunks=chunks)


def load_tables(d, m):
    for e in range(d.ne):
        b = e << 21
        d.wblk(b + 0x0D0000, m["srt"])
        for l in range(1, 12):
            lim, first, base = m["cmp"][l - 1]
            d.wr(b + 0x0E0000 + 16 * l, int(lim))
            d.wr(b + 0x0E0000 + 16 * l + 4, int(first))
            d.wr(b + 0x0E0000 + 16 * l + 8, int(base))


def load_chunk(d, e, c):
    b = e << 21
    d.wblk(b, c["words"])
    dm = np.zeros((32, NTM, 2), np.uint32)
    dm[:, :c["nt"], :] = c["desc"].transpose(1, 0, 2)
    d.wblk(b + 0x080000, dm.reshape(-1))
    d.wr(b + 0x100000, c["nt"])


def run(d, mask):
    d.wr(0xE00008, mask)
    while d.rd(0xE00008):
        pass


def check(d, e, c):
    b = e << 21
    cy, st, wo = d.rd(b + 0x100008), d.rd(b + 0x10000C), d.rd(b + 0x100010)
    stat = d.rd(b + 0x100004)
    crc = d.rblk(b + 0x100000 + 4 * 64, 32) ^ np.uint32(0xFFFFFFFF)
    ns = d.rblk(b + 0x100000 + 4 * 128, 32)
    crc_bad = int((crc != c["crc"]).sum())
    sym_bad = int((ns != c["nt"] * 512).sum())
    ok = crc_bad == 0 and sym_bad == 0 and cy == c["cc"] and st == c["cs"] and wo == c["cw"] and not stat & 4
    return ok, dict(cycles=cy, stalls=st, words=wo, crc_bad=crc_bad, sym_bad=sym_bad, timeout=bool(stat & 4))


def rails():
    """INA226 rails (hwmon): name -> W."""
    out = {}
    for h in glob.glob("/sys/class/hwmon/hwmon*"):
        try:
            name = open(h + "/name").read().strip()
            p = h + "/power1_input"
            if os.path.exists(p):
                lab = name
                if os.path.exists(h + "/device/label"):
                    lab = open(h + "/device/label").read().strip()
                out[lab or name] = int(open(p).read()) / 1e6
        except OSError:
            pass
    return out


def do_matrix(d, path):
    t = time.time()
    m = load_hmat(path)
    load_tables(d, m)
    ch = m["chunks"]
    bad, nsym, cyc_sum, cyc_par, first_bad = 0, 0, 0, 0, None
    t_load = t_run = 0.0
    for i in range(0, len(ch), d.ne):
        grp = ch[i:i + d.ne]
        t1 = time.time()
        for e, c in enumerate(grp):
            load_chunk(d, e, c)
        t2 = time.time()
        run(d, (1 << len(grp)) - 1)
        t3 = time.time()
        t_load += t2 - t1
        t_run += t3 - t2
        mx = 0
        for e, c in enumerate(grp):
            ok, r = check(d, e, c)
            bad += not ok
            nsym += 32 * c["nt"] * 512
            cyc_sum += r["cycles"]
            mx = max(mx, r["cycles"])
            if not ok and first_bad is None:
                first_bad = dict(chunk=i + e, t0=c["t0"], c_cycles=c["cc"], c_stalls=c["cs"], c_words=c["cw"], **r)
        cyc_par += mx
    f = d.khz * 1e3
    return dict(job=os.path.basename(path), N=m["N"], K=m["K"], bits=m["bits"], ntt=m["ntt"], chunks=len(ch),
                bad_chunks=bad, ok=int(bad == 0), symbols=nsym, cycles_sum=cyc_sum, cycles_parallel=cyc_par,
                engines=d.ne, clk_mhz=d.khz / 1e3,
                rate_engine_gsym_s=round(nsym / (cyc_sum / d.ne) * f / 1e9 if cyc_sum else 0, 4),
                rate_parallel_gsym_s=round(nsym / cyc_par * f / 1e9 if cyc_par else 0, 4),
                load_s=round(t_load, 3), run_s=round(t_run, 4), total_s=round(time.time() - t, 3), first_bad=first_bad)


def do_power(d, path, secs):
    m = load_hmat(path)
    load_tables(d, m)
    grp = m["chunks"][:d.ne]
    for e, c in enumerate(grp):
        load_chunk(d, e, c)
    mask = (1 << len(grp)) - 1
    idle = [rails() for _ in range(10)]
    time.sleep(0.5)
    samples, n, cyc = [], 0, 0
    t0 = tn = time.time()
    while time.time() - t0 < secs:
        run(d, mask)
        n += 1
        cyc += d.rd(0x100008)
        if time.time() - tn > 0.2:
            samples.append(rails())
            tn = time.time()
    wall = time.time() - t0
    ok = all(check(d, e, c)[0] for e, c in enumerate(grp))
    avg = lambda ss: {k: round(sum(s[k] for s in ss) / len(ss), 4) for k in ss[0]}
    return dict(job=os.path.basename(path), mode="power", runs=n, wall_s=round(wall, 2), ok=int(ok),
                duty=round(cyc / (wall * d.khz * 1e3), 3), idle_w=avg(idle), busy_w=avg(samples),
                symbols_per_run=sum(32 * c["nt"] * 512 for c in grp), engines=d.ne, clk_mhz=d.khz / 1e3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--power", type=float, default=0)
    a = ap.parse_args()
    d = Dev()
    print("# engines %d, AW %d, NTM %d, LAT %d, clock %.1f MHz (measured %.2f MHz)" % (
        d.ne, d.aw, d.ntm, d.lat, d.khz / 1e3, d.mhz_meas), file=sys.stderr, flush=True)
    if abs(d.mhz_meas - d.khz / 1e3) > 0.02 * d.khz / 1e3:
        sys.exit("fabric clock %.2f MHz != %.1f MHz: is the Si570 at 300 MHz?" % (d.mhz_meas, d.khz / 1e3))
    bad = 0
    for f in a.files:
        r = do_power(d, f, a.power) if a.power else do_matrix(d, f)
        bad += not r["ok"]
        print(json.dumps(r), flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
