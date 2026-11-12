"""FPGA bring-up host (runs on the board host, e.g. the ZCU102 with the Nexys 4 DDR on its USB port): sends each job
file (build/huff_slice, hw/huff/tb/slice_main.cpp) to the Huffman engine on the Nexys (hw/huff/fpga/nexys_top.sv)
over the USB-UART, runs it, reads back the decoded symbols, and checks them and the cycle / stall / word counts
against the C model's (stored in the job). Protocol: hw/huff/fpga/nexys_top.sv header.

  python3 fpga_host.py [--port /dev/ttyUSB1] [--baud 1000000] job ... > results.jsonl
"""
import argparse
import json
import struct
import sys
import time

import serial


def read_exact(s, n, what):
    b = s.read(n)
    if len(b) != n:
        raise RuntimeError("%s: got %d of %d bytes" % (what, len(b), n))
    return b


def ack(s, what):
    b = read_exact(s, 1, what)
    if b != b"\xac":
        raise RuntimeError("%s: bad ack %r" % (what, b))


def run_job(s, path):
    J = open(path, "rb").read()
    assert J[:4] == b"HJOB", path
    ver, nt, nw, cc, cst, cw, t0, ntt = struct.unpack_from("<8I", J, 4)
    o = 36
    lut, o = J[o:o + 4096], o + 4096
    srt, o = J[o:o + 256], o + 256
    cmp_, o = J[o:o + 66], o + 66
    desc, o = J[o:o + nt * 32 * 8], o + nt * 32 * 8
    words, o = J[o:o + nw * 4], o + nw * 4
    exp = J[o:o + 32 * nt * 512]
    t = time.time()
    s.write(b"\x01" + lut); ack(s, "lut")
    s.write(b"\x02" + srt); ack(s, "srt")
    s.write(b"\x03" + cmp_); ack(s, "cmp")
    s.write(b"\x04" + bytes([nt]) + desc); ack(s, "desc")
    s.write(b"\x05" + struct.pack("<I", nw) + words); ack(s, "words")
    t_load = time.time() - t
    s.write(b"\x06" + bytes([nt]))
    r = read_exact(s, 14, "run")
    if r[0] != 0xA5:
        raise RuntimeError("run: bad reply %r" % r)
    hc, hs, hw = struct.unpack_from("<3I", r, 1)
    tmo = r[13]
    s.write(b"\x07" + bytes([nt]))
    got = read_exact(s, 32 * nt * 512, "read")
    mism = sum(a != b for a, b in zip(got, exp))
    return dict(job=path.rsplit("/", 1)[-1], t0=t0, nt=nt, ntt=ntt, n_words=nw, symbols=32 * nt * 512, mismatch=mism,
                cycles=hc, c_cycles=cc, stalls=hs, c_stalls=cst, words=hw, c_words=cw, timeout=tmo,
                ok=int(mism == 0 and hc == cc and hs == cst and hw == cw and not tmo),
                load_s=round(t_load, 3), total_s=round(time.time() - t, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="+")
    ap.add_argument("--port", default="/dev/ttyUSB1")
    ap.add_argument("--baud", type=int, default=1000000)
    a = ap.parse_args()
    s = serial.Serial(a.port, a.baud, timeout=5)
    time.sleep(0.1)
    s.reset_input_buffer()
    s.write(b"\x08")
    p = s.read(2)
    if p != b"\x5a\x01":
        sys.exit("ping failed: %r (is the bitstream loaded? %s at %d baud?)" % (p, a.port, a.baud))
    bad = 0
    for j in a.jobs:
        try:
            r = run_job(s, j)
        except RuntimeError as e:
            r = dict(job=j.rsplit("/", 1)[-1], ok=0, error=str(e))
            s.reset_input_buffer()
        bad += not r["ok"]
        print(json.dumps(r), flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
