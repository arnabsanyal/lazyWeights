"""Huffman code lengths and coded size from a histogram (comparison sizes only)."""
import heapq
import itertools

import numpy as np


def code_lengths(counts):
    c = np.asarray(counts, dtype=np.int64)
    syms = [int(i) for i in np.nonzero(c)[0]]
    lens = np.zeros(len(c), dtype=np.int64)
    if len(syms) == 0:
        return lens
    if len(syms) == 1:
        lens[syms[0]] = 1
        return lens
    tie = itertools.count()
    heap = [(int(c[s]), next(tie), [s]) for s in syms]
    heapq.heapify(heap)
    while len(heap) > 1:
        a, _, sa = heapq.heappop(heap)
        b, _, sb = heapq.heappop(heap)
        for s in sa + sb:
            lens[s] += 1
        heapq.heappush(heap, (a + b, next(tie), sa + sb))
    return lens


def coded_bits(counts):
    c = np.asarray(counts, dtype=np.int64)
    return int((c * code_lengths(c)).sum())
