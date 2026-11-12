"""Merge liberty files into one library (header of the first, cell groups of all): Yosys 0.38's abc takes one."""
import re
import sys


def cells(text):
    out, i = [], 0
    for m in re.finditer(r"\n\s*cell\s*\(", text):
        if m.start() < i:
            continue
        j, depth = text.index("{", m.end()), 0
        k = j
        while True:
            c = text[k]
            depth += c == "{"
            depth -= c == "}"
            k += 1
            if depth == 0:
                break
        out.append(text[m.start():k])
        i = k
    return out


texts = [open(f).read() for f in sys.argv[2:]]
head = texts[0][:re.search(r"\n\s*cell\s*\(", texts[0]).start()]
head = re.sub(r"library\s*\(\s*[^)]*\)", "library (asap7_merged)", head, count=1)
with open(sys.argv[1], "w") as f:
    f.write(head + "".join(c for t in texts for c in cells(t)) + "\n}\n")
