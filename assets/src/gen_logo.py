"""LazyWeights logo: a dumbbell at rest ("lazy weights") whose plates are columns of bit cells, shortest at the ends
like the tails of a concentrated code histogram, with z's rising from it, smallest first. Writes the badge (mark) and
the README header, light and dark, with all text outlined to paths (no font needed to view them).

  python3 assets/src/gen_logo.py      (needs fontTools and the DejaVu / Noto fonts; writes assets/*.svg)
"""
import os

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FONTS = {
    "serif-italic": "/usr/share/fonts/truetype/noto/NotoSerif-Italic.ttf",
    "serif-bolditalic": "/usr/share/fonts/truetype/noto/NotoSerif-BoldItalic.ttf",
    "sans": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "mono": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
}
THEMES = {
    "dark": dict(bg="#111827", ring="#334155", bar="#94A3B8", cell="#4C9AFF", hollow="#4C9AFF", z="#BFDBFE",
                 word="#4C9AFF", title="#F3F4F6", sub="#9CA3AF"),
    "light": dict(bg="#F5F8FC", ring="#0F172A", bar="#475569", cell="#0057B8", hollow="#0057B8", z="#5B8FD6",
                  word="#0057B8", title="#111827", sub="#4B5563"),
}
_fonts = {}


def text_path(s, font, size, x, y, anchor="start", tracking=0.0):
    """Outline s at baseline (x, y); returns (path d, advance width)."""
    f = _fonts.setdefault(font, TTFont(FONTS[font]))
    gs, cmap, hmtx = f.getGlyphSet(), f.getBestCmap(), f["hmtx"]
    k = size / f["head"].unitsPerEm
    names = [cmap[ord(c)] for c in s]
    width = sum(hmtx[n][0] for n in names) * k + tracking * (len(s) - 1)
    x0 = x - (width if anchor == "end" else width / 2 if anchor == "middle" else 0)
    pen = SVGPathPen(gs, ntos=lambda v: ("%.2f" % v).rstrip("0").rstrip("."))
    cx = x0
    for n in names:
        gs[n].draw(TransformPen(pen, (k, 0, 0, -k, cx, y)))
        cx += hmtx[n][0] * k + tracking
    return pen.getCommands(), width


def badge(c, wordmark=True):
    o = ['<circle cx="100" cy="100" r="92" fill="%s"/>' % c["bg"],
         '<circle cx="100" cy="100" r="90.9" fill="none" stroke="%s" stroke-width="1.3"/>' % c["ring"]]
    cy = 108 if wordmark else 112
    # bar and collars
    o.append('<rect x="19" y="%g" width="162" height="6" rx="3" fill="%s"/>' % (cy - 3, c["bar"]))
    for x in (70.0, 126.0):
        o.append('<rect x="%g" y="%g" width="4" height="18" rx="1.5" fill="%s"/>' % (x, cy - 9, c["bar"]))
    # plates: columns of bit cells, 5 / 3 / 1 from the inside out (a concentrated code histogram, mirrored); the
    # odd counts put a cell over the bar, and the rarer outer codes are drawn lighter
    cw, ch, gap = 12.5, 9.6, 2.6
    for k, (x, n) in enumerate([(55.0, 5), (40.0, 3), (25.0, 1)]):
        h = n * ch + (n - 1) * gap
        for xx in (x, 200 - x - cw):
            for i in range(n):
                o.append('<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" rx="2" fill="%s" fill-opacity="%.2f"/>'
                         % (xx, cy - h / 2 + i * (ch + gap), cw, ch, c["cell"], (1.0, 0.8, 0.6)[k]))
    # z's rising to the upper right, growing
    for x, y, s in ((118, 72, 13), (132, 57, 17), (148, 40, 22)):
        d, _ = text_path("z", "serif-bolditalic", s, x, y)
        o.append('<path d="%s" fill="%s"/>' % (d, c["z"]))
    if wordmark:
        d, _ = text_path("lazyweights", "mono", 12.5, 100, 158, "middle", tracking=0.9)
        o.append('<path d="%s" fill="%s"/>' % (d, c["word"]))
    return o


def svg(w, h, body, title, desc):
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d" role="img" '
            'aria-labelledby="t d"><title id="t">%s</title><desc id="d">%s</desc>%s</svg>\n'
            % (w, h, w, h, title, desc, "".join(body)))


DESC = ("LazyWeights logo: a dumbbell at rest whose plates are columns of bit cells (5, 3, 1 from the inside out, "
        "like a concentrated code histogram), with z's rising from it.")
for name, c in THEMES.items():
    open(os.path.join(OUT, "lazyweights-mark-%s.svg" % name), "w").write(
        svg(200, 200, badge(c), "LazyWeights", DESC))
    head = badge(c)
    d, _ = text_path("LazyWeights", "serif-italic", 62, 222, 98)
    head.append('<path d="%s" fill="%s"/>' % (d, c["title"]))
    for i, line in enumerate(("Decode-on-demand edge LLMs:", "Huffman-coded weights, decoded near DRAM")):
        d, w = text_path(line, "sans", 18, 226, 136 + i * 25)
        assert 226 + w < 652, (line, w)
        head.append('<path d="%s" fill="%s"/>' % (d, c["sub"]))
    open(os.path.join(OUT, "lazyweights-header-%s.svg" % name), "w").write(
        svg(660, 200, head, "LazyWeights", DESC + " With the paper title."))
print("wrote", sorted(f for f in os.listdir(OUT) if f.endswith(".svg")))
