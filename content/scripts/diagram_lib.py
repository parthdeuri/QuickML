"""Drawing helpers shared by the diagram scripts: theme colours and the G canvas.

G.n(id, text, cx, cy, w, h, kind)  a rounded box (kind: io, op, param, loss, hi, cool)
G.a(p, q, label, dashed, kind)     an arrow between points or ("node-id", "l|r|t|b") anchors
G.p([points...], ...)              a polyline arrow
G.text(x, y, s, color)             a free text label
G.grid(x0, y0, rows, cols, cell, fills, texts)   a grid of cells (top-left corner)
The canvas is 100 units wide; height is given per diagram. PNGs are 4.4 in wide.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parent.parent.parent / "app/src/main/assets/diagrams"

THEMES = {
    "light": dict(text="#1b2430", muted="#5b6675", line="#44505f",
                  io=("#d6f5ec", "#12a37f"), op=("#eef1f5", "#7a8696"),
                  param=("#ece3ff", "#7a4fd6"), loss=("#ffe6d6", "#d9622b"),
                  hi=("#fff2b8", "#c79a00"), cool=("#d9ebff", "#2f78c4")),
    "dark": dict(text="#e8edf3", muted="#9aa6b5", line="#aab4c2",
                 io=("#10372f", "#1de9b6"), op=("#222a35", "#7d8a9b"),
                 param=("#2f2450", "#b388ff"), loss=("#4a2a1a", "#ff8a50"),
                 hi=("#4a4012", "#ffd54f"), cool=("#18304d", "#64b5f6")),
}


class G:
    def __init__(self, h, theme):
        self.h, self.t = h, THEMES[theme]
        self.fig = plt.figure(figsize=(4.4, 4.4 * h / 100))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, h)
        self.ax.axis("off")
        self.nodes = {}

    # ---- primitives ----
    def n(self, nid, text, cx, cy, w=22, h=9, kind="op", fs=8.5, circle=False):
        fill, edge = self.t[kind]
        if circle:
            self.ax.add_patch(Circle((cx, cy), w / 2, fc=fill, ec=edge, lw=1.3, zorder=3))
        else:
            self.ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                              boxstyle="round,pad=0,rounding_size=1.8", fc=fill, ec=edge,
                              lw=1.3, zorder=3))
        self.ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
                     color=self.t["text"], zorder=4, linespacing=1.15)
        self.nodes[nid] = (cx, cy, w, h)

    def pt(self, p):
        if isinstance(p[0], str):
            cx, cy, w, h = self.nodes[p[0]]
            s = p[1]
            return {"l": (cx - w / 2, cy), "r": (cx + w / 2, cy),
                    "t": (cx, cy + h / 2), "b": (cx, cy - h / 2)}[s]
        return p

    def a(self, p, q, label=None, dashed=False, kind="line", lpos=0.5, loff=(0, 0)):
        self.p([p, q], label, dashed, kind, lpos, loff)

    def p(self, pts, label=None, dashed=False, kind="line", lpos=0.5, loff=(0, 0), arrow=True):
        pts = [self.pt(x) for x in pts]
        col = self.t[kind] if kind == "line" else self.t[kind][1]
        ls = (0, (4, 3)) if dashed else "-"
        for i in range(len(pts) - 1):
            last = i == len(pts) - 2
            self.ax.add_patch(FancyArrowPatch(
                pts[i], pts[i + 1], arrowstyle="-|>" if (last and arrow) else "-",
                mutation_scale=9, color=col, lw=1.3, linestyle=ls, zorder=2,
                shrinkA=0, shrinkB=0))
        if label:
            (x0, y0), (x1, y1) = pts[0], pts[-1]
            self.text((x0 + x1) / 2 * lpos * 2 + loff[0] if lpos != 0.5 else (x0 + x1) / 2 + loff[0],
                      (y0 + y1) / 2 + loff[1], label, "muted")

    def text(self, x, y, s, color="text", fs=8, ha="center", style="normal"):
        self.ax.text(x, y, s, ha=ha, va="center", fontsize=fs, color=self.t[color],
                     style=style, zorder=5, linespacing=1.25)

    def grid(self, x0, y0, rows, cols, cell, fills=None, texts=None, fs=7.5):
        """(x0, y0) is the top-left corner; fills/texts keyed by (row, col)."""
        fills = fills or {}
        texts = texts or {}
        for r in range(rows):
            for c in range(cols):
                k = fills.get((r, c), "op")
                fill, edge = self.t[k]
                x, y = x0 + c * cell, y0 - (r + 1) * cell
                self.ax.add_patch(Rectangle((x, y), cell, cell, fc=fill, ec=edge, lw=1, zorder=3))
                if (r, c) in texts:
                    self.text(x + cell / 2, y + cell / 2, texts[(r, c)], fs=fs)

    def save(self, name, theme):
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / f"{name}_{theme}.png"
        self.fig.savefig(path, dpi=220, transparent=True)
        plt.close(self.fig)
        # 8-bit palette keeps the APK small; the flat colours lose nothing visible
        from PIL import Image
        Image.open(path).quantize(colors=96, method=Image.Quantize.FASTOCTREE).save(
            path, optimize=True)


def render(diagrams, want=None):
    """Renders {slug: (height, draw_fn)} in both themes; `want` limits it to some slugs."""
    slugs = want or list(diagrams)
    for slug in slugs:
        height, fn = diagrams[slug]
        for theme in THEMES:
            g = G(height, theme)
            fn(g)
            g.save(slug, theme)
    print("rendered", len(slugs), "diagrams")
