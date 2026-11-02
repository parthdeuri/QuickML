#!/usr/bin/env python3
"""Renders the card diagrams to transparent PNGs (light and dark theme).

Run: python content/scripts/make_diagrams.py [slug ...]
Needs matplotlib (local only; the PNGs are committed, CI never runs this).
Output: app/src/main/assets/diagrams/<slug>_light.png and <slug>_dark.png
Every canvas is 100 units wide; PNGs are 4.4 in wide so text stays readable on a phone.
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


# ------------------------------------------------------------------ diagrams
def backpropagation(g):
    y = 52
    g.n("x", "x", 7, y, 10, kind="io")
    g.n("w1", "W1, b1", 27, y, 17, kind="param")
    g.n("tanh", "tanh", 47, y, 14)
    g.n("w2", "W2, b2", 67, y, 17, kind="param")
    g.n("yh", "ŷ", 86, y, 10, kind="io")
    g.n("L", "loss\nL(ŷ, y)", 86, 33, 17, 11, kind="loss")
    for a, b in [("x", "w1"), ("w1", "tanh"), ("tanh", "w2"), ("w2", "yh")]:
        g.a((a, "r"), (b, "l"))
    g.a(("yh", "b"), ("L", "t"))
    g.text(46, 63, "forward pass: left to right", "muted")
    g.n("d2", "dW2, db2", 67, 14, 20, kind="loss")
    g.n("dt", "∂ tanh", 47, 14, 14, kind="loss")
    g.n("d1", "dW1, db1", 27, 14, 20, kind="loss")
    g.a(("L", "b"), ("d2", "r"), dashed=True, kind="loss")
    g.a(("d2", "l"), ("dt", "r"), dashed=True, kind="loss")
    g.a(("dt", "l"), ("d1", "r"), dashed=True, kind="loss")
    g.text(40, 3, "gradients flow back (chain rule)", "muted")


def convolution_2d(g):
    inp = {(r, c): "io" for r in range(3) for c in range(3)}
    g.grid(5, 52, 5, 5, 7, inp)
    g.text(22.5, 56, "input (5×5)", "muted")
    g.text(44, 34, "∗", fs=14)
    letters = dict(zip([(r, c) for r in range(3) for c in range(3)], "abcdefghi"))
    g.grid(50, 45, 3, 3, 7, {k: "param" for k in letters}, letters)
    g.text(60.5, 49, "kernel (3×3)", "muted")
    g.text(76, 34, "=", fs=14)
    g.grid(82, 40, 3, 3, 5.5, {(0, 0): "hi"})
    g.text(90, 44, "output", "muted")
    g.text(50, 12, "slide the kernel: each stop gives one output", "text")
    g.text(50, 4, "out = (n − k + 2p) / s + 1", "muted")
    g.text(22.5, 12 + 12, "", "text")


def lstm_cell(g):
    g.n("c0", "c(t−1)", 9, 62, 14, kind="io")
    g.n("mul", "×", 30, 62, 7, circle=True, fs=10)
    g.n("add", "+", 58, 62, 7, circle=True, fs=10)
    g.n("c1", "c(t)", 90, 62, 14, kind="io")
    g.a(("c0", "r"), ("mul", "l"))
    g.a(("mul", "r"), ("add", "l"))
    g.a(("add", "r"), ("c1", "l"))
    g.n("ig", "i ⊙ g", 58, 44, 15, kind="param")
    g.a(("ig", "t"), ("add", "b"))
    g.n("gates", "gates  f  i  g  o\nfrom [h(t−1), x(t)]", 38, 22, 46, 12, kind="param")
    g.a((30, 28), ("mul", "b"), "f", loff=(3, 0))
    g.a((58, 28), ("ig", "b"), "i, g", loff=(4, 0))
    g.n("tanh", "tanh", 90, 46, 14)
    g.n("o", "⊙ o", 90, 32, 14, kind="param")
    g.n("h", "h(t)", 90, 14, 14, kind="io")
    g.a(("c1", "b"), ("tanh", "t"))
    g.a(("tanh", "b"), ("o", "t"))
    g.a(("o", "b"), ("h", "t"))
    g.a(("gates", "r"), ("o", "l"), "o", loff=(0, 2.5))
    g.text(40, 5, "f forget · i input · g candidate · o output", "muted", fs=7)


def residual_connections(g):
    g.n("x", "x", 38, 64, 14, kind="io")
    g.n("f", "F(x)\n(layers)", 38, 44, 26, 12, kind="param")
    g.n("add", "+", 38, 24, 8, circle=True, fs=11)
    g.n("y", "y = x + F(x)", 38, 8, 34, kind="io")
    g.a(("x", "b"), ("f", "t"))
    g.a(("f", "b"), ("add", "t"))
    g.a(("add", "b"), ("y", "t"))
    g.p([("x", "r"), (72, 64), (72, 24), ("add", "r")], kind="loss", dashed=True)
    g.text(86, 44, "identity\nshortcut", "muted")
    g.text(86, 24, "", "muted")


def self_attention(g):
    g.n("x", "x", 6, 38, 9, kind="io")
    for nid, name, y in [("q", "Q", 62), ("k", "K", 40), ("v", "V", 16)]:
        g.n("w" + nid, "W" + nid, 24, y, 11, kind="param")
        g.n(nid, name, 40, y, 11)
        g.a(("x", "r"), ("w" + nid, "l"))
        g.a(("w" + nid, "r"), (nid, "l"))
    g.n("A", "A = softmax\n(Q Kᵀ / √d)", 76, 51, 36, 13, kind="hi")
    g.a(("q", "r"), ("A", "l"))
    g.a(("k", "r"), ("A", "l"))
    g.n("out", "out = A · V", 76, 16, 36, kind="io")
    g.a(("A", "b"), ("out", "t"))
    g.a(("v", "r"), ("out", "l"))
    g.text(50, 4, "A[i, j] = how much position i reads from j", "muted", fs=7.5)


def multi_head_attention(g):
    g.n("x", "x", 50, 62, 12, kind="io")
    for i, (nm, cx) in enumerate([("head 1", 17), ("head 2", 50), ("head h", 83)]):
        g.n(f"h{i}", f"{nm}\nattention\n(Qi, Ki, Vi)", cx, 40, 28, 15, kind="param")
        g.a(("x", "b"), (f"h{i}", "t"))
    g.n("cat", "concat", 50, 20, 30)
    for i in range(3):
        g.a((f"h{i}", "b"), ("cat", "t"))
    g.n("wo", "Wo", 50, 6, 14, kind="param")
    g.a(("cat", "b"), ("wo", "t"))
    g.text(80, 6, "→ out", "muted")


def transformer_block(g):
    cx = 36
    g.n("x", "x", cx, 94, 14, kind="io")
    g.n("ln1", "LayerNorm", cx, 82, 30)
    g.n("att", "Self-attention", cx, 71, 36, kind="param")
    g.n("a1", "+", cx, 59, 8, circle=True, fs=11)
    g.n("ln2", "LayerNorm", cx, 47, 30)
    g.n("mlp", "MLP (GELU)", cx, 36, 36, kind="param")
    g.n("a2", "+", cx, 24, 8, circle=True, fs=11)
    g.n("out", "out", cx, 11, 14, kind="io")
    for a, b in [("x", "ln1"), ("ln1", "att"), ("att", "a1"), ("a1", "ln2"),
                 ("ln2", "mlp"), ("mlp", "a2"), ("a2", "out")]:
        g.a((a, "b"), (b, "t"))
    g.p([("x", "r"), (78, 94), (78, 59), ("a1", "r")], kind="loss", dashed=True)
    g.p([(78, 59), (78, 24), ("a2", "r")], kind="loss", dashed=True)
    g.text(89, 76, "skip\nconnection", "muted")
    g.text(89, 41, "skip\nconnection", "muted")


def autoencoder(g):
    y = 40
    g.n("x", "x", 7, y, 10, kind="io")
    g.n("enc", "encoder", 28, y, 19, kind="param")
    g.n("z", "z", 50, y, 10, 14, kind="hi")
    g.n("dec", "decoder", 72, y, 19, kind="param")
    g.n("xh", "x̂", 92, y, 10, kind="io")
    for a, b in [("x", "enc"), ("enc", "z"), ("z", "dec"), ("dec", "xh")]:
        g.a((a, "r"), (b, "l"))
    g.text(50, 26, "z is a small bottleneck:", "muted")
    g.text(50, 20, "it must keep what the decoder needs to rebuild x", "muted", fs=7.2)
    g.n("loss", "loss = ‖x − x̂‖²", 50, 8, 40, kind="loss")


def vae_reparameterization(g):
    g.n("x", "x", 50, 94, 12, kind="io")
    g.n("enc", "encoder", 50, 82, 24, kind="param")
    g.n("mu", "μ", 30, 66, 14, kind="io")
    g.n("ls", "log σ²", 70, 66, 18, kind="io")
    g.a(("x", "b"), ("enc", "t"))
    g.a(("enc", "b"), ("mu", "t"))
    g.a(("enc", "b"), ("ls", "t"))
    g.n("z", "z = μ + σ ⊙ ε", 40, 50, 44, kind="hi")
    g.n("eps", "ε ~ N(0, I)", 86, 50, 24, kind="op")
    g.a(("mu", "b"), ("z", "t"))
    g.a(("ls", "b"), ("z", "r"))
    g.a(("eps", "l"), ("z", "r"))
    g.n("dec", "decoder", 40, 36, 24, kind="param")
    g.n("xh", "x̂", 40, 22, 12, kind="io")
    g.a(("z", "b"), ("dec", "t"))
    g.a(("dec", "b"), ("xh", "t"))
    g.text(50, 7, "loss = ‖x − x̂‖² + KL(N(μ,σ²) ‖ N(0,I))", "muted", fs=7.5)


def gan_minimax_loss(g):
    g.n("z", "noise z", 22, 92, 22, kind="op")
    g.n("G", "Generator", 22, 76, 26, kind="param")
    g.n("fake", "fake sample", 22, 60, 26, kind="loss")
    g.n("real", "real data", 78, 60, 26, kind="io")
    g.a(("z", "b"), ("G", "t"))
    g.a(("G", "b"), ("fake", "t"))
    g.n("D", "Discriminator", 50, 40, 36, kind="param")
    g.a(("fake", "b"), ("D", "t"))
    g.a(("real", "b"), ("D", "t"))
    g.n("out", "real or fake?", 50, 22, 32, kind="hi")
    g.a(("D", "b"), ("out", "t"))
    g.text(50, 10, "D learns to tell them apart;\nG learns to fool D", "muted")
    g.p([("out", "l"), (6, 22), (6, 76), ("G", "l")], "G gets D's verdict", dashed=True,
        kind="loss", loff=(0, 0))
    g.ax.texts[-1].set_visible(False)


def cross_validation(g):
    fills = {(r, c): ("hi" if r == c else "io") for r in range(5) for c in range(5)}
    texts = {(r, c): ("VAL" if r == c else "train") for r in range(5) for c in range(5)}
    g.grid(16, 80, 5, 5, 14, fills, texts, fs=7)
    for r in range(5):
        g.text(8, 80 - 14 * r - 7, f"{r + 1}", "muted")
    g.text(8, 85, "fold", "muted", fs=7)
    g.text(50, 4, "score = mean of the 5 validation scores", "text")


def mlp_forward_pass(g):
    g.n("x", "x   (d features)", 50, 54, 44, kind="io")
    g.n("h", "h = act(x W1 + b1)   (m units)", 50, 40, 70, kind="param")
    g.n("l", "logits = h W2 + b2", 50, 26, 50, kind="param")
    g.n("p", "p = softmax(logits)", 50, 12, 50, kind="io")
    for a, b in [("x", "h"), ("h", "l"), ("l", "p")]:
        g.a((a, "b"), (b, "t"))


def kv_cache(g):
    g.n("tok", "new token t", 50, 60, 30, kind="io")
    g.n("qkv", "q_t, k_t, v_t", 50, 46, 34, kind="param")
    g.a(("tok", "b"), ("qkv", "t"))
    g.n("K", "K cache:  k1  k2 … kt", 27, 28, 48, kind="hi")
    g.n("V", "V cache:  v1  v2 … vt", 27, 15, 48, kind="hi")
    g.p([(40, 41), (40, 36), (27, 36), ("K", "t")], "append k_t, v_t", loff=(-3, 3))
    g.n("out", "out_t =\nsoftmax(q_t Kᵀ/√d) · V", 77, 22, 42, 14, kind="io")
    g.p([("qkv", "r"), (90, 46), ("out", "t")])
    g.a(("K", "r"), ("out", "l"))
    g.text(50, 4, "older keys and values are never recomputed", "muted", fs=7.5)


def lora_low_rank_adaptation(g):
    g.n("x", "x", 7, 28, 9, kind="io")
    g.n("W", "W   (frozen)", 44, 48, 34)
    g.n("A", "A\n(d × r)", 32, 14, 17, 11, kind="param")
    g.n("B", "B\n(r × d)", 56, 14, 17, 11, kind="param")
    g.n("add", "+", 78, 28, 8, circle=True, fs=11)
    g.n("y", "y", 94, 28, 9, kind="io")
    g.p([("x", "t"), (7, 48), ("W", "l")])
    g.p([("x", "b"), (7, 14), ("A", "l")])
    g.a(("A", "r"), ("B", "l"))
    g.p([("W", "r"), (78, 48), ("add", "t")])
    g.p([("B", "r"), (78, 14), ("add", "b")])
    g.a(("add", "r"), ("y", "l"))
    g.text(50, 62, "only A and B are trained (r ≪ d)", "muted", fs=7.5)
    g.text(50, 2, "y = x W + (α / r) · x A B", "text")


def k_means(g):
    g.n("init", "init centres", 40, 64, 34, kind="io")
    g.n("assign", "assign each point to\nthe nearest centre", 40, 46, 52, 13, kind="param")
    g.n("move", "move each centre to\nits cluster mean", 40, 26, 52, 13, kind="param")
    g.a(("init", "b"), ("assign", "t"))
    g.a(("assign", "b"), ("move", "t"))
    g.p([("move", "r"), (80, 26), (80, 46), ("assign", "r")], "repeat", kind="loss", dashed=True)
    g.ax.texts[-1].set_visible(False)
    g.text(88, 36, "repeat", "muted")
    g.n("done", "stop when assignments\nstop changing", 40, 7, 52, 11, kind="hi")
    g.a(("move", "b"), ("done", "t"))


def diffusion_forward_process(g):
    xs = [10, 30, 50, 70, 90]
    labels = ["x₀", "x₁", "x₂", "…", "x_T"]
    kinds = ["io", "io", "op", "op", "loss"]
    for i, (cx, lab, k) in enumerate(zip(xs, labels, kinds)):
        g.n(f"n{i}", lab, cx, 34, 12, 12, kind=k)
    for i in range(4):
        g.a((f"n{i}", "r"), (f"n{i + 1}", "l"))
    g.text(50, 48, "forward: add a little noise each step", "muted")
    g.text(10, 22, "clean", "muted")
    g.text(90, 22, "pure noise", "muted")
    g.p([(88, 24), (12, 24)], dashed=True, kind="param")
    g.ax.patches[-1].set_visible(True)
    g.text(50, 18, "the model learns the reverse: denoise step by step", "muted", fs=7.5)
    g.text(50, 6, r"$q(x_t\,|\,x_0)=\mathcal{N}(\sqrt{\bar\alpha_t}\,x_0,\;(1-\bar\alpha_t)I)$", "text")


def batch_normalization(g):
    fills = {(r, c): "op" for r in range(5) for c in range(5)}
    for r in range(5):
        fills[(r, 1)] = "param"
    for c in range(5):
        fills[(2, c)] = "io"
    fills[(2, 1)] = "hi"
    g.grid(24, 68, 5, 5, 11, fills, {(r, c): "x" for r in range(5) for c in range(5)}, fs=8)
    g.text(51, 72, "features →", "muted")
    g.text(11, 40, "batch\n↓", "muted")
    g.n("lb", "column = batch norm", 27, 6, 44, kind="param", fs=8)
    g.n("ln", "row = layer norm", 77, 6, 40, kind="io", fs=8)


DIAGRAMS = {
    "backpropagation": (68, backpropagation),
    "convolution-2d": (62, convolution_2d),
    "lstm-cell": (72, lstm_cell),
    "residual-connections": (76, residual_connections),
    "self-attention": (72, self_attention),
    "multi-head-attention": (72, multi_head_attention),
    "transformer-block": (104, transformer_block),
    "autoencoder": (52, autoencoder),
    "vae-reparameterization": (100, vae_reparameterization),
    "gan-minimax-loss": (100, gan_minimax_loss),
    "cross-validation": (88, cross_validation),
    "mlp-forward-pass": (64, mlp_forward_pass),
    "kv-cache": (68, kv_cache),
    "lora-low-rank-adaptation": (68, lora_low_rank_adaptation),
    "k-means": (76, k_means),
    "diffusion-forward-process": (56, diffusion_forward_process),
    "batch-normalization": (76, batch_normalization),
}

if __name__ == "__main__":
    want = sys.argv[1:] or list(DIAGRAMS)
    for slug in want:
        h, fn = DIAGRAMS[slug]
        for theme in THEMES:
            g = G(h, theme)
            fn(g)
            g.save(slug, theme)
    print("rendered", len(want), "diagrams")
