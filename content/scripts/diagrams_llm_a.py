"""Card diagrams (LLM set A): GPT, decoder block, FFN, Llama, sliding window, MoE, T5, Flash, speculative."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagram_lib import G, render  # noqa: E402


def gpt_decoder_only(g):
    cx = 38
    g.n("ids", "token ids", cx, 88, 26, kind="io")
    g.n("emb", "E[id] + P[t]", cx, 75, 34, kind="param")
    g.n("blk", "N × causal block", cx, 62, 38, kind="param")
    g.n("ln", "final LayerNorm", cx, 49, 38)
    g.n("lg", "h Eᵀ", cx, 36, 24, kind="hi")
    g.n("out", "logits (T, V)", cx, 23, 32, kind="io")
    for a, b in [("ids", "emb"), ("emb", "blk"), ("blk", "ln"), ("ln", "lg"), ("lg", "out")]:
        g.a((a, "b"), (b, "t"))
    g.p([("emb", "r"), (80, 75), (80, 36), ("lg", "r")], kind="hi", dashed=True)
    g.text(89, 55, "weight\ntying", "muted")
    g.text(50, 9, "row t of the logits scores token t+1", "muted", fs=7.5)


def decoder_only_block(g):
    cx = 32
    g.n("x", "x", cx, 94, 14, kind="io")
    g.n("ln1", "LayerNorm", cx, 82, 30)
    g.n("att", "Causal attention", cx, 70, 40, kind="param")
    g.n("a1", "+", cx, 58, 8, circle=True, fs=11)
    g.n("ln2", "LayerNorm", cx, 46, 30)
    g.n("ffn", "Feed-forward", cx, 34, 36, kind="param")
    g.n("a2", "+", cx, 22, 8, circle=True, fs=11)
    g.n("out", "out", cx, 9, 14, kind="io")
    for a, b in [("x", "ln1"), ("ln1", "att"), ("att", "a1"), ("a1", "ln2"),
                 ("ln2", "ffn"), ("ffn", "a2"), ("a2", "out")]:
        g.a((a, "b"), (b, "t"))
    g.p([("x", "r"), (60, 94), (60, 58), ("a1", "r")], kind="loss", dashed=True)
    g.p([(60, 58), (60, 22), ("a2", "r")], kind="loss", dashed=True)
    g.text(54, 40, "skip", "muted")
    fills = {(r, c): ("io" if c <= r else "loss") for r in range(5) for c in range(5)}
    g.grid(70, 82, 5, 5, 5.6, fills)
    g.text(84, 88, "causal mask", "muted", fs=7.5)
    g.text(84, 49, "no peeking at\nlater tokens", "muted", fs=7.5)


def feed_forward_gelu_swiglu(g):
    g.n("x", "x", 50, 73, 12, kind="io")
    # plain FFN, left
    g.n("w1", "x W1  (d→h)", 24, 60, 34, kind="param")
    g.n("ge", "GELU", 24, 45, 20)
    g.n("w2", "W2  (h→d)", 24, 29, 30, kind="param")
    g.n("o1", "out", 24, 14, 14, kind="io")
    g.a(("x", "b"), ("w1", "t"))
    g.a(("w1", "b"), ("ge", "t"))
    g.a(("ge", "b"), ("w2", "t"))
    g.a(("w2", "b"), ("o1", "t"))
    # gated, right
    g.n("A", "x W1", 62, 60, 18, kind="param")
    g.n("B", "x W3", 86, 60, 18, kind="param")
    g.n("si", "SiLU", 62, 47, 18)
    g.n("mul", "×", 74, 37, 8, circle=True, fs=11)
    g.n("w2b", "W2", 74, 26, 18, kind="param")
    g.n("o2", "out", 74, 14, 14, kind="io")
    g.a(("x", "b"), ("A", "t"))
    g.a(("x", "b"), ("B", "t"))
    g.a(("A", "b"), ("si", "t"))
    g.p([("si", "b"), (62, 37), ("mul", "l")])
    g.p([("B", "b"), (86, 37), ("mul", "r")])
    g.a(("mul", "b"), ("w2b", "t"))
    g.a(("w2b", "b"), ("o2", "t"))
    g.text(24, 5, "GELU: 2dh weights", "muted", fs=7.5)
    g.text(74, 5, "SwiGLU: 3dh, h ≈ 8d/3", "muted", fs=7.5)


def llama_rmsnorm_gqa(g):
    g.n("x", "x", 50, 76, 12, kind="io")
    g.n("nm", "RMSNorm: x / rms(x) · g", 50, 64, 48, kind="param")
    g.a(("x", "b"), ("nm", "t"))
    xs = [15, 38, 62, 85]
    for i, cx in enumerate(xs):
        g.n(f"q{i}", f"Q head {i + 1}", cx, 46, 20)
        g.a(("nm", "b"), (f"q{i}", "t"))
    g.n("kv0", "K, V head 1", 27, 24, 34, kind="hi")
    g.n("kv1", "K, V head 2", 73, 24, 34, kind="hi")
    g.a(("q0", "b"), ("kv0", "t"))
    g.a(("q1", "b"), ("kv0", "t"))
    g.a(("q2", "b"), ("kv1", "t"))
    g.a(("q3", "b"), ("kv1", "t"))
    g.text(50, 8, "4 query heads share 2 KV heads: half the cache", "muted", fs=7.2)


def sliding_window_attention(g):
    T, W, cell = 8, 3, 7
    x0, top = 32, 70
    fills = {}
    for i in range(T):
        for j in range(T):
            if j <= i and j > i - W:
                fills[(i, j)] = "hi" if i == j else "io"
    g.grid(x0, top, T, T, cell, fills)
    g.text(x0 + T * cell / 2, top + 4, "keys j →", "muted", fs=7.5)
    g.text(17, top - T * cell / 2, "query\ni ↓", "muted", fs=7.5)
    g.text(50, 8, "each token sees itself + 2 before (W = 3)", "text", fs=7.5)
    g.text(50, 2, "cost T·W, cache = last W keys", "muted", fs=7.5)


def mixture_of_experts_routing(g):
    g.n("x", "token x", 50, 88, 24, kind="io")
    g.n("r", "router  z = x Wr", 50, 76, 40, kind="param")
    g.a(("x", "b"), ("r", "t"))
    xs = [15, 38, 62, 85]
    for i, cx in enumerate(xs):
        on = i in (1, 3)
        g.n(f"e{i}", f"e{i + 1}\n" + ("active" if on else "idle"), cx, 54, 20, 11,
            kind="param" if on else "op")
    g.a(("r", "b"), ("e1", "t"))
    g.a(("r", "b"), ("e3", "t"))
    g.text(39, 66, "g2", "muted", fs=7.5)
    g.text(70, 66, "g4", "muted", fs=7.5)
    g.text(86, 76, "top-2\nof 4", "muted", fs=7.5)
    g.n("sum", "Σ gᵢ · eᵢ(x)", 50, 33, 36, kind="hi")
    g.a(("e1", "b"), ("sum", "t"))
    g.a(("e3", "b"), ("sum", "t"))
    g.n("y", "y", 50, 19, 12, kind="io")
    g.a(("sum", "b"), ("y", "t"))
    g.text(50, 8, "idle experts cost nothing; aux loss evens the load", "muted", fs=7)


def t5_span_corruption(g):
    def row(y, toks, kinds, centers):
        for i, (t, k, cx) in enumerate(zip(toks, kinds, centers)):
            g.n(f"{y}-{i}", t, cx, y, 12.5, 8, kind=k, fs=7)

    c7 = [8 + 14 * i for i in range(7)]
    g.text(2, 96, "input text", "muted", fs=7.5, ha="left")
    row(88, ["the", "cat", "sat", "on", "the", "mat", "today"],
        ["io", "loss", "loss", "io", "loss", "loss", "io"], c7)
    g.text(2, 72, "encoder input", "muted", fs=7.5, ha="left")
    c5 = [22, 36, 50, 64, 78]
    row(64, ["the", "<X0>", "on", "<X1>", "today"], ["io", "hi", "io", "hi", "io"], c5)
    g.a((50, 84), ("64-2", "t"))
    g.text(66, 76, "spans → sentinels", "muted", fs=7.5, ha="left")
    g.n("ed", "encoder → decoder", 50, 44, 44, kind="param")
    g.a(("64-2", "b"), ("ed", "t"))
    g.text(2, 31, "target (loss here only)", "muted", fs=7.5, ha="left")
    row(20, ["<X0>", "cat", "sat", "<X1>", "the", "mat", "</s>"],
        ["hi", "loss", "loss", "hi", "loss", "loss", "op"], c7)
    g.a(("ed", "b"), (50, 24))
    g.text(2, 6, "sentinels mark gaps; decoder rewrites the spans", "muted", fs=7.2, ha="left")


def flash_attention_online_softmax(g):
    cx = 36
    g.n("kv", "K, V block j", cx, 90, 44, kind="io")
    g.n("s", "scores s (T × B)", cx, 77, 44)
    g.n("m", "m_new = max(m, s)", cx, 64, 44, kind="param")
    g.n("rs", "ℓ, o × exp(m − m_new)", cx, 51, 44, kind="hi")
    g.n("ac", "add this block", cx, 38, 44, kind="param")
    g.n("out", "o / ℓ", cx, 22, 24, kind="io")
    for a, b in [("kv", "s"), ("s", "m"), ("m", "rs"), ("rs", "ac")]:
        g.a((a, "b"), (b, "t"))
    g.a(("ac", "b"), ("out", "t"))
    g.text(54, 30, "last block", "muted", fs=7.5, ha="left")
    g.n("st", "state\nm, ℓ, o", 84, 38, 24, 12, kind="cool")
    g.a(("ac", "r"), ("st", "l"))
    g.p([("st", "t"), (84, 90), ("kv", "r")], dashed=True, kind="loss")
    g.text(92, 66, "next\nblock", "muted", fs=7.5)
    g.text(50, 8, "exact softmax; T × T never stored", "muted", fs=7.5)


def speculative_decoding(g):
    g.n("dr", "draft q: proposes γ = 4", 50, 88, 84, kind="param")
    xs = [20, 40, 60, 80]
    for i, cx in enumerate(xs):
        g.n(f"t{i}", f"x{i + 1}", cx, 73, 16, kind="io")
        g.a((cx, 83.5), (f"t{i}", "t"))
    g.n("tg", "target p: scores all in 1 pass", 50, 57, 84, kind="param")
    for i, cx in enumerate(xs):
        g.a((f"t{i}", "b"), (cx, 62))
    marks = [("x1 ✓", "io"), ("x2 ✓", "io"), ("x3 ✗", "loss"), ("x4 skip", "op")]
    for i, (cx, (txt, k)) in enumerate(zip(xs, marks)):
        g.n(f"v{i}", txt, cx, 41, 16 if i < 3 else 18, kind=k, fs=8)
        g.a((cx, 52), (f"v{i}", "t"))
    g.n("rs", "resample from p − q", 60, 25, 40, kind="hi")
    g.a(("v2", "b"), ("rs", "t"))
    g.text(50, 11, "keep x1 x2, fix x3: 3 tokens per target pass", "muted", fs=7.2)


DIAGRAMS = {
    "gpt-decoder-only": (96, gpt_decoder_only),
    "decoder-only-block": (102, decoder_only_block),
    "feed-forward-gelu-swiglu": (80, feed_forward_gelu_swiglu),
    "llama-rmsnorm-gqa": (82, llama_rmsnorm_gqa),
    "sliding-window-attention": (82, sliding_window_attention),
    "mixture-of-experts-routing": (96, mixture_of_experts_routing),
    "t5-span-corruption": (100, t5_span_corruption),
    "flash-attention-online-softmax": (98, flash_attention_online_softmax),
    "speculative-decoding": (96, speculative_decoding),
}

if __name__ == "__main__":
    render(DIAGRAMS, sys.argv[1:] or None)
