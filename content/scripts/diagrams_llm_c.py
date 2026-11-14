"""Card diagrams (RAG retrieval, multimodal, CTC, audio). Run: python content/scripts/diagrams_llm_c.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagram_lib import G, render  # noqa: E402


def document_chunking(g):
    cell, x0, n = 6, 8, 14
    g.text(50, 66, "document: 14 tokens", "muted")
    g.grid(x0, 62, 1, n, cell, {(0, c): "op" for c in range(n)})
    size, stride = 6, 4
    ys = [44, 33, 22]
    for i, y in enumerate(ys):
        s = i * stride
        g.n(f"c{i}", "", x0 + (s + size / 2) * cell, y, size * cell, 8, kind="io")
        g.text(4, y, str(i + 1), "muted")
    # overlap strips (3 tokens shared by neighbouring chunks)
    for i in range(2):
        s = (i + 1) * stride
        w = (size - stride) * cell
        cx = x0 + s * cell + w / 2
        g.n(f"o{i}a", "", cx, ys[i + 1], w, 8, kind="hi")
        g.n(f"o{i}b", "", cx, ys[i], w, 8, kind="hi")
    g.text(50, 11, "size 6 · overlap 2 · stride = 6 − 2 = 4", "text", fs=7.5)
    g.text(50, 4, "yellow = tokens shared by neighbours", "muted", fs=7.5)


def ann_vector_index(g):
    g.text(34, 76, "IVF: 12 k-means cells", "muted")
    fills, texts = {}, {}
    for r in range(3):
        for c in range(4):
            texts[(r, c)] = f"c{r * 4 + c + 1}"
            fills[(r, c)] = "hi" if (r == 1 and c in (2, 3)) else "op"
    g.grid(8, 72, 3, 4, 13, fills, texts)
    g.n("q", "query", 84, 52.5, 18, kind="io")
    g.a(("q", "l"), (60, 52.5))
    g.text(84, 62, "nearest\ncentroids", "muted", fs=7.5)
    g.text(80, 40, "nprobe = 2", "muted", fs=7.5)
    g.n("scan", "scan only the\n2 yellow cells", 34, 22, 44, 11, kind="cool")
    g.a((34, 33), ("scan", "t"))
    g.n("out", "exact top-k of\nscanned vectors", 34, 7, 44, 11, kind="io")
    g.a(("scan", "b"), ("out", "t"))


def hybrid_search_rrf(g):
    g.n("q", "query", 50, 84, 18, kind="io")
    g.n("bm", "BM25", 24, 72, 24, kind="param")
    g.n("de", "dense", 76, 72, 24, kind="param")
    g.a(("q", "b"), ("bm", "t"))
    g.a(("q", "b"), ("de", "t"))
    L, R = "ACBD", "BAEC"
    for i in range(4):
        y = 64 - 8 * i - 4
        g.text(14, y, str(i + 1), "muted")
        g.text(37, y, f"1/{61 + i}", "muted", fs=7)
        g.text(66, y, str(i + 1), "muted")
        g.text(89, y, f"1/{61 + i}", "muted", fs=7)
    g.grid(18, 64, 4, 1, 8, {(r, 0): "io" for r in range(4)}, {(r, 0): L[r] for r in range(4)})
    g.grid(70, 64, 4, 1, 8, {(r, 0): "cool" for r in range(4)}, {(r, 0): R[r] for r in range(4)})
    g.n("fuse", "fuse: Σ 1/(60 + rank)", 50, 20, 52, 9, kind="hi")
    g.a((22, 32), ("fuse", "t"))
    g.a((74, 32), ("fuse", "t"))
    g.n("out", "A  B  C  E  D", 50, 7, 34, 8, kind="io")
    g.a(("fuse", "b"), ("out", "t"))


def reranking_mmr(g):
    cx = 38
    g.n("c", "100 candidates", cx, 74, 34, kind="io")
    g.n("x", "cross-encoder rerank", cx, 62, 42, kind="param")
    g.n("m", "pick next:\nλ·rel − (1−λ)·overlap", cx, 46, 48, 13, kind="hi")
    g.n("s", "selected set S", cx, 28, 32, kind="io")
    g.n("o", "top few (diverse)", cx, 12, 36, kind="io")
    g.a(("c", "b"), ("x", "t"))
    g.a(("x", "b"), ("m", "t"))
    g.a(("m", "b"), ("s", "t"))
    g.a(("s", "b"), ("o", "t"))
    g.p([("s", "r"), (74, 28), (74, 46), ("m", "r")], dashed=True, kind="loss")
    g.text(86, 37, "repeat\nuntil k", "muted", fs=7.5)
    g.text(78, 68, "joint\nquery+doc\nscore", "muted", fs=7.5)
    g.text(84, 12, "overlap = max\nsim to S", "muted", fs=7)


def query_transformation_hyde(g):
    cx = 36
    ys = [84, 70, 56, 42, 28, 14]
    g.n("q", "question q", cx, ys[0], 30, kind="io")
    g.n("llm", "LLM writes an answer", cx, ys[1], 44, kind="param")
    g.n("d", "hypothetical answer", cx, ys[2], 38, kind="cool")
    g.n("e", "encoder", cx, ys[3], 24, kind="param")
    g.n("v", "embedding v", cx, ys[4], 30, kind="hi")
    g.n("r", "nearest real docs", cx, ys[5], 36, kind="io")
    for a, b in [("q", "llm"), ("llm", "d"), ("d", "e"), ("e", "v"), ("v", "r")]:
        g.a((a, "b"), (b, "t"))
    g.text(80, 56, "looks like a\ndocument", "muted", fs=7.5)
    g.text(80, 70, "facts may\nbe wrong", "muted", fs=7.5)
    g.text(80, 14, "real facts\nfrom here", "muted", fs=7.5)


def vlm_projector_image_tokens(g):
    g.grid(21, 94, 2, 2, 9, {(r, c): "io" for r in range(2) for c in range(2)})
    g.text(66, 85, "image cut into\nP×P patches", "muted", fs=7.5)
    g.n("enc", "vision encoder", 30, 68, 36, kind="param")
    g.n("pr", "projector (MLP)", 30, 55, 38, kind="param")
    g.a((30, 76), ("enc", "t"))
    g.a(("enc", "b"), ("pr", "t"))
    g.text(54, 61.5, "d_v → d_l", "muted", fs=7.5, ha="left")
    fills = {(0, c): ("hi" if c < 4 else "io") for c in range(8)}
    texts = {(0, c): ("img" if c < 4 else "t") for c in range(8)}
    g.grid(18, 44, 1, 8, 8, fills, texts)
    g.a((26, 50.5), (26, 44))
    g.text(34, 32, "image tokens", "muted", fs=7.5)
    g.text(66, 32, "text tokens", "muted", fs=7.5)
    g.n("llm", "LLM reads one sequence", 50, 17, 52, kind="param")
    g.a((50, 36), ("llm", "t"))


def multimodal_rag_fusion(g):
    g.n("q", "query", 50, 86, 18, kind="io")
    g.n("ti", "text index", 24, 74, 30, kind="param")
    g.n("ii", "image index", 76, 74, 30, kind="param")
    g.a(("q", "b"), ("ti", "t"))
    g.a(("q", "b"), ("ii", "t"))
    ts, isc = ["12.1", "9.4", "7.0"], ["0.82", "0.74", "0.31"]
    g.grid(12, 64, 3, 1, 9, {(r, 0): "io" for r in range(3)}, {(r, 0): f"T{r + 1}" for r in range(3)})
    g.grid(62, 64, 3, 1, 9, {(r, 0): "cool" for r in range(3)}, {(r, 0): f"I{r + 1}" for r in range(3)})
    for r in range(3):
        g.text(27, 64 - 9 * r - 4.5, ts[r], "muted", fs=7.5)
        g.text(82, 64 - 9 * r - 4.5, isc[r], "muted", fs=7.5)
    g.text(49, 49, "unrelated\nscales", "muted", fs=7.5)
    g.n("norm", "normalise: min-max or rank", 50, 26, 56, kind="param")
    g.a((16.5, 37), ("norm", "t"))
    g.a((66.5, 37), ("norm", "t"))
    g.n("fuse", "weighted sum → top-k", 50, 11, 46, kind="hi")
    g.a(("norm", "b"), ("fuse", "t"))


def ctc_greedy_decoding(g):
    g.text(50, 82, "argmax label at each frame", "muted")
    row = "aa-b-bb"
    g.grid(15, 76, 1, 7, 10,
           {(0, c): ("op" if ch == "-" else "io") for c, ch in enumerate(row)},
           {(0, c): ("–" if ch == "-" else ch) for c, ch in enumerate(row)}, fs=9)
    r2 = "a-b-b"
    g.grid(25, 50, 1, 5, 10,
           {(0, c): ("op" if ch == "-" else "io") for c, ch in enumerate(r2)},
           {(0, c): ("–" if ch == "-" else ch) for c, ch in enumerate(r2)}, fs=9)
    g.grid(35, 26, 1, 3, 10, {(0, c): "hi" for c in range(3)},
           {(0, 0): "a", (0, 1): "b", (0, 2): "b"}, fs=9)
    g.a((50, 66), (50, 50))
    g.text(54, 58, "1 merge repeats", "muted", fs=7.5, ha="left")
    g.a((50, 40), (50, 26))
    g.text(54, 33, "2 drop blanks", "muted", fs=7.5, ha="left")
    g.text(24, 21, "output", "muted", fs=7.5)
    g.text(50, 8, "a blank between equal letters keeps both", "muted", fs=7.5)


def audio_log_mel_spectrogram(g):
    cx = 28
    ys = [86, 72, 58, 44, 30]
    labels = ["waveform", "overlapping frames", "FFT → power", "mel filterbank", "log"]
    kinds = ["io", "op", "op", "op", "op"]
    for i, (y, t, k) in enumerate(zip(ys, labels, kinds)):
        g.n(f"n{i}", t, cx, y, 40, kind=k)
    for i in range(4):
        g.a((f"n{i}", "b"), (f"n{i + 1}", "t"))
    g.text(76, 72, "hop < frame:\nframes overlap", "muted", fs=7.5)
    g.text(76, 58, "Hann window\nthen real FFT", "muted", fs=7.5)
    fills = {}
    for r in range(5):
        for c in range(6):
            fills[(r, c)] = "hi" if r >= 3 else ("cool" if r >= 1 else "op")
    g.grid(60, 42, 5, 6, 6, fills)
    g.a(("n4", "r"), (60, 30))
    g.text(78, 47, "mel bins", "muted", fs=7.5)
    g.text(78, 7, "time frames →", "muted", fs=7.5)


DIAGRAMS = {
    "document-chunking": (70, document_chunking),
    "ann-vector-index": (80, ann_vector_index),
    "hybrid-search-rrf": (92, hybrid_search_rrf),
    "reranking-mmr": (84, reranking_mmr),
    "query-transformation-hyde": (94, query_transformation_hyde),
    "vlm-projector-image-tokens": (100, vlm_projector_image_tokens),
    "multimodal-rag-fusion": (92, multimodal_rag_fusion),
    "ctc-greedy-decoding": (86, ctc_greedy_decoding),
    "audio-log-mel-spectrogram": (94, audio_log_mel_spectrogram),
}

if __name__ == "__main__":
    render(DIAGRAMS, sys.argv[1:] or None)
