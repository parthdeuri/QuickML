import math, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "document-chunking.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
fixed_chunks, chunk_count, recursive_split = ns["fixed_chunks"], ns["chunk_count"], ns["recursive_split"]

# hand-computed example: n=10, size=4, overlap=1 -> stride 3
assert fixed_chunks(list(range(10)), 4, 1) == [[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9]]
assert chunk_count(10, 4, 1) == math.ceil(9 / 3) == 3

for size in range(2, 12):
    for overlap in range(0, size):
        for n in range(1, 60):
            toks = list(range(n))
            ch = fixed_chunks(toks, size, overlap)
            assert all(1 <= len(c) <= size for c in ch)
            for a, b in zip(ch, ch[1:]):
                assert len(a) == size
                assert (a[-overlap:] if overlap else []) == b[:overlap]
            rebuilt = ch[0] + [x for c in ch[1:] for x in c[overlap:]]
            assert rebuilt == toks, (n, size, overlap)
            assert chunk_count(n, size, overlap) == len(ch), (n, size, overlap)

text = ("Intro paragraph about retrieval. It has two sentences.\n\n"
        "Second paragraph is longer. " + "word " * 40 + "end.\nA line.\n\n"
        + "x" * 55 + "\n\nLast one.")
for m in (10, 20, 30, 64, 200):
    parts = recursive_split(text, m)
    assert all(len(p) <= m for p in parts), m
    assert "".join(parts) == text, m
# paragraph boundaries are preferred when they fit
two = "aaaa\n\nbbbb"
assert recursive_split(two, 6) == ["aaaa\n\n", "bbbb"]
assert recursive_split(two, 100) == [two]
print("document-chunking OK")
