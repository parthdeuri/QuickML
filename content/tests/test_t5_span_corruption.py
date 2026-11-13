import numpy as np, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "t5-span-corruption.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
corrupt, recon = ns["corrupt_spans"], ns["reconstruct"]

# hand example
toks = "thank you for inviting me to your party last week".split()
inp, tgt = corrupt(toks, [(2, 2), (7, 1)])
assert inp == ["thank", "you", "<extra_id_0>", "me", "to", "your",
               "<extra_id_1>", "last", "week"], inp
assert tgt == ["<extra_id_0>", "for", "inviting", "<extra_id_1>",
               "party", "</s>"], tgt

rng = np.random.default_rng(0)
for _ in range(200):
    n = int(rng.integers(10, 40))
    toks = [f"w{i}" for i in range(n)]
    k = int(rng.integers(1, 5))
    # random non-overlapping spans: sorted cut points, gap of >= 1 between
    pts = np.sort(rng.choice(np.arange(1, n), size=2 * k, replace=False))
    spans = []
    for a, b in zip(pts[::2], pts[1::2]):
        spans.append((int(a), int(b - a)))
    # drop spans touching the next one so they stay disjoint and separate
    spans = [s for i, s in enumerate(spans)
             if i == len(spans) - 1 or s[0] + s[1] < spans[i + 1][0]]
    inp, tgt = corrupt(toks, spans)
    assert recon(inp, tgt) == toks
    m, kk = sum(s[1] for s in spans), len(spans)
    assert len(tgt) == m + kk + 1
    assert len(inp) == n - m + kk
    assert [t for t in inp if t.startswith("<extra_id_")] == \
        [f"<extra_id_{i}>" for i in range(kk)]
    assert tgt[-1] == "</s>"
print("t5-span-corruption OK")
