import numpy as np, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "few-shot-prompting.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

rng = np.random.default_rng(0)
N, d, k = 40, 8, 5
E = rng.normal(size=(N, d))
q = rng.normal(size=d)
# brute-force top-k by cosine
cos = np.array([e @ q / (np.linalg.norm(e) * np.linalg.norm(q)) for e in E])
ref = sorted(range(N), key=lambda i: (-cos[i], i))[:k]
got = ns["select_examples"](q, E, k)
assert list(got) == ref, (got, ref)
# deterministic tie-break: duplicate vectors keep index order
E2 = np.vstack([E[3], E[3], E[3], E[:5]])
assert list(ns["select_examples"](E[3], E2, 3)) == [0, 1, 2]
# scale invariance of cosine
assert list(ns["select_examples"](5 * q, 7 * E, k)) == ref

# prompt: exactly k example blocks in order, ends awaiting the answer
pool = [(f"text{i}", "pos" if i % 2 else "neg") for i in range(N)]
ex = [pool[i] for i in reversed(ref)]  # most similar last
p = ns["build_prompt"]("  Classify sentiment.  ", ex, "new text")
blocks = p.split("\n\n")
assert blocks[0] == "Classify sentiment."
assert len(blocks) == 1 + k + 1
for b, (x, y) in zip(blocks[1:-1], ex):
    assert b == f"Input: {x}\nLabel: {y}"
assert blocks[-1] == "Input: new text\nLabel:"
assert p.endswith("Label:") and p.count("Label:") == k + 1

# label balance
c, imb = ns["label_balance"](["a", "a", "a", "b"])
assert c == {"a": 3, "b": 1} and imb == 0.75
print("few-shot-prompting OK")
