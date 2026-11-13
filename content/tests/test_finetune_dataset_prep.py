import numpy as np, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "finetune-dataset-prep.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

# normalisation and exact dedup
assert ns["normalise"]("  Hello,   WORLD!! ") == "hello world"
texts = ["The cat sat.", "the  cat sat", "THE CAT SAT!!!", "A dog ran."]
assert len({ns["dedup_key"](t) for t in texts}) == 2

# MinHash estimate vs exact Jaccard on word shingles
rng = np.random.default_rng(1)
vocab = [f"w{i}" for i in range(300)]
base = list(rng.choice(vocab, 200))
other = base[:150] + list(rng.choice(vocab, 50))
ta, tb = " ".join(base), " ".join(other)
A, B = ns["shingle_set"](ta, 3), ns["shingle_set"](tb, 3)
exact = len(A & B) / len(A | B)
k = 2048
est = ns["estimate_jaccard"](ns["minhash_signature"](A, k, 7), ns["minhash_signature"](B, k, 7))
tol = 4 * np.sqrt(exact * (1 - exact) / k)
assert abs(est - exact) < tol, (est, exact, tol)
# identical sets give identical signatures; disjoint sets almost never match
sa = ns["minhash_signature"](A, k, 7)
assert ns["estimate_jaccard"](sa, ns["minhash_signature"](set(A), k, 7)) == 1.0
C = ns["shingle_set"](" ".join(rng.choice([f"z{i}" for i in range(300)], 200)), 3)
assert ns["estimate_jaccard"](sa, ns["minhash_signature"](C, k, 7)) < 0.02
# deterministic for a fixed seed
assert (ns["minhash_signature"](A, 64, 3) == ns["minhash_signature"](A, 64, 3)).all()

# group split: no group on both sides, all items kept
items = [{"id": i, "src": f"s{i % 17}"} for i in range(200)]
tr, va = ns["group_split"](items, lambda x: x["src"], 0.2, 0)
assert len(tr) + len(va) == 200 and len(va) >= 40
assert not ({x["src"] for x in tr} & {x["src"] for x in va})
assert ns["group_split"](items, lambda x: x["src"], 0.2, 0)[1] == va

# n-gram overlap flags a planted eval snippet
train = [" ".join(rng.choice(vocab, 60)) for _ in range(20)]
snippet = " ".join(train[4].split()[10:30])
eval_texts = [snippet, "completely different words appear here in this unseen sentence ok yes"]
ov = ns["ngram_overlap"](train, eval_texts, 8)
assert ov[0] == 1.0 and ov[1] == 0.0, ov
print("finetune-dataset-prep OK")
