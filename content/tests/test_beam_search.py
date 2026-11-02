import itertools, numpy as np, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "beam-search.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

V, L = 4, 3
rng = np.random.default_rng(0)
T = rng.normal(size=(V + 1, V))          # row V is the start token
T = T - np.log(np.exp(T).sum(1, keepdims=True))   # log-softmax rows
step = lambda seq: T[seq[-1]]
BOS = V

def score(seq):
    prev, s = BOS, 0.0
    for t in seq:
        s += T[prev, t]
        prev = t
    return s

all_seqs = sorted(itertools.product(range(V), repeat=L), key=lambda q: -score(q))
best = ns["beam_search"](step, BOS, L, V ** L)
assert len(best) == V ** L
for (seq, sc), ref in zip(best, all_seqs):
    assert tuple(seq[1:]) == ref and np.isclose(sc, score(ref))
# probabilities of all full sequences sum to one
assert np.isclose(sum(np.exp(sc) for _, sc in best), 1.0)
# B=1 is greedy; wider beams never score worse than narrower ones here
g = ns["beam_search"](step, BOS, L, 1)[0]
seq, prev = [], BOS
for _ in range(L):
    prev = int(T[prev].argmax())
    seq.append(prev)
assert g[0][1:] == seq
tops = [ns["beam_search"](step, BOS, L, B)[0][1] for B in (1, 2, 4, 64)]
assert tops[-1] >= tops[0] and tops[-1] == max(score(q) for q in all_seqs)
assert all(t <= tops[-1] + 1e-12 for t in tops)
print("beam-search OK")
