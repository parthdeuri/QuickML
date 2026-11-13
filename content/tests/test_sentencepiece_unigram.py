import numpy as np, yaml
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "sentencepiece-unigram.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

freqs = {"▁low": 5, "▁lower": 2, "▁newest": 6, "▁widest": 3}
# initial vocabulary: all substrings, random probabilities summing to 1
subs = sorted({w[i:j] for w in freqs for i in range(len(w)) for j in range(i + 1, len(w) + 1)})
rng = np.random.default_rng(0)
r = rng.random(len(subs)) + 0.05
p = dict(zip(subs, r / r.sum()))
assert abs(sum(p.values()) - 1) < 1e-12
logp = {s: np.log(v) for s, v in p.items()}

def all_segs(w):
    if not w:
        yield []
        return
    for j in range(1, len(w) + 1):
        if w[:j] in p:
            for rest in all_segs(w[j:]):
                yield [w[:j]] + rest

for w in list(freqs) + ["▁lowest", "▁wide", "▁ne"]:
    segs = list(all_segs(w))
    scores = [sum(logp[t] for t in s) for s in segs]
    seg, sc = ns["viterbi_segment"](w, logp)
    assert abs(sc - max(scores)) < 1e-12, (w, sc, max(scores))
    assert "".join(seg) == w and abs(sum(logp[t] for t in seg) - sc) < 1e-12
    assert seg == segs[int(np.argmax(scores))]
assert ns["viterbi_segment"]("xyz", logp) == (None, -np.inf)

# expected counts equal brute-force posterior over all segmentations
newp, ll = ns["em_step"](freqs, p)
cnt = dict.fromkeys(p, 0.0)
ref_ll = 0.0
for w, f in freqs.items():
    segs = list(all_segs(w))
    probs = np.array([np.prod([p[t] for t in s]) for s in segs])
    ref_ll += f * np.log(probs.sum())
    for s, pr in zip(segs, probs / probs.sum()):
        for t in s:
            cnt[t] += f * pr
tot = sum(cnt.values())
assert abs(ll - ref_ll) < 1e-9
for s in p:
    assert abs(newp[s] - cnt[s] / tot) < 1e-12, s
assert abs(sum(newp.values()) - 1) < 1e-12

# EM never lowers the corpus likelihood
prev, cur = -np.inf, p
for _ in range(15):
    cur, ll = ns["em_step"](freqs, cur)
    assert ll >= prev - 1e-9, (ll, prev)
    prev = ll
assert abs(sum(cur.values()) - 1) < 1e-12

# prune tokens with probability 0 after EM, renormalise, still segmentable
kept = {s: v for s, v in cur.items() if v > 1e-3}
z = sum(kept.values())
kept = {s: v / z for s, v in kept.items()}
lp = {s: np.log(v) for s, v in kept.items()}
assert abs(sum(kept.values()) - 1) < 1e-12
for w in freqs:
    seg, _ = ns["viterbi_segment"](w, lp)
    assert seg is not None and "".join(seg) == w

# optional cross-check with the tokenizers library
try:
    from tokenizers import models
    vocab = [("<unk>", 0.0)] + [(s, float(l)) for s, l in logp.items()]
    uni = models.Unigram(vocab, unk_id=0)
    for w in freqs:
        ref = [t.value for t in uni.tokenize(w)]
        assert ref == ns["viterbi_segment"](w, logp)[0], (w, ref)
except ImportError:
    pass
print("sentencepiece-unigram OK")
