import re, yaml
from collections import Counter
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "bpe-tokenization.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

# Classic corpus: counts low 5, lower 2, newest 6, widest 3
words = ["low"] * 5 + ["lower"] * 2 + ["newest"] * 6 + ["widest"] * 3
merges = ns["bpe_train"](words, 6)
# Hand derivation: (e,s) and (s,t) tie at 9 -> smaller pair (e,s); then
# (es,t)=9; then (l,o)=(o,w)=7 tie -> (l,o); then (lo,w)=7.
assert merges[:4] == [("e", "s"), ("es", "t"), ("l", "o"), ("lo", "w")]

# Independent string-based implementation (Sennrich et al. style)
def ref_train(words, k):
    vocab = Counter(" ".join(w) for w in words)
    out = []
    for _ in range(k):
        pc = Counter()
        for w, f in vocab.items():
            sy = w.split()
            for a, b in zip(sy, sy[1:]):
                pc[(a, b)] += f
        if not pc:
            break
        best = min(pc, key=lambda p: (-pc[p], p))
        pat = re.compile(r"(?<!\S)" + re.escape(" ".join(best)) + r"(?!\S)")
        vocab = Counter({pat.sub("".join(best), w): f for w, f in vocab.items()})
        out.append(best)
    return out

for k in (1, 3, 6, 10, 50):
    assert ns["bpe_train"](words, k) == ref_train(words, k), k
assert ns["bpe_encode"]("lowest", merges) == ["low", "est"]
assert ns["bpe_encode"]("newer", merges)[0] == "n"
assert "".join(ns["bpe_encode"]("zzz", merges)) == "zzz"
# merge counts stop once everything is one symbol
assert len(ns["bpe_train"](["ab"], 5)) == 1
print("bpe-tokenization OK")
