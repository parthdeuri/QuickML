import numpy as np, yaml, sacrebleu
from pathlib import Path
from nltk.translate.bleu_score import corpus_bleu as nltk_bleu

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "bleu-score.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
bleu = ns["corpus_bleu"]
rng = np.random.default_rng(0)
vocab = [f"w{i}" for i in range(12)]    # small vocabulary => n-gram overlap

def make_corpus(n_sent, drop):
    refs, hyps = [], []
    for _ in range(n_sent):
        r = list(rng.choice(vocab, size=rng.integers(8, 20)))
        h = [w if rng.random() > 0.15 else rng.choice(vocab) for w in r]
        if drop:                         # shorter hypothesis => brevity penalty
            h = h[: max(4, len(h) - rng.integers(0, 6))]
        refs.append(r); hyps.append(h)
    return hyps, refs

for drop in (False, True):
    hyps, refs = make_corpus(40, drop)
    mine = bleu(hyps, refs)
    # sacreBLEU: tokenize='none' (input already split on spaces), no smoothing
    sb = sacrebleu.corpus_bleu([" ".join(h) for h in hyps],
                               [[" ".join(r) for r in refs]],
                               tokenize="none", smooth_method="none").score
    assert abs(mine - sb) < 1e-8, (mine, sb)
    nl = 100 * nltk_bleu([[r] for r in refs], hyps)   # default 4-gram, no smoothing
    assert abs(mine - nl) < 1e-8, (mine, nl)

# hand-checked cases
h = "the cat sat on the mat today".split()
assert abs(bleu([h], [h]) - 100) < 1e-9                       # identical => 100
assert bleu([["a", "b", "c", "d", "e"]], [["x", "y", "z", "u", "v"]]) == 0.0
# clipping: 'the the the the' vs 'the cat' has unigram precision 1/4 (not 4/4)
rm = ns["ngrams"](["the"] * 4, 1)[("the",)]
assert rm == 4 and min(rm, ns["ngrams"](["the", "cat"], 1)[("the",)]) == 1
# brevity penalty: a prefix of the reference has all precisions 1, BLEU = 100*exp(1-r/c)
ref = list("abcdefghij")
prefix = ref[:5]
assert abs(bleu([prefix], [ref]) - 100 * np.exp(1 - 10 / 5)) < 1e-9
print("bleu-score OK")
