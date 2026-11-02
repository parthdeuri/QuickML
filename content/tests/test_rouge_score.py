import numpy as np, yaml
from pathlib import Path
from rouge_score import rouge_scorer

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "rouge-score.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rouge_n, rouge_l = ns["rouge_n"], ns["rouge_l"]

# rouge-score lower-cases, strips non-alphanumerics, no stemmer here
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
rng = np.random.default_rng(0)
vocab = ["alpha", "beta", "gamma", "delta", "eps", "zeta", "eta", "theta", "iota", "kappa"]
for _ in range(60):
    ref = list(rng.choice(vocab, size=rng.integers(5, 25)))
    hyp = list(rng.choice(vocab, size=rng.integers(3, 25)))
    ref_s, hyp_s = " ".join(ref), " ".join(hyp)
    ref_scores = scorer.score(ref_s, hyp_s)       # (target, prediction)
    for name, mine in (("rouge1", rouge_n(hyp, ref, 1)),
                       ("rouge2", rouge_n(hyp, ref, 2)),
                       ("rougeL", rouge_l(hyp, ref))):
        s = ref_scores[name]
        assert np.allclose(mine, (s.precision, s.recall, s.fmeasure)), (name, mine, s)

# hand-checked: ref "the cat sat on the mat", hyp "the cat on the mat"
ref, hyp = "the cat sat on the mat".split(), "the cat on the mat".split()
p, r, f = rouge_n(hyp, ref, 1)
assert (p, r) == (1.0, 5 / 6)
p, r, f = rouge_n(hyp, ref, 2)            # shared bigrams: the-cat, on-the, the-mat
assert (p, r) == (3 / 4, 3 / 5)
assert ns["lcs_len"](hyp, ref) == 5
assert ns["lcs_len"]([], ref) == 0 and rouge_n([], ref, 1) == (0.0, 0.0, 0.0)
# clipping: repeated hypothesis word gets credit only once per reference occurrence
assert rouge_n(["the"] * 4, ["the", "cat"], 1)[0] == 1 / 4
print("rouge-score OK")
