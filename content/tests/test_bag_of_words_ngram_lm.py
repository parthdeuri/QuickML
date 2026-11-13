import numpy as np, yaml
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "bag-of-words-ngram-lm.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

# bag of words equals sklearn CountVectorizer
docs = ["the cat sat on the mat", "the dog sat on the log", "the cat saw the dog"]
vocab = ns["vocab_of"](docs)
X = ns["bow"](docs, vocab)
cv = CountVectorizer()
ref = cv.fit_transform(docs).toarray()
assert list(cv.get_feature_names_out()) == vocab
assert np.array_equal(X, ref)

# bigram model, add-k smoothing
sents = ["the cat sat", "the dog sat"]
V = ns["vocab_of"](sents, ["<s>", "</s>"])
C = ns["bigram_counts"](sents, V)
for k in (1.0, 0.1):
    P = ns["bigram_prob"](C, k)
    assert np.allclose(P.sum(axis=1), 1.0)  # a distribution for every context
    assert (P > 0).all()

# hand-computed: V = 6, k = 1
P = ns["bigram_prob"](C, 1.0)
idx = {w: i for i, w in enumerate(V)}
assert np.isclose(P[idx["<s>"], idx["the"]], 3 / 8)
assert np.isclose(P[idx["the"], idx["cat"]], 2 / 8)
assert np.isclose(P[idx["cat"], idx["sat"]], 2 / 7)
assert np.isclose(P[idx["sat"], idx["</s>"]], 3 / 8)
lp = ns["sentence_logprob"]("the cat sat", P, V)
assert np.isclose(lp, np.log(3 / 8 * 2 / 8 * 2 / 7 * 3 / 8)), lp
print("bag-of-words-ngram-lm OK")
