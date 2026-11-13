import numpy as np
import yaml
d = yaml.safe_load(open("content/patterns/compare-bm25-vs-dense-retrieval.yaml"))
ns = {}
exec(d["template"], ns)
bm25, cosine, rrf = ns["bm25"], ns["cosine"], ns["rrf"]
# exact code: BM25 finds doc 0, the blurry embedding prefers doc 2
q = ["e4021"]
assert bm25(q).argmax() == 0 and cosine(q).argmax() != 0
# paraphrase: BM25 scores zero everywhere, dense finds doc 1
q = ["internet"]
assert not bm25(q).any() and cosine(q).argmax() == 1
# fusion: for the code query the fused top is the BM25 winner or tied
q = ["e4021"]
f = rrf(bm25(q), cosine(q))
assert f.argmax() == 0
q = ["internet"]
f = rrf(bm25(q), cosine(q))
assert f.argmax() == 1
print("pattern compare-bm25-vs-dense-retrieval OK")
