import numpy as np, yaml
from pathlib import Path
from sklearn.metrics import average_precision_score

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "rag-evaluation-faithfulness.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

rng = np.random.default_rng(0)
for _ in range(200):
    k = int(rng.integers(1, 12))
    rel = rng.integers(0, 2, k)
    if rel.sum() == 0:
        assert ns["context_precision"](rel) == 0.0
        continue
    scores = np.arange(k, 0, -1)  # strictly descending rank scores
    ref = average_precision_score(rel, scores)
    assert np.isclose(ns["context_precision"](rel), ref), (rel, ref)
# hand case: [1,0,1] -> (1/1 + 2/3)/2 = 5/6 ; normalised by #relevant
assert np.isclose(ns["context_precision"]([1, 0, 1]), 5 / 6)
assert ns["context_precision"]([1, 1, 0, 0]) == 1.0
assert np.isclose(ns["context_precision"]([0, 0, 1]), 1 / 3)

# recall hand case: 2 of 3 facts present (case-insensitive)
ctx = "Paris is the capital of France. The Seine flows through it."
assert np.isclose(ns["context_recall"](["capital of france", "the seine", "eiffel tower"], ctx), 2 / 3)
assert ns["context_recall"](["PARIS"], ctx) == 1.0

# faithfulness: verbatim claims -> 1, no overlap -> 0, mixed -> fraction
assert ns["faithfulness"](["Paris is the capital of France", "The Seine flows through it"], ctx) == 1.0
assert ns["faithfulness"](["quantum entanglement spooky", "banana smoothie recipe"], ctx) == 0.0
assert ns["faithfulness"](["Paris is the capital of France", "banana smoothie recipe"], ctx) == 0.5
assert ns["faithfulness"](["x y z"], ctx, support_fn=lambda c, t: True) == 1.0

rep = ns["rag_report"]([1, 0, 1], ["the seine"], ctx, ["Paris is the capital of France"],
                       "What is the capital of France", "Paris is the capital of France")
assert np.isclose(rep["context_precision"], 5 / 6) and rep["context_recall"] == 1.0
assert rep["faithfulness"] == 1.0 and rep["answer_relevance"] == 1.0
print("rag-evaluation-faithfulness OK")
