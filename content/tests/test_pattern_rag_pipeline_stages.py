import yaml, numpy as np
d = yaml.safe_load(open("content/patterns/rag-pipeline-stages.yaml", encoding="utf-8"))
assert d["id"] == "rag-pipeline-stages"
ns = {}
exec(d["template"], ns)
ch = ns["chunks"](list(range(10)), 4, 1)
assert ch[0] == [0, 1, 2, 3] and ch[1][0] == 3 and ch[-1][-1] == 9
D = np.array([[1.0, 0], [0, 1.0], [1.0, 1.0]])
assert ns["retrieve"](np.array([5.0, 0]), D, 2) == [0, 2]
fused = ns["rrf"]([[1, 2, 3], [2, 1, 4]])
assert fused[:2] == [1, 2] and set(fused) == {1, 2, 3, 4}
assert ns["rrf"]([[7, 8], [8, 7, 9]])[0] == 7
assert ns["recall_at_k"]([3, 1, 2], [1, 2], 2) == 0.5
assert ns["recall_at_k"]([3, 1, 2], [1, 2], 3) == 1.0
print("pattern rag-pipeline-stages OK")
