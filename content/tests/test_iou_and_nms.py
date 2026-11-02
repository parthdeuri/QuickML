import numpy as np, yaml, torch, torchvision
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "iou-and-nms.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)

# hand-computed exact cases
a = np.array([[0.0, 0, 2, 2]])
b = np.array([[1.0, 1, 3, 3], [0, 0, 2, 2], [5, 5, 6, 6], [0, 0, 1, 2]])
assert np.allclose(ns["iou"](a, b), [[1 / 7, 1.0, 0.0, 0.5]])

rng = np.random.default_rng(0)
def rand_boxes(n):
    xy = rng.uniform(0, 50, (n, 2))
    wh = rng.uniform(5, 30, (n, 2))
    return np.hstack([xy, xy + wh])

A, B = rand_boxes(12), rand_boxes(9)
ref = torchvision.ops.box_iou(torch.tensor(A), torch.tensor(B)).numpy()
assert np.allclose(ns["iou"](A, B), ref)

for thr in (0.1, 0.3, 0.5, 0.7):
    boxes = rand_boxes(60)
    scores = rng.permutation(60).astype(float)   # distinct scores, no ties
    keep = ns["nms"](boxes, scores, thr)
    rk = torchvision.ops.nms(torch.tensor(boxes), torch.tensor(scores), thr).tolist()
    assert [int(i) for i in keep] == rk, thr
print("iou-and-nms OK")
