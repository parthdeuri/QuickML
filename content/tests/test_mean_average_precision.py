"""AP convention under test: all-point interpolated AP (VOC2010+ style), single
IoU threshold, each detection matched to the best still-unmatched GT (COCO style).

References:
  * torchmetrics MeanAveragePrecision at IoU 0.5 with a dense recall grid
    (COCO uses 101 points; 10001 points approximates all-point AP to ~1e-4).
  * an independent sklearn-based all-point AP: TP flags are computed by a
    separate matcher, then precision_recall_curve on the (score, TP) list with
    recall rescaled from matched GTs to all GTs; AP is the area under its
    monotone precision envelope.
"""
import numpy as np, yaml, torch
from pathlib import Path

card = yaml.safe_load((Path(__file__).parent.parent / "problems" / "mean-average-precision.yaml").read_text(encoding="utf-8"))
ns = {}
exec(card["code"], ns)
rng = np.random.default_rng(0)

# ---- hand-built scene: 3 classes, 6 images -------------------------------
n_img, classes = 6, [0, 1, 2]
gts = {c: {} for c in classes}
dets = {c: [] for c in classes}


def rand_box():
    x, y = rng.uniform(0, 60, 2)
    w, h = rng.uniform(20, 40, 2)
    return np.array([x, y, x + w, y + h])


for c in classes:
    for img in range(n_img):
        k = rng.integers(0, 3)
        boxes = [rand_box() for _ in range(k)]
        if boxes:
            gts[c][img] = boxes
        for b in boxes:            # most GTs detected with jitter; some missed
            if rng.random() < 0.8:
                dets[c].append((img, rng.uniform(0.3, 1.0), b + rng.normal(0, 3, 4)))
        for _ in range(rng.integers(1, 4)):   # false positives / duplicates
            dets[c].append((img, rng.uniform(0.0, 0.8), rand_box()))
        if boxes and rng.random() < 0.5:      # duplicate on a GT
            dets[c].append((img, rng.uniform(0.2, 0.9), boxes[0] + rng.normal(0, 2, 4)))
    gts[c] = {k: v for k, v in gts[c].items() if v}

ap = ns["average_precision"]
m = ns["mean_ap"](dets, gts, 0.5)

# ---- 1) independent sklearn-based all-point AP ---------------------------
from sklearn.metrics import precision_recall_curve


def iou_ref(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua


def sk_ap(d, g, thr=0.5):
    n_gt = sum(len(v) for v in g.values())
    taken = {i: set() for i in g}
    flags, scores = [], []
    for img, s, box in sorted(d, key=lambda t: -t[1]):
        cands = [(iou_ref(box, gb), j) for j, gb in enumerate(g.get(img, [])) if j not in taken[img]]
        hit = 0
        if cands:
            best, j = max(cands)
            if best >= thr:
                taken[img].add(j)
                hit = 1
        flags.append(hit)
        scores.append(s)
    if sum(flags) == 0:
        return 0.0
    prec, rec, _ = precision_recall_curve(np.array(flags), np.array(scores))
    rec = rec * sum(flags) / n_gt      # sklearn recall is over matched GTs; rescale to all GTs
    prec = np.maximum.accumulate(prec)               # envelope: max precision at recall >= r
    return float(np.sum(-np.diff(rec) * prec[:-1]))


for c in classes:
    a, b = ap(dets[c], gts[c]), sk_ap(dets[c], gts[c])
    assert np.isclose(a, b), (c, a, b)
assert np.isclose(m, np.mean([sk_ap(dets[c], gts[c]) for c in classes]))

# ---- 2) torchmetrics MeanAveragePrecision --------------------------------
try:
    from torchmetrics.detection import MeanAveragePrecision
    metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5],
                                  rec_thresholds=np.linspace(0, 1, 10001).tolist(),
                                  class_metrics=True)
    preds, targets = [], []
    for img in range(n_img):
        pb, ps, pl, tb, tl = [], [], [], [], []
        for c in classes:
            for i, s, b in dets[c]:
                if i == img:
                    pb.append(b); ps.append(s); pl.append(c)
            for b in gts[c].get(img, []):
                tb.append(b); tl.append(c)
        preds.append(dict(boxes=torch.tensor(np.array(pb).reshape(-1, 4), dtype=torch.float32),
                          scores=torch.tensor(ps, dtype=torch.float32),
                          labels=torch.tensor(pl, dtype=torch.int64)))
        targets.append(dict(boxes=torch.tensor(np.array(tb).reshape(-1, 4), dtype=torch.float32),
                            labels=torch.tensor(tl, dtype=torch.int64)))
    metric.update(preds, targets)
    res = metric.compute()
    per_cls = res["map_per_class"].numpy()
    for c in classes:
        assert abs(per_cls[c] - ap(dets[c], gts[c])) < 2e-3, (c, per_cls[c], ap(dets[c], gts[c]))
    assert abs(float(res["map"]) - m) < 2e-3, (float(res["map"]), m)
    print("  (also matched torchmetrics MeanAveragePrecision at IoU 0.5, 10001-pt recall grid)")
except ImportError:
    print("  torchmetrics not installed; used sklearn cross-check only")

# ---- 3) sanity cases ------------------------------------------------------
g1 = {0: [np.array([0., 0, 10, 10])]}
assert ap([(0, 0.9, np.array([0., 0, 10, 10]))], g1) == 1.0
assert ap([], g1) == 0.0
assert ap([(0, 0.9, np.array([50., 50, 60, 60]))], g1) == 0.0
# a duplicate on an already matched GT is a false positive but AP stays 1 (it ranks last)
assert ap([(0, 0.9, g1[0][0]), (0, 0.5, g1[0][0])], g1) == 1.0
# ...but ranking the duplicate first costs nothing either; a miss ranked first does:
assert np.isclose(ap([(0, 0.9, np.array([50., 50, 60, 60])), (0, 0.5, g1[0][0])], g1), 0.5)
print("mean-average-precision OK")
